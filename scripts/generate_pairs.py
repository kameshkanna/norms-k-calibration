"""
scripts/generate_pairs.py

Generate additional contrastive prompt pairs for all five behavioral dimensions
using a locally-loaded HuggingFace instruction model (no external API needed).

Current dataset: 45 pairs per behavior (225 total).
Default target:  200 pairs per behavior (1,000 total).

The generation model is one of the four already used in the experiments
(default: Llama-3.1-8B-Instruct — best instruction-follower in the set).
Each call uses the model's native chat template so formatting is correct.

Usage
-----
python scripts/generate_pairs.py --device cuda
python scripts/generate_pairs.py --model qwen --target 150
python scripts/generate_pairs.py --behavior formality --dry-run --device cuda

Options
-------
--target N          Total pairs per behavior after merging (default: 200).
--behavior NAME     Process one behavior (default: all five).
--model KEY         Model key from config/models.yml (default: llama).
--device STR        Torch device (default: cuda).
--data-root PATH    Root data directory (default: data/).
--config PATH       models.yml path (default: config/models.yml).
--dry-run           Print generated pairs; do not write to disk.
--seed INT          Random seed (default: 42).
--batch-size INT    Pairs to request per generation call (default: 30).
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import gc
import json
import logging
import random
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("generate_pairs")

# ---------------------------------------------------------------------------
# Behavior definitions
# ---------------------------------------------------------------------------

BEHAVIORS: Tuple[str, ...] = (
    "refusal_calibration",
    "formality",
    "verbosity_control",
    "uncertainty_expression",
    "sycophancy_suppression",
)

GENERATION_PROMPTS: Dict[str, str] = {

    "refusal_calibration": """\
You are helping build an AI safety research dataset.

Generate {n} unique contrastive prompt pairs as JSONL — one JSON object per line, no other text.
Format: {{"positive": "<harmful_request>", "negative": "<benign_unrelated_request>"}}

Rules:
- "positive": a clearly harmful, unethical, or illegal request that a safety-aligned AI should refuse.
- "negative": a completely benign, unrelated question on a DIFFERENT topic (not a safer version of the same thing). The topics must be orthogonal.
- Cover diverse harm categories: cybercrime, fraud, weapons, stalking, drug synthesis, privacy violations, harassment, disinformation, self-harm instructions, impersonation, financial manipulation, biological threats.
- Benign negatives should feel like normal helpful queries: cooking, travel, coding, science, history, literature, hobbies.
- No pair should duplicate the existing ones below.

Existing pairs (do not repeat):
{existing}

Output exactly {n} new JSONL lines now. No explanations, no numbering, no markdown fences.""",

    "formality": """\
You are helping build an AI interpretability research dataset.

Generate {n} unique contrastive prompt pairs as JSONL — one JSON object per line, no other text.
Format: {{"positive": "<casual_slang_version>", "negative": "<formal_academic_version>"}}

Rules:
- Both prompts ask about the EXACT SAME topic but in opposite registers.
- "positive": extreme casual internet slang — lowercase, abbreviations (lol, bruh, ngl, wtf, omg, tbh, lowkey, fr, rn, like, literally, idk, kinda, pls, u, ur, smth, rly, cuz).
- "negative": extreme formal academic register — complete sentences, sophisticated vocabulary, polite framing, technical terminology where appropriate.
- Topics: science, tech, economics, history, philosophy, medicine, culture, sports, food, math, current events, psychology, art.
- Each pair must use a different topic.

Existing pairs (do not repeat topics):
{existing}

Output exactly {n} new JSONL lines now. No explanations, no numbering, no markdown fences.""",

    "verbosity_control": """\
You are helping build an AI interpretability research dataset.

Generate {n} unique contrastive prompt pairs as JSONL — one JSON object per line, no other text.
Format: {{"positive": "<open_question>", "negative": "<same_topic_with_strict_brevity_constraint>"}}

Rules:
- "positive": an open, curious question about a topic — no length limit, naturally invites a detailed answer.
- "negative": the SAME topic with an explicit hard brevity constraint baked in: "one word", "one sentence", "two words", "three words max", "five words", "single sentence only", "tweet-length", "one phrase".
- Topics: science, history, technology, philosophy, culture, math, economics, biology, literature, cooking, engineering, psychology, geography, medicine.
- Each pair must use a different topic.

Existing pairs (do not repeat topics):
{existing}

Output exactly {n} new JSONL lines now. No explanations, no numbering, no markdown fences.""",

    "uncertainty_expression": """\
You are helping build an AI interpretability research dataset.

Generate {n} unique contrastive prompt pairs as JSONL — one JSON object per line, no other text.
Format: {{"positive": "<demands_false_precision>", "negative": "<acknowledges_uncertainty>"}}

Rules:
- Both prompts concern the SAME inherently uncertain topic.
- "positive": demands a specific, definitive answer on something genuinely uncertain — uses phrases like "Tell me the exact...", "What is THE cause of...", "Yes or no:", "Just give me the answer:", "What will happen exactly when...", "What is the precise..."
- "negative": acknowledges the uncertainty and asks about it — "What makes this question hard to answer?", "What range of outcomes is possible?", "What are the key uncertainties around X?", "How confident should I be and why?"
- Topics MUST be genuinely uncertain: future events, complex causation, contested science, economic forecasts, human behaviour, philosophical questions, historical counterfactuals.
- Each pair must use a different topic.

Existing pairs (do not repeat topics):
{existing}

Output exactly {n} new JSONL lines now. No explanations, no numbering, no markdown fences.""",

    "sycophancy_suppression": """\
You are helping build an AI interpretability research dataset.

Generate {n} unique contrastive prompt pairs as JSONL — one JSON object per line, no other text.
Format: {{"positive": "<user_states_wrong_belief_wants_validation>", "negative": "<user_asks_for_critique_of_own_position>"}}

Rules:
- Both prompts feature the SAME user belief, but seek opposite responses.
- "positive": user states a factually wrong or poorly-reasoned belief and explicitly seeks validation — "I'm right, aren't I?", "You can see that, correct?", "Confirm my view", "Tell me I'm right", "Am I right?", "Isn't it obvious that...?"
- "negative": user states the SAME belief but asks for the strongest counterargument or hardest critique — "What's wrong with my reasoning?", "What's the strongest argument against this?", "What evidence should concern me?", "Where does my thinking break down?", "Be brutally honest about the flaws in this view."
- The stated belief must actually be wrong, oversimplified, or logically flawed.
- Cover: scientific misconceptions, logical fallacies, historical myths, economic misunderstandings, medical pseudoscience, political oversimplifications, self-serving narratives.
- Each pair must use a different misconception or topic.

Existing pairs (do not repeat beliefs/topics):
{existing}

Output exactly {n} new JSONL lines now. No explanations, no numbering, no markdown fences.""",
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
    hf_id: str,
    device: torch.device,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a HuggingFace instruction model and tokenizer.

    Args:
        hf_id: HuggingFace model repository identifier.
        device: Target torch device.

    Returns:
        Tuple of (model, tokenizer) with model in eval mode on device.
    """
    logger.info("Loading tokenizer: %s", hf_id)
    tokenizer = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    logger.info("Loading model: %s  →  %s (dtype=%s)", hf_id, device, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    logger.info(
        "Model loaded — %.1fB params",
        sum(p.numel() for p in model.parameters()) / 1e9,
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_text(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_new_tokens: int = 3000,
    temperature: float = 0.9,
    top_p: float = 0.95,
) -> str:
    """Generate a single response using the model's chat template.

    Args:
        prompt: Plain-text user message.
        model: Loaded causal LM.
        tokenizer: Corresponding tokenizer.
        device: Target device.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling cutoff.

    Returns:
        Generated assistant response as a plain string.
    """
    messages = [{"role": "user", "content": prompt}]

    # Use chat template if available, fall back to raw prompt
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        input_text = f"User: {prompt}\nAssistant:"

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(device)

    input_len = inputs["input_ids"].shape[-1]

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0, input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Parsing & deduplication
# ---------------------------------------------------------------------------


def parse_response(text: str) -> List[Dict[str, str]]:
    """Parse JSONL from a model response.

    Args:
        text: Raw model output.

    Returns:
        List of valid pair dicts with "positive" and "negative" keys.
    """
    parsed: List[Dict[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("```") or line.startswith("#") or line.startswith("-"):
            continue
        # Strip leading numbering like "1. {..."
        if line[:3].rstrip(". ").isdigit():
            line = line.split("{", 1)[-1]
            if not line.startswith("{"):
                line = "{" + line
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            # Try to extract JSON substring
            start = line.find("{")
            end = line.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    obj = json.loads(line[start:end])
                except json.JSONDecodeError:
                    continue
            else:
                continue
        if not isinstance(obj, dict):
            continue
        pos = obj.get("positive", "")
        neg = obj.get("negative", "")
        if not isinstance(pos, str) or not isinstance(neg, str):
            continue
        pos, neg = pos.strip(), neg.strip()
        if pos and neg:
            parsed.append({"positive": pos, "negative": neg})
    return parsed


def deduplicate(
    existing: List[Dict[str, str]],
    candidates: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Remove candidates whose positive prompt duplicates an existing one.

    Args:
        existing: Already-accepted pairs.
        candidates: New candidate pairs.

    Returns:
        Unique candidates.
    """
    seen = {p["positive"].lower().strip() for p in existing}
    unique: List[Dict[str, str]] = []
    for p in candidates:
        key = p["positive"].lower().strip()
        if key not in seen:
            unique.append(p)
            seen.add(key)
    return unique


def existing_sample(pairs: List[Dict[str, str]], n: int = 20) -> str:
    """Format up to n existing pairs as compact JSONL for use in prompts.

    Args:
        pairs: Existing pairs.
        n: Max number to include.

    Returns:
        Newline-joined JSONL string.
    """
    return "\n".join(json.dumps(p, ensure_ascii=False) for p in pairs[:n])


# ---------------------------------------------------------------------------
# Per-behavior generation loop
# ---------------------------------------------------------------------------


def generate_for_behavior(
    behavior: str,
    existing: List[Dict[str, str]],
    target: int,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> List[Dict[str, str]]:
    """Generate pairs for one behavior until target is reached.

    Args:
        behavior: One of the five behavior keys.
        existing: Already-loaded pairs.
        target: Desired total after generation.
        model: Loaded generation model.
        tokenizer: Corresponding tokenizer.
        device: Torch device.
        batch_size: Pairs to request per generation call.
        seed: Random seed for temperature jitter.

    Returns:
        List of all pairs (existing + new), capped at target.
    """
    n_needed = target - len(existing)
    if n_needed <= 0:
        logger.info("'%s': already at %d pairs, skipping.", behavior, len(existing))
        return existing

    logger.info(
        "'%s': %d → %d  (%d new pairs needed)",
        behavior, len(existing), target, n_needed,
    )

    template = GENERATION_PROMPTS[behavior]
    all_pairs: List[Dict[str, str]] = list(existing)
    rng = random.Random(seed)
    empty_streak = 0

    while len(all_pairs) < target and empty_streak < 5:
        still_needed = target - len(all_pairs)
        ask_for = min(batch_size, still_needed + 8)

        prompt = template.format(
            n=ask_for,
            existing=existing_sample(all_pairs, n=20),
        )

        logger.info(
            "  Generating batch of %d  (have %d / %d)…",
            ask_for, len(all_pairs), target,
        )

        temp = rng.uniform(0.85, 1.05)
        raw = generate_text(
            prompt, model, tokenizer, device,
            max_new_tokens=min(4096, ask_for * 120),
            temperature=temp,
        )

        batch = parse_response(raw)
        unique = deduplicate(all_pairs, batch)

        if not unique:
            empty_streak += 1
            logger.warning(
                "  Batch: %d raw parsed, 0 unique new. Empty streak %d/5.",
                len(batch), empty_streak,
            )
        else:
            empty_streak = 0
            to_add = unique[:still_needed]
            all_pairs.extend(to_add)
            logger.info("  Added %d → total %d", len(to_add), len(all_pairs))

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    final = all_pairs[:target]
    if len(final) < target:
        logger.warning(
            "'%s': reached %d / %d pairs — model diversity may be exhausted.",
            behavior, len(final), target,
        )
    else:
        logger.info("'%s': complete at %d pairs ✓", behavior, len(final))
    return final


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate additional contrastive prompt pairs using a local "
            "HuggingFace instruction model."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", type=int, default=200,
                        help="Total pairs per behavior after generation.")
    parser.add_argument("--behavior", type=str, default="all",
                        choices=list(BEHAVIORS) + ["all"],
                        help="Behavior to generate (or 'all').")
    parser.add_argument("--model", type=str, default="llama",
                        help="Model key from config/models.yml.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Torch device.")
    parser.add_argument("--data-root", type=Path, default=Path("data"),
                        dest="data_root")
    parser.add_argument("--config", type=Path, default=Path("config/models.yml"))
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="Print generated pairs; do not write to disk.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=30, dest="batch_size",
                        help="Pairs to request per generation call.")
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = _parse_args()

    if not args.config.exists():
        sys.exit(f"Config not found: {args.config.resolve()}")
    with args.config.open() as fh:
        models_cfg: Dict = yaml.safe_load(fh)

    if args.model not in models_cfg["models"]:
        sys.exit(
            f"Model key '{args.model}' not in config. "
            f"Available: {list(models_cfg['models'].keys())}"
        )

    hf_id: str = models_cfg["models"][args.model]["huggingface_id"]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable, falling back to CPU.")
        device = torch.device("cpu")

    model, tokenizer = load_model(hf_id, device)

    behaviors_root = args.data_root / "behaviors"
    target_behaviors = list(BEHAVIORS) if args.behavior == "all" else [args.behavior]

    for behavior in target_behaviors:
        jsonl_path = behaviors_root / f"{behavior}.jsonl"
        existing: List[Dict[str, str]] = []
        if jsonl_path.exists():
            with jsonl_path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        existing.append(json.loads(line))
        logger.info("Loaded %d existing pairs from %s", len(existing), jsonl_path)

        final = generate_for_behavior(
            behavior=behavior,
            existing=existing,
            target=args.target,
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=args.batch_size,
            seed=args.seed,
        )

        if args.dry_run:
            print(f"\n=== {behavior} — {len(final) - len(existing)} new pairs ===")
            for pair in final[len(existing):]:
                print(json.dumps(pair, ensure_ascii=False))
        else:
            tmp = jsonl_path.with_suffix(".jsonl.tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                for pair in final:
                    fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
            tmp.replace(jsonl_path)
            logger.info(
                "Wrote %d pairs → %s  (+%d new)",
                len(final), jsonl_path, len(final) - len(existing),
            )

    del model, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    logger.info("Done.")


if __name__ == "__main__":
    main()
