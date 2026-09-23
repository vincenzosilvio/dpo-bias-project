"""
Generates raw candidate completions from the base model for every
(occupation, template) prompt. Run on Colab with a GPU runtime, from the
repo root:

    python src/generate_dataset.py --samples-per-prompt 8

Output: data/raw_completions.jsonl, one JSON object per line:
    occupation, stereotype, split, template_id, prompt,
    messages     -- exact chat context (system + user) the sample came from
    completion, sample_id,
    n_new_tokens -- generated length in tokens
    truncated    -- True if generation hit max_new_tokens without EOS
    gen_config   -- model id, sampling params, seed (reproducibility)

Generation covers ALL occupations, held-out and control included: their raw
completions are the base model's generation-bias baseline, which
review_sample.py reports and which we will recompute after DPO.
label_completions.py decides which records may become training pairs.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from occupations import build_messages, build_prompts  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument(
        "--max-new-tokens", type=int, default=800,
        help="Set from a measured length distribution (40 samples, cap 1024, "
             "0 truncated: median 379, p95 679, max 718 tokens). Must be "
             "large enough that most stories END. Truncated samples "
             "are excluded from pairs: training DPO on text cut mid-sentence "
             "teaches abrupt endings and would contaminate the helpfulness "
             "check (claim 2).",
    )
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(REPO_ROOT / "data" / "raw_completions.jsonl"))
    parser.add_argument("--limit-prompts", type=int, default=None,
                        help="Smoke test: only the first N prompts.")
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # bf16 only where the GPU supports it natively: compute capability >= 8
    # (Ampere and later: A100, L4, H100...). T4 (7.5) and P100 (6.0) don't.
    # NOT torch.cuda.is_bf16_supported(): by default it also counts
    # emulated bf16 and returns True on a T4 (observed on Kaggle).
    # Elsewhere use float32: 0.5B weights are ~2 GB, fp16 risks overflow.
    use_bf16 = (torch.cuda.is_available()
                and torch.cuda.get_device_capability(0)[0] >= 8)
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map="auto",
    )
    model.eval()

    eos_ids = model.generation_config.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids])

    gen_config = {
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        # Same seed != same samples on different hardware/library versions:
        # record both, and never mix outputs from different setups.
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "dtype": str(dtype),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }

    prompts = build_prompts()
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]
    print(f"{len(prompts)} prompts x {args.samples_per_prompt} samples = "
          f"{len(prompts) * args.samples_per_prompt} generations.")

    n_truncated = 0
    n_total = 0
    with open(args.output, "w") as f_out:
        for p in prompts:
            messages = build_messages(p["prompt"])
            encoded = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True,
                return_tensors="pt", return_dict=True,
            ).to(model.device)
            prompt_len = encoded["input_ids"].shape[-1]

            # All samples for a prompt in one call: same distribution as a
            # loop of single samples, several times faster on a T4.
            with torch.no_grad():
                out = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_return_sequences=args.samples_per_prompt,
                    pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                                  else tokenizer.eos_token_id),
                )

            for sample_id, seq in enumerate(out):
                gen_ids = seq[prompt_len:].tolist()
                # Length up to and including the first EOS (rest is padding).
                eos_pos = next((i for i, t in enumerate(gen_ids) if t in eos_ids), None)
                truncated = eos_pos is None
                n_new = len(gen_ids) if truncated else eos_pos + 1
                completion = tokenizer.decode(
                    gen_ids[:n_new], skip_special_tokens=True
                ).strip()
                n_truncated += truncated
                n_total += 1
                f_out.write(json.dumps({
                    **p,
                    "messages": messages,
                    "completion": completion,
                    "sample_id": sample_id,
                    "n_new_tokens": n_new,
                    "truncated": truncated,
                    "gen_config": gen_config,
                }) + "\n")

            print(f"  done: {p['occupation']} ({p['stereotype']}, t{p['template_id']})")

    print(f"Wrote {n_total} completions to {args.output}")
    print(f"Truncated (no EOS within {args.max_new_tokens} tokens): "
          f"{n_truncated}/{n_total} ({100 * n_truncated / max(n_total, 1):.1f}%)")
    if n_truncated / max(n_total, 1) > 0.15:
        print("WARNING: >15% truncated. Raise --max-new-tokens and regenerate "
              "rather than losing these samples to the truncation filter.")


if __name__ == "__main__":
    main()
