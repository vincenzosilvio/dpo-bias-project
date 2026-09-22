"""
Generates raw candidate completions from the base model for every
(occupation, prompt) pair. Run this on Colab with a GPU runtime.

Output: data/raw_completions.jsonl, one JSON object per line:
    {
        "occupation": str,
        "stereotype": "male" | "female" | "balanced",
        "prompt": str,
        "completion": str,
        "sample_id": int
    }

These raw completions get scored and paired into DPO preference pairs by
label_completions.py — this script only generates, it does not label.

Usage (from repo root, in Colab):
    !pip install -q -r requirements.txt
    !python src/generate_dataset.py --model Qwen/Qwen2.5-0.5B-Instruct --samples-per-prompt 4
"""

import argparse
import json
import os

from occupations import build_prompts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face model id of the base model to sample from.",
    )
    parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=4,
        help="Number of completions to sample per prompt (higher = more "
             "chance of finding a stereotyped/non-stereotyped pair, but "
             "slower).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature. Needs to be high enough that completions "
             "actually vary — if every sample is identical there's nothing "
             "to build preference pairs from.",
    )
    parser.add_argument(
        "--output",
        default="data/raw_completions.jsonl",
    )
    parser.add_argument(
        "--limit-prompts",
        type=int,
        default=None,
        help="For a quick smoke test, only run on the first N prompts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Imported here so `python src/occupations.py` alone (no GPU deps) still works.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    prompts = build_prompts()
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]
    print(f"Generating {args.samples_per_prompt} completions each for "
          f"{len(prompts)} prompts ({len(prompts) * args.samples_per_prompt} "
          f"total generations).")

    with open(args.output, "w") as f_out:
        for occ_name, stereotype, prompt in prompts:
            # Use the chat template so instruct-tuned models behave as expected.
            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)

            for sample_id in range(args.samples_per_prompt):
                with torch.no_grad():
                    output_ids = model.generate(
                        inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                completion = tokenizer.decode(
                    output_ids[0][inputs.shape[-1]:], skip_special_tokens=True
                ).strip()

                record = {
                    "occupation": occ_name,
                    "stereotype": stereotype,
                    "prompt": prompt,
                    "completion": completion,
                    "sample_id": sample_id,
                }
                f_out.write(json.dumps(record) + "\n")

            print(f"  done: {occ_name} ({stereotype}) - '{prompt[:40]}...'")

    print(f"Wrote completions to {args.output}")


if __name__ == "__main__":
    main()
