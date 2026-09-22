"""
Scores raw completions for gender-stereotype signal and builds DPO
preference pairs (chosen = less stereotyped, rejected = more stereotyped).

Current approach: a pronoun-based heuristic. For a "male"-stereotyped
occupation, a completion that uses female pronouns (she/her/hers) for the
occupation-holder counts as counter-stereotypical (higher score); one using
male pronouns (he/him/his) counts as stereotype-reinforcing (lower score).
Mirrored for "female"-stereotyped occupations. "Balanced" occupations are
scored for gender-neutrality (fewer gendered pronouns relative to neutral
phrasing = higher score).

KNOWN LIMITATION (flag this explicitly in the write-up): a pronoun count is
a crude proxy. It doesn't catch stereotyping expressed through adjectives,
role framing, or narrative content when no pronoun is used at all. Before
finalizing the dataset, spot-check a random sample of the pairs by hand —
plan for at least an afternoon of manual review, not zero. A natural next
step (flagged as a TODO, not yet implemented) is to add a regard/sentiment
classifier as reported in the BOLD paper (Dhamala et al., 2021) as a second,
independent signal — search the Hugging Face Hub for a regard-classification
model, since the exact best current model id isn't something to hardcode
here without checking it works.

Output: data/dpo_pairs.jsonl, one JSON object per line:
    {"prompt": str, "chosen": str, "rejected": str, "occupation": str, "stereotype": str}

Usage:
    python src/label_completions.py --input data/raw_completions.jsonl --output data/dpo_pairs.jsonl
"""

import argparse
import json
import re
from collections import defaultdict

MALE_PRONOUNS = re.compile(r"\b(he|him|his)\b", re.IGNORECASE)
FEMALE_PRONOUNS = re.compile(r"\b(she|her|hers)\b", re.IGNORECASE)

# Minimum/maximum acceptable length ratio between chosen and rejected, so
# DPO doesn't just learn "prefer longer/shorter text" instead of "prefer
# less stereotyped text". Tune this after inspecting real data.
MAX_LENGTH_RATIO = 1.6


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw_completions.jsonl")
    parser.add_argument("--output", default="data/dpo_pairs.jsonl")
    parser.add_argument(
        "--min-pronoun-count",
        type=int,
        default=1,
        help="Skip completions with fewer than this many gendered pronouns "
             "-- with zero pronouns, the heuristic has no signal to work with.",
    )
    return parser.parse_args()


def pronoun_counts(text):
    return {
        "male": len(MALE_PRONOUNS.findall(text)),
        "female": len(FEMALE_PRONOUNS.findall(text)),
    }


def counter_stereotype_score(completion_text, stereotype_label):
    """
    Higher score = less stereotyped (more desirable as the "chosen" completion).
    """
    counts = pronoun_counts(completion_text)
    male_n, female_n = counts["male"], counts["female"]

    if stereotype_label == "male":
        # Female pronouns for a male-coded occupation = counter-stereotype.
        return female_n - male_n
    elif stereotype_label == "female":
        # Male pronouns for a female-coded occupation = counter-stereotype.
        return male_n - female_n
    else:  # "balanced": reward gender-neutral phrasing (fewer gendered pronouns overall)
        return -(male_n + female_n)


def length_ratio(a, b):
    la, lb = max(len(a), 1), max(len(b), 1)
    return max(la, lb) / min(la, lb)


def main():
    args = parse_args()

    grouped = defaultdict(list)  # (occupation, prompt) -> list of records
    skipped_no_signal = 0

    with open(args.input) as f_in:
        for line in f_in:
            record = json.loads(line)
            counts = pronoun_counts(record["completion"])
            if (counts["male"] + counts["female"]) < args.min_pronoun_count:
                skipped_no_signal += 1
                continue
            record["score"] = counter_stereotype_score(
                record["completion"], record["stereotype"]
            )
            grouped[(record["occupation"], record["prompt"])].append(record)

    pairs = []
    skipped_no_contrast = 0
    skipped_length_mismatch = 0

    for (occupation, prompt), records in grouped.items():
        if len(records) < 2:
            continue
        records.sort(key=lambda r: r["score"], reverse=True)
        best, worst = records[0], records[-1]

        if best["score"] == worst["score"]:
            skipped_no_contrast += 1
            continue
        if length_ratio(best["completion"], worst["completion"]) > MAX_LENGTH_RATIO:
            skipped_length_mismatch += 1
            continue

        pairs.append({
            "prompt": prompt,
            "chosen": best["completion"],
            "rejected": worst["completion"],
            "occupation": occupation,
            "stereotype": best["stereotype"],
        })

    with open(args.output, "w") as f_out:
        for pair in pairs:
            f_out.write(json.dumps(pair) + "\n")

    print(f"Read completions, skipped {skipped_no_signal} with no pronoun signal.")
    print(f"Skipped {skipped_no_contrast} groups with no score contrast, "
          f"{skipped_length_mismatch} for length mismatch.")
    print(f"Wrote {len(pairs)} DPO preference pairs to {args.output}")
    if len(pairs) < 50:
        print("WARNING: fewer than 50 pairs. Consider increasing "
              "--samples-per-prompt in generate_dataset.py, or lowering "
              "--min-pronoun-count, before training.")


if __name__ == "__main__":
    main()
