"""
Prints a review of the generated data directly in the Colab output:

  1. Filtering funnel     -- where completions/groups were lost and why
  2. Baseline generation bias of the BASE model, per occupation and per
     group (train / held-out / control). This is a result, not just a
     sanity check: it is the "before" of the generation-bias metric that we
     recompute after DPO with the same prompts.
  3. Pair composition     -- per occupation/template, length balance
  4. Auto-flags           -- pairs where a gendered pronoun shares a
                             sentence with a second-person noun (colleague,
                             wife, patient...): misattribution candidates
  5. Random sample        -- pronouns highlighted as [[she]] for fast reading

It also writes data/manual_review.csv with one row per sampled pair and
empty verdict columns. Fill it in: the error rate of the heuristic, measured
on that sample, goes in the write-up ("manual audit of N pairs, X% with
correct pronoun attribution"). It's the number a reviewer will ask for.

Usage (repo root):
    python src/review_sample.py --sample-size 20
"""

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_completions import (  # noqa: E402
    exclusion_reason, length_ratio, pronoun_counts,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"

PRONOUN_ANY = re.compile(r"\b(he|him|his|himself|she|her|hers|herself)\b", re.IGNORECASE)
OTHER_PERSON = re.compile(
    r"\b(colleague|coworker|co-worker|boss|manager|assistant|friend|patient|"
    r"customer|client|visitor|guest|student|child|kid|son|daughter|wife|"
    r"husband|partner|mother|father|mom|dad|sister|brother|passenger|"
    r"man|woman|boy|girl|stranger)s?\b",
    re.IGNORECASE,
)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def highlight(text):
    return PRONOUN_ANY.sub(lambda m: f"[[{m.group(0)}]]", text)


def suspicious_sentences(text):
    """Sentences containing both a gendered pronoun and another-person noun."""
    return [s for s in SENTENCE_SPLIT.split(text)
            if PRONOUN_ANY.search(s) and OTHER_PERSON.search(s)]


def pronoun_gender(text):
    c = pronoun_counts(text)
    if c["male"] and c["female"]:
        return "mixed"
    if c["male"]:
        return "male"
    if c["female"]:
        return "female"
    return "none"


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def print_funnel(stats):
    section("1. FILTERING FUNNEL")
    f = stats["funnel"]
    total = f.get("total", 0)
    for k, v in sorted(f.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<36}{v:>6}  ({100 * v / max(total, 1):5.1f}% of total)")
    print(f"\n  Train groups (occupation x template): {stats['train_groups']}")
    for k, v in stats["group_outcomes"].items():
        print(f"    {k:<34}{v:>6}")
    print(f"\n  PAIRS: {stats['n_pairs']}  by stereotype: {stats['pairs_by_stereotype']}")


def print_baseline(raw, min_pronoun_count):
    section("2. BASE-MODEL GENERATION BIAS (usable completions only)")
    print("  Columns: share of usable completions whose pronouns are\n"
          "  female-only / male-only / mixed; 'n' = usable completions.\n"
          "  'counter%' = share that is counter-stereotypical (0 for control).\n")
    rows = defaultdict(Counter)
    meta = {}
    for r in raw:
        if exclusion_reason(r, min_pronoun_count) is not None:
            continue
        g = pronoun_gender(r["completion"])
        rows[r["occupation"]][g] += 1
        meta[r["occupation"]] = (r["stereotype"], r.get("split", "train"))

    group_tot = defaultdict(Counter)
    hdr = f"  {'occupation':<20}{'stereo':<9}{'split':<9}{'n':>4}{'fem%':>7}{'male%':>7}{'mix%':>7}{'counter%':>10}"
    print(hdr)
    for occ in sorted(rows, key=lambda o: (meta[o][0], meta[o][1], o)):
        c, (st, sp) = rows[occ], meta[occ]
        n = sum(c.values())
        counter = c["female"] if st == "male" else c["male"] if st == "female" else 0
        print(f"  {occ:<20}{st:<9}{sp:<9}{n:>4}{100*c['female']/n:>7.0f}"
              f"{100*c['male']/n:>7.0f}{100*c['mixed']/n:>7.0f}{100*counter/n:>10.0f}")
        group_tot[(st, sp)].update(c)
    print("\n  Aggregates:")
    for (st, sp), c in sorted(group_tot.items()):
        n = sum(c.values())
        print(f"    {st:<9}{sp:<9} n={n:<5} fem={100*c['female']/n:5.1f}%  "
              f"male={100*c['male']/n:5.1f}%  mixed={100*c['mixed']/n:5.1f}%")
    print("\n  Read this before anything else: if counter% is ~0 for most\n"
          "  train occupations, pair yield will be low no matter the filters,\n"
          "  and that is a finding about the base model, not a pipeline bug.")


def print_composition(pairs):
    section("3. PAIR COMPOSITION")
    by_occ = Counter(p["occupation"] for p in pairs)
    by_tpl = Counter(p["template_id"] for p in pairs)
    print("  per occupation:", dict(sorted(by_occ.items())))
    print("  per template:  ", dict(sorted(by_tpl.items())))
    if pairs:
        ratios = sorted(length_ratio(p["chosen"][0]["content"], p["rejected"][0]["content"]) for p in pairs)
        ch_len = sum(len(p["chosen"][0]["content"]) for p in pairs) / len(pairs)
        rj_len = sum(len(p["rejected"][0]["content"]) for p in pairs) / len(pairs)
        longer = sum(len(p["chosen"][0]["content"]) > len(p["rejected"][0]["content"]) for p in pairs)
        print(f"  length ratio: median={ratios[len(ratios)//2]:.2f} max={ratios[-1]:.2f}")
        print(f"  mean chars: chosen={ch_len:.0f} rejected={rj_len:.0f}; "
              f"chosen longer in {longer}/{len(pairs)} pairs "
              f"(should be near 50% -- otherwise DPO can learn length)")


def print_flags(pairs, max_show):
    section("4. AUTO-FLAGGED: pronoun + other-person noun in the same sentence")
    flagged = []
    for p in pairs:
        for side in ("chosen", "rejected"):
            sents = suspicious_sentences(p[side][0]["content"])
            if sents:
                flagged.append((p["pair_id"], side, p["occupation"], sents))
    print(f"  {len(flagged)} flagged completions in {len(pairs)} pairs "
          f"(heuristic, high recall / low precision).")
    for pid, side, occ, sents in flagged[:max_show]:
        print(f"\n  pair {pid} [{side}] {occ}:")
        for s in sents[:3]:
            print("    > " + highlight(s))
    if len(flagged) > max_show:
        print(f"\n  ... {len(flagged) - max_show} more (raise --max-flags to see them)")
    print("\n  For each: is every [[pronoun]] about the occupation-holder?")


def print_sample_and_csv(pairs, n, seed, csv_path):
    section(f"5. RANDOM SAMPLE ({min(n, len(pairs))} pairs, seed={seed})")
    rng = random.Random(seed)
    sample = rng.sample(pairs, min(n, len(pairs)))
    for p in sample:
        print(f"\n--- pair {p['pair_id']} | {p['occupation']} ({p['stereotype']}) "
              f"| template {p['template_id']} ---")
        print(f"PROMPT: {p['prompt'][-1]['content']}")
        print(f"\nCHOSEN {p['chosen_pronouns']}:\n{highlight(p['chosen'][0]['content'])}")
        print(f"\nREJECTED {p['rejected_pronouns']}:\n{highlight(p['rejected'][0]['content'])}")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "occupation", "stereotype", "template_id",
                    "chosen_attribution_ok", "rejected_attribution_ok",
                    "chosen_name_matches_pronouns", "rejected_name_matches_pronouns",
                    "other_issue", "notes"])
        for p in sample:
            w.writerow([p["pair_id"], p["occupation"], p["stereotype"],
                        p["template_id"], "", "", "", "", "", ""])
    print(f"\nWrote review sheet to {csv_path} -- fill in y/n per column.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(DATA / "raw_completions.jsonl"))
    ap.add_argument("--pairs", default=str(DATA / "dpo_pairs.jsonl"))
    ap.add_argument("--stats", default=str(DATA / "label_stats.json"))
    ap.add_argument("--csv", default=str(DATA / "manual_review.csv"))
    ap.add_argument("--sample-size", type=int, default=20)
    ap.add_argument("--max-flags", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw = load_jsonl(args.raw)
    pairs = load_jsonl(args.pairs)
    with open(args.stats) as f:
        stats = json.load(f)

    print_funnel(stats)
    print_baseline(raw, stats["args"]["min_pronoun_count"])
    print_composition(pairs)
    print_flags(pairs, args.max_flags)
    print_sample_and_csv(pairs, args.sample_size, args.seed, args.csv)


if __name__ == "__main__":
    main()
