"""
Review of the SHORT-form dataset, printed in the notebook output:

  1. Funnel               -- exclusions overall and per template
  2. Calibration          -- p_female per train occupation, rule decision
  3. Base-model bias      -- p_female by group (stereotype x occupation
                             split x template split), all samples. This is
                             the "before" of the evaluation metric.
  4. Pair composition     -- by occupation / direction, flags, names
  5. Excluded examples    -- a few texts per exclusion reason, to check the
                             filters are not discarding good data
  6. Pair sample          -- rejected / chosen with pronouns highlighted
                             and the swap log

Writes data/short/manual_review.csv (one row per sampled pair, empty
verdict columns). The audited error rates go in the write-up.

Usage (repo root):
    python src/review_sample.py --sample-size 25
"""

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data" / "short"

PRONOUN_ANY = re.compile(r"\b(he|him|his|himself|she|her|hers|herself)\b", re.IGNORECASE)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def highlight(text):
    return PRONOUN_ANY.sub(lambda m: f"[[{m.group(0)}]]", text)


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def pf(c):
    n = c["male"] + c["female"]
    return (c["female"] / n if n else float("nan")), n


def print_funnel(stats, analysis):
    section("1. FUNNEL")
    f = stats["funnel"]
    total = f["total"]
    for k, v in sorted(f.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<32}{v:>6}  ({100 * v / total:5.1f}%)")
    print("\n  Usable share per template (all occupations):")
    by_t = defaultdict(Counter)
    for a in analysis:
        by_t[(a["template_id"], a["template_split"])][a["reason"] or "usable"] += 1
    for (t, ts), c in sorted(by_t.items()):
        n = sum(c.values())
        top = ", ".join(f"{k}={v}" for k, v in c.most_common(5) if k != "usable")
        print(f"    t{t} ({ts:<7}) usable {100 * c['usable'] / n:5.1f}%   top exclusions: {top}")
    print(f"\n  Pair source: {stats['pair_source_outcome']}")
    print(f"  Swap failures: {stats['swap_failures']}")


def print_calibration(cal):
    section("2. CALIBRATION (pre-registered rule: " +
            ", ".join(f"{k}={v}" for k, v in cal["rule"].items()) + ")")
    print(f"  {'occupation':<18}{'stereo':<9}{'n':>4}{'p_female':>10}  decision")
    for occ, c in sorted(cal["occupations"].items(), key=lambda kv: (kv[1]["stereotype"], kv[0])):
        p = f"{'n/a':>10}" if c["p_female"] is None else f"{c['p_female']:>10.2f}"
        d = c["decision"] + (f" -> push to {c['target_gender']}" if c["target_gender"] else "")
        print(f"  {occ:<18}{c['stereotype']:<9}{c['n']:>4}{p}  {d}")


def print_baseline(analysis):
    section("3. BASE-MODEL BIAS: p_female among usable completions (all samples)")
    groups = defaultdict(Counter)
    per_occ = defaultdict(Counter)
    meta = {}
    for a in analysis:
        if a["reason"] is None:
            groups[(a["stereotype"], a["split"], a["template_split"])][a["gender"]] += 1
            per_occ[a["occupation"]][a["gender"]] += 1
            meta[a["occupation"]] = (a["stereotype"], a["split"])
    print(f"  {'stereotype':<11}{'occ split':<10}{'tmpl split':<11}{'n':>5}{'p_female':>10}")
    for (st, sp, ts), c in sorted(groups.items()):
        p, n = pf(c)
        print(f"  {st:<11}{sp:<10}{ts:<11}{n:>5}{p:>10.2f}")
    fem, mal = Counter(), Counter()
    for (st, _, _), c in groups.items():
        if st == "female":
            fem.update(c)
        elif st == "male":
            mal.update(c)
    pfem, _ = pf(fem)
    pmal, _ = pf(mal)
    print(f"\n  GAP p_female(female-coded) - p_female(male-coded) = {pfem - pmal:+.2f}"
          "\n  (headline metric: DPO should shrink it, not reverse it)")
    print(f"\n  {'occupation':<18}{'stereo':<9}{'split':<8}{'n':>5}{'p_female':>10}")
    for occ in sorted(per_occ, key=lambda o: (meta[o], o)):
        p, n = pf(per_occ[occ])
        print(f"  {occ:<18}{meta[occ][0]:<9}{meta[occ][1]:<8}{n:>5}{p:>10.2f}")


def print_composition(pairs, stats):
    section("4. PAIR COMPOSITION")
    print(f"  pairs: {len(pairs)}   by direction: {stats['pairs_by_direction']}")
    print("  by occupation:", dict(sorted(Counter(p["occupation"] for p in pairs).items())))
    print("  by template:  ", dict(sorted(Counter(p["template_id"] for p in pairs).items())))
    flagged = sum(p["other_person_flag"] for p in pairs)
    print(f"  other-person noun present: {flagged}/{len(pairs)} (check these in the audit)")
    print(f"  name pool sizes: {stats['name_pool_sizes']}")
    print("  most frequent replacement names:",
          Counter(p["new_name"] for p in pairs).most_common(8))


def print_excluded(raw, analysis, per_reason, seed):
    section(f"5. EXCLUDED EXAMPLES (up to {per_reason} per reason) -- are the filters right?")
    rng = random.Random(seed)
    by_reason = defaultdict(list)
    for r, a in zip(raw, analysis):
        if a["reason"] not in (None, "truncated"):
            by_reason[a["reason"]].append(r)
    for reason, rs in sorted(by_reason.items()):
        print(f"\n  --- {reason} ({len(rs)}) ---")
        for r in rng.sample(rs, min(per_reason, len(rs))):
            print(f"  [{r['occupation']}] {highlight(r['completion'])}")


def print_sample(pairs, n, seed, csv_path):
    section(f"6. PAIR SAMPLE ({min(n, len(pairs))} pairs, seed={seed})")
    sample = random.Random(seed).sample(pairs, min(n, len(pairs)))
    for p in sample:
        flag = "  [other-person noun]" if p["other_person_flag"] else ""
        print(f"\n--- pair {p['pair_id']} | {p['occupation']} | t{p['template_id']} | "
              f"{p['rejected_gender']} -> {p['chosen_gender']}{flag}")
        print(f"REJECTED: {highlight(p['rejected'][0]['content'])}")
        print(f"CHOSEN:   {highlight(p['chosen'][0]['content'])}")
        print("swaps:", ", ".join(f"{a}->{b}" for a, b in p["swap_log"]))
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "occupation", "direction",
                    "single_character_ok", "pronouns_refer_to_holder",
                    "swap_correct", "chosen_fluent", "notes"])
        for p in sample:
            w.writerow([p["pair_id"], p["occupation"],
                        f"{p['rejected_gender']}->{p['chosen_gender']}", "", "", "", "", ""])
    print(f"\nWrote {csv_path} -- fill in y/n per column.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA))
    ap.add_argument("--sample-size", type=int, default=25)
    ap.add_argument("--excluded-per-reason", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    d = Path(args.data_dir)

    raw = load_jsonl(d / "raw_completions.jsonl")
    analysis = load_jsonl(d / "analysis.jsonl")
    pairs = load_jsonl(d / "dpo_pairs.jsonl")
    with open(d / "pair_stats.json") as f:
        stats = json.load(f)
    with open(d / "calibration.json") as f:
        cal = json.load(f)

    print_funnel(stats, analysis)
    print_calibration(cal)
    print_baseline(analysis)
    print_composition(pairs, stats)
    print_excluded(raw, analysis, args.excluded_per_reason, args.seed)
    print_sample(pairs, args.sample_size, args.seed, d / "manual_review.csv")


if __name__ == "__main__":
    main()
