"""
Builds the DPO dataset from SHORT completions (README, lesson #9):

1. ANALYZE every completion (text_utils.analyze): text filters, pure
   pronouns, exactly one named character whose name agrees with the
   pronouns. Results -> data/short/analysis.jsonl (reused by review).

2. CALIBRATE on the calibration half (sample_id < N_CALIBRATION), train
   occupations x train templates only: per occupation, the share of usable
   completions with female pronouns, p_female. PRE-REGISTERED RULE (fixed
   before any short-form data existed; do not tune it after seeing results):
       an occupation gets pairs  iff  n >= MIN_CALIBRATION_N
                                 and  |p_female - 0.5| >= CALIBRATION_MARGIN
       direction: from the model's majority gender towards the minority one.
   Occupations near parity get no pairs. The human "stereotype" label is NOT
   used here -- the model's own measured bias decides (for this model,
   "scientist" is female-majority: lesson #9). Stereotype labels are only
   used to group results in evaluation.
   -> data/short/calibration.json

3. PAIR the pair-source half (sample_id >= N_CALIBRATION), same cells:
   every usable completion of a targeted occupation whose gender is the
   majority one becomes REJECTED; its counterfactual swap (counterfactual.py)
   becomes CHOSEN. Replacement names are sampled from the names the model
   itself produced for the target gender.
   -> data/short/dpo_pairs.jsonl (TRL conversational format + metadata)
   -> data/short/pair_stats.json

Usage (repo root):
    python src/build_pairs.py
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from counterfactual import build_name_pool, sample_name, swap_gender  # noqa: E402
from occupations import build_messages  # noqa: E402
from text_utils import analyze, get_nlp  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data" / "short"

# ---- Pre-registered calibration rule (see module docstring) --------------
N_CALIBRATION = 8          # sample ids 0..7 of 16 per prompt
MIN_CALIBRATION_N = 12     # usable calibration completions per occupation
CALIBRATION_MARGIN = 0.20  # target iff p_female <= 0.30 or >= 0.70
# With n ~ 32 the standard error of p is ~0.09, so a 0.20 margin is ~2 SE.
# --------------------------------------------------------------------------


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DATA / "raw_completions.jsonl"))
    ap.add_argument("--out-dir", default=str(DATA))
    ap.add_argument("--min-pronoun-count", type=int, default=1,
                    help="1 is safe here because analyze() also requires the "
                         "name's gender to agree with the pronouns (lesson #10).")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def calibration_decision(n, n_female):
    """Apply the pre-registered rule. Returns (p_female, target_gender|None, why)."""
    if n < MIN_CALIBRATION_N:
        return (n_female / n if n else None), None, "too_few_usable"
    p = n_female / n
    if abs(p - 0.5) < CALIBRATION_MARGIN - 1e-9:   # tolerance: p=0.70 must count as >= 0.70
        return p, None, "near_parity"
    return p, ("male" if p > 0.5 else "female"), "targeted"


def in_train_cell(r):
    return r["split"] == "train" and r["template_split"] == "train"


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    nlp = get_nlp()

    records = [json.loads(line) for line in open(args.input)]

    # 1. Analyze ----------------------------------------------------------
    funnel = Counter()
    with open(out_dir / "analysis.jsonl", "w") as f:
        for r in records:
            a = analyze(r, args.min_pronoun_count, nlp)
            r["_a"] = a
            funnel["total"] += 1
            funnel[f"excluded_{a['reason']}" if a["reason"] else "usable"] += 1
            f.write(json.dumps({k: r[k] for k in (
                "occupation", "stereotype", "split", "template_id",
                "template_split", "sample_id")} | a) + "\n")

    usable = [r for r in records if r["_a"]["reason"] is None]
    name_pool = build_name_pool((r["_a"]["name"], r["_a"]["name_gender"]) for r in usable)

    # 2. Calibrate ----------------------------------------------------------
    cal = defaultdict(Counter)
    for r in usable:
        if in_train_cell(r) and r["sample_id"] < N_CALIBRATION:
            cal[r["occupation"]][r["_a"]["gender"]] += 1
    train_occs = sorted({r["occupation"] for r in records if r["split"] == "train"})
    calibration = {}
    for occ in train_occs:
        n = cal[occ]["male"] + cal[occ]["female"]
        p, target, why = calibration_decision(n, cal[occ]["female"])
        stereotype = next(r["stereotype"] for r in records if r["occupation"] == occ)
        calibration[occ] = {"stereotype": stereotype, "n": n, "p_female": p,
                            "target_gender": target, "decision": why}
    with open(out_dir / "calibration.json", "w") as f:
        json.dump({"rule": {"N_CALIBRATION": N_CALIBRATION,
                            "MIN_CALIBRATION_N": MIN_CALIBRATION_N,
                            "CALIBRATION_MARGIN": CALIBRATION_MARGIN},
                   "occupations": calibration}, f, indent=2)

    # 3. Pair ---------------------------------------------------------------
    pairs, swap_fail = [], Counter()
    source_outcome = Counter()
    for r in usable:
        if not in_train_cell(r) or r["sample_id"] < N_CALIBRATION:
            continue
        target = calibration[r["occupation"]]["target_gender"]
        if target is None:
            source_outcome["occupation_not_targeted"] += 1
            continue
        if r["_a"]["gender"] == target:
            source_outcome["already_target_gender"] += 1
            continue
        new_name = sample_name(name_pool[target], rng, exclude=r["_a"]["name"])
        swapped, log = swap_gender(r["completion"], target, new_name, nlp)
        if swapped is None:
            swap_fail[log] += 1
            source_outcome["swap_failed"] += 1
            continue
        source_outcome["paired"] += 1
        pairs.append({
            "prompt": r.get("messages") or build_messages(r["prompt"]),
            "chosen": [{"role": "assistant", "content": swapped}],
            "rejected": [{"role": "assistant", "content": r["completion"]}],
            "occupation": r["occupation"],
            "stereotype": r["stereotype"],
            "template_id": r["template_id"],
            "source_sample_id": r["sample_id"],
            "rejected_gender": r["_a"]["gender"],
            "chosen_gender": target,
            "old_name": r["_a"]["name"],
            "new_name": new_name if r["_a"]["name_gender"] != "unisex" else r["_a"]["name"],
            "swap_log": log,
            "other_person_flag": r["_a"]["other_person_flag"],
        })

    with open(out_dir / "dpo_pairs.jsonl", "w") as f:
        for i, p in enumerate(pairs):
            f.write(json.dumps({"pair_id": i, **p}) + "\n")

    stats = {
        "args": vars(args),
        "funnel": dict(funnel),
        "pair_source_outcome": dict(source_outcome),
        "swap_failures": dict(swap_fail),
        "n_pairs": len(pairs),
        "pairs_by_direction": dict(Counter(f"{p['rejected_gender']}->{p['chosen_gender']}" for p in pairs)),
        "pairs_by_occupation": dict(Counter(p["occupation"] for p in pairs)),
        "name_pool_sizes": {g: len(v) for g, v in name_pool.items()},
    }
    with open(out_dir / "pair_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats["funnel"], indent=2))
    print("\nCalibration (train occupations x train templates, calibration half):")
    for occ, c in calibration.items():
        p = "  n/a" if c["p_female"] is None else f"{c['p_female']:.2f}"
        print(f"  {occ:<18}{c['stereotype']:<8} n={c['n']:<3} p_female={p}  "
              f"-> {c['decision']}" + (f" (push to {c['target_gender']})" if c["target_gender"] else ""))
    print("\nPair source:", dict(source_outcome), "| swap failures:", dict(swap_fail))
    print(f"Wrote {len(pairs)} pairs, by direction {stats['pairs_by_direction']}")
    if len(pairs) < 200:
        print("WARNING: fewer than 200 pairs -- check the funnel before training.")


if __name__ == "__main__":
    main()
