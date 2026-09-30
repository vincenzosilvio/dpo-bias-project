"""
Filters raw completions, classifies each one by the gender its pronouns
assign to the occupation-holder, and builds DPO preference pairs:
chosen = counter-stereotypical, rejected = stereotypical, same prompt.

Classification (per completion, for a male- or female-coded occupation):
    "counter"  -- ONLY counter-stereotypical pronouns (e.g. only she/her
                  for "engineer")
    "stereo"   -- ONLY stereotypical pronouns
    "mixed"    -- both he-family and she-family pronouns: EXCLUDED
    no pronoun signal, refusal, first-person narration, truncated,
    real-person mention: EXCLUDED

Why purity instead of the old score = (counter - stereo) difference:
  * A mixed completion is exactly where pronoun misattribution lives (a
    second character's pronouns credited to the occupation-holder). The old
    score let "5 she + 4 he" count as counter-stereotypical. Now it can
    never enter a pair. This turns lesson #6 in the README from "hope manual
    review catches it" into a filter, and leaves manual review to check the
    residual cases (a single-gender completion whose pronouns still belong
    to someone else).
  * The old filter only required best_score > 0. Two counter-stereotypical
    completions (+5 vs +1) still formed a pair -- the same degenerate pair
    as lesson #5, mirrored. Pairing counter vs stereo by class rules it out.

Pairing scope:
  * split == "heldout" occupations are never paired (generalisation test).
  * "balanced" occupations are never paired (control group). Under the old
    score they could not produce pairs anyway (score was always <= 0), but
    silently; now it is explicit and by design.

Output:
  data/dpo_pairs.jsonl -- TRL conversational format:
      prompt   : [system msg, user msg]  (exact generation context)
      chosen   : [assistant msg]
      rejected : [assistant msg]
      + metadata (occupation, stereotype, template_id, sample ids, counts)
  data/label_stats.json -- the filtering funnel, read by review_sample.py

Usage (repo root):
    python src/label_completions.py
"""

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from occupations import build_messages  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

MALE_PRONOUNS = re.compile(r"\b(he|him|his|himself)\b", re.IGNORECASE)
FEMALE_PRONOUNS = re.compile(r"\b(she|her|hers|herself)\b", re.IGNORECASE)

# Safety net for real, well-known people (observed: Mark Zuckerberg for
# "CEO"). Not exhaustive -- add names found during manual review and rerun.
REAL_PERSON_BLOCKLIST = [
    "mark zuckerberg", "elon musk", "bill gates", "jeff bezos",
    "tim cook", "sundar pichai", "satya nadella", "sam altman",
    "larry page", "sergey brin", "jack ma", "warren buffett",
    "steve jobs", "richard branson", "indra nooyi", "mary barra",
    "ginni rometty", "sheryl sandberg",
    "florence nightingale", "marie curie", "albert einstein",
    "amelia earhart", "neil armstrong",
]

# Lesson #1 was fixed with a system prompt; this catches the residue so it
# never reaches a pair, and review_sample.py reports how much residue there is.
REFUSAL_PATTERN = re.compile(
    r"\b(as an ai|ai language model|i'm sorry|i am sorry|i cannot|i can't "
    r"(help|assist|write)|i'm unable|i am unable)\b",
    re.IGNORECASE,
)

# Quoted dialogue is removed before checking for first-person narration:
# "I" inside dialogue is legitimate, "I" in the narration means the model
# wrote as a narrator/character, and then he/she likely refer to others.
QUOTED = re.compile(r'"[^"]*"|“[^”]*”')
FIRST_PERSON = re.compile(r"\bI\b|\b(?i:my|me|myself)\b")

# Max length ratio (characters) between chosen and rejected, so DPO does
# not learn a length preference instead of the gender contrast.
MAX_LENGTH_RATIO = 1.6


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(REPO_ROOT / "data" / "raw_completions.jsonl"))
    parser.add_argument("--output", default=str(REPO_ROOT / "data" / "dpo_pairs.jsonl"))
    parser.add_argument("--stats-output", default=str(REPO_ROOT / "data" / "label_stats.json"))
    parser.add_argument("--min-pronoun-count", type=int, default=2,
                        help="Minimum gendered pronouns for a completion to "
                             "carry signal. 2 rather than 1: a single pronoun "
                             "is the most likely to be a stray reference.")
    parser.add_argument("--max-pairs-per-group", type=int, default=2,
                        help="Max pairs per (occupation, template). Each "
                             "completion is used at most once. Keep low: "
                             "pairs from one group are strongly correlated.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def pronoun_counts(text):
    return {
        "male": len(MALE_PRONOUNS.findall(text)),
        "female": len(FEMALE_PRONOUNS.findall(text)),
    }


def mentions_real_person(text):
    lowered = text.lower()
    return any(name in lowered for name in REAL_PERSON_BLOCKLIST)


def has_first_person_narration(text):
    return bool(FIRST_PERSON.search(QUOTED.sub(" ", text)))


def exclusion_reason(record, min_pronoun_count):
    """Return why a completion can't be used at all, or None if usable."""
    text = record["completion"]
    if not text.strip():
        return "empty"
    if record.get("truncated", False):
        return "truncated"
    if mentions_real_person(text):
        return "real_person"
    if REFUSAL_PATTERN.search(text):
        return "refusal"
    if has_first_person_narration(text):
        return "first_person"
    c = pronoun_counts(text)
    if c["male"] + c["female"] < min_pronoun_count:
        return "no_signal"
    return None


def classify(text, stereotype):
    """'counter', 'stereo' or 'mixed' for male/female-coded occupations."""
    c = pronoun_counts(text)
    if c["male"] > 0 and c["female"] > 0:
        return "mixed"
    pronoun_gender = "male" if c["male"] > 0 else "female"
    return "stereo" if pronoun_gender == stereotype else "counter"


def length_ratio(a, b):
    la, lb = max(len(a), 1), max(len(b), 1)
    return max(la, lb) / min(la, lb)


def make_pairs(counters, stereos, max_pairs, rng):
    """
    Greedy one-to-one matching: each counter completion is paired with the
    unused stereo completion closest in length, if within MAX_LENGTH_RATIO.
    Returns (pairs, n_rejected_for_length).
    """
    counters = counters[:]
    rng.shuffle(counters)
    available = stereos[:]
    pairs, n_length = [], 0
    for ch in counters:
        if len(pairs) >= max_pairs or not available:
            break
        best = min(available, key=lambda r: length_ratio(ch["completion"], r["completion"]))
        if length_ratio(ch["completion"], best["completion"]) > MAX_LENGTH_RATIO:
            n_length += 1
            continue
        available.remove(best)
        pairs.append((ch, best))
    return pairs, n_length


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    funnel = Counter()
    classes = Counter()
    groups = defaultdict(lambda: {"counter": [], "stereo": []})

    with open(args.input) as f_in:
        for line in f_in:
            r = json.loads(line)
            funnel["total"] += 1
            r.setdefault("split", "train")
            r.setdefault("template_id", -1)
            reason = exclusion_reason(r, args.min_pronoun_count)
            if reason:
                funnel[f"excluded_{reason}"] += 1
                continue
            if r["stereotype"] == "balanced":
                funnel["not_paired_control"] += 1
                continue
            cls = classify(r["completion"], r["stereotype"])
            classes[f"{r['split']}/{cls}"] += 1
            if cls == "mixed":
                funnel["excluded_mixed"] += 1
                continue
            if r["split"] != "train":
                funnel["not_paired_heldout"] += 1
                continue
            funnel["eligible"] += 1
            groups[(r["occupation"], r["prompt"])][cls].append(r)

    pairs = []
    group_outcomes = Counter()
    for (occupation, prompt), g in sorted(groups.items()):
        if not g["counter"] or not g["stereo"]:
            group_outcomes["no_counter" if not g["counter"] else "no_stereo"] += 1
            continue
        matched, n_length = make_pairs(g["counter"], g["stereo"], args.max_pairs_per_group, rng)
        funnel["pair_candidates_rejected_length"] += n_length
        group_outcomes["paired" if matched else "length_mismatch_only"] += 1
        for ch, rj in matched:
            messages = ch.get("messages") or build_messages(prompt)
            pairs.append({
                "prompt": messages,
                "chosen": [{"role": "assistant", "content": ch["completion"]}],
                "rejected": [{"role": "assistant", "content": rj["completion"]}],
                "occupation": occupation,
                "stereotype": ch["stereotype"],
                "template_id": ch["template_id"],
                "chosen_sample_id": ch["sample_id"],
                "rejected_sample_id": rj["sample_id"],
                "chosen_pronouns": pronoun_counts(ch["completion"]),
                "rejected_pronouns": pronoun_counts(rj["completion"]),
            })

    with open(args.output, "w") as f_out:
        for i, pair in enumerate(pairs):
            f_out.write(json.dumps({"pair_id": i, **pair}) + "\n")

    stats = {
        "args": vars(args),
        "funnel": dict(funnel),
        "classes": dict(classes),
        "train_groups": len(groups),
        "group_outcomes": dict(group_outcomes),
        "n_pairs": len(pairs),
        "pairs_by_stereotype": dict(Counter(p["stereotype"] for p in pairs)),
    }
    with open(args.stats_output, "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats["funnel"], indent=2))
    print("Train groups:", dict(group_outcomes))
    print(f"Wrote {len(pairs)} DPO pairs to {args.output} "
          f"(by stereotype: {stats['pairs_by_stereotype']})")
    if len(pairs) < 100:
        print("WARNING: fewer than 100 pairs. Do not train yet -- see "
              "review_sample.py output for WHERE the funnel loses data.")


if __name__ == "__main__":
    main()
