"""Unit tests for the pairing logic. Run: python -m pytest tests/ -q
Each test pins down a bug found in a past review round -- if one of these
fails, a known failure mode is back."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from label_completions import (  # noqa: E402
    classify, exclusion_reason, has_first_person_narration, make_pairs,
)


def rec(text, **kw):
    return {"completion": text, "truncated": False, **kw}


def test_mixed_pronouns_never_counter():
    # Old score: 5 she - 4 he = +1 -> "counter-stereotypical". Misattribution.
    text = "She fixed it. " * 5 + "He watched. " * 4
    assert classify(text, "male") == "mixed"


def test_pure_classes():
    assert classify("She wired the panel. Her tools...", "male") == "counter"
    assert classify("He wired the panel. His tools...", "male") == "stereo"
    assert classify("He checked the chart. His shift...", "female") == "counter"


def test_two_counter_completions_do_not_pair():
    # Old filter (best > 0) paired +5 against +1: degenerate, mirrored.
    counters = [rec("She did it. " * 5, sample_id=0), rec("She did.", sample_id=1)]
    pairs, _ = make_pairs(counters, [], max_pairs=2, rng=random.Random(0))
    assert pairs == []


def test_pairs_are_one_to_one_and_length_bounded():
    c = [rec("She " + "x" * 100, sample_id=i) for i in range(3)]
    s = [rec("He " + "y" * 100, sample_id=10), rec("He " + "y" * 1000, sample_id=11)]
    pairs, n_len = make_pairs(c, s, max_pairs=3, rng=random.Random(0))
    assert len(pairs) == 1                       # only one length-compatible stereo
    assert len({id(r) for _, r in pairs}) == len(pairs)


def test_exclusions():
    assert exclusion_reason(rec("As an AI language model, she ..."), 1) == "refusal"
    assert exclusion_reason(rec("Mark Zuckerberg walked in. He smiled. He sat."), 1) == "real_person"
    assert exclusion_reason({"completion": "She ran. She sat.", "truncated": True}, 1) == "truncated"
    assert exclusion_reason(rec("She ran."), 2) == "no_signal"
    assert exclusion_reason(rec("She ran. She sat."), 2) is None


def test_first_person_only_outside_dialogue():
    assert not has_first_person_narration('She said, "I love my job." She smiled.')
    assert has_first_person_narration("I watched her work. She was fast.")
    assert has_first_person_narration("My colleague said she was fast.")
