"""Selection rule, GAP statistics and the training split: the evaluation logic
that the reported results depend on."""
import random

import numpy as np

from analyze_eval import gap_cell, paired_gap_diff, parity_distance, per_occupation
from evaluate import MAX_USABLE_DROP, select_checkpoint
from train_dpo import stratified_split


def m(step, usable, gap):
    return {"step": step, "usable_rate": usable,
            "bias": {"gap": {"train_occ|train_tmpl": {"gap": gap}}}}


def test_selection_picks_smallest_abs_gap_among_guarded():
    base = {"usable_rate": 0.57}
    ckpts = [m(10, 0.59, 0.48), m(60, 0.82, 0.17), m(100, 0.76, 0.04), m(153, 0.77, -0.10)]
    assert select_checkpoint(base, ckpts)["step"] == 100


def test_selection_guard_excludes_low_quality_checkpoints():
    base = {"usable_rate": 0.57}
    ckpts = [m(20, 0.04, 0.00), m(10, 0.57 - MAX_USABLE_DROP + 0.01, 0.40)]
    assert select_checkpoint(base, ckpts)["step"] == 10        # the perfect GAP fails the guard
    assert select_checkpoint(base, [m(20, 0.04, 0.0)]) is None   # run 1: nothing passes


def test_selection_overshoot_counts_as_distance_and_ties_go_early():
    base = {"usable_rate": 0.5}
    assert select_checkpoint(base, [m(40, 0.6, -0.05), m(60, 0.6, 0.06)])["step"] == 40
    assert select_checkpoint(base, [m(60, 0.6, 0.05), m(40, 0.6, -0.05)])["step"] == 40


def rows(spec):
    """spec: {(occupation, stereotype): (n_female, n_male)} -> usable rows (train cell)."""
    out = []
    for (occ, st), (nf, nm) in spec.items():
        for g, n in (("female", nf), ("male", nm)):
            out += [{"occupation": occ, "stereotype": st, "split": "train",
                     "template_split": "train", "_a": {"gender": g}}] * n
    return out


SPEC = {("nurse", "female"): (18, 2), ("teacher", "female"): (16, 4),
        ("plumber", "male"): (2, 18), ("engineer", "male"): (4, 16)}


def test_gap_point_estimate_and_ci_contains_it():
    g = gap_cell(rows(SPEC), {"train"}, {"train"}, np.random.default_rng(0))
    assert abs(g["gap"] - (34 / 40 - 6 / 40)) < 1e-9
    lo, hi = g["ci95_cluster"]
    assert lo <= g["gap"] <= hi and g["interpretable"]


def test_paired_difference_of_identical_models_is_zero():
    # texts are resampled independently per model, so the CI has width, but
    # it must be centred on 0
    r = rows(SPEC)
    d = paired_gap_diff(r, r, {"train"}, {"train"}, np.random.default_rng(0))
    lo, hi = d["ci95_cluster"]
    assert d["diff"] == 0 and lo < 0 < hi and abs(lo + hi) < 0.05


def test_parity_distance():
    po = per_occupation(rows(SPEC))
    value, n = parity_distance(po)
    assert n == 4 and abs(value - np.mean([0.4, 0.3, 0.4, 0.3])) < 1e-9


def test_stratified_split_is_deterministic_and_per_occupation():
    pairs = [{"pair_id": i, "occupation": o} for i, o in enumerate(["a"] * 20 + ["b"] * 6 + ["c"] * 3)]
    t1, e1 = stratified_split(pairs, 0.10, 42)
    t2, e2 = stratified_split(pairs, 0.10, 42)
    assert [p["pair_id"] for p in e1] == [p["pair_id"] for p in e2]
    held = sorted(p["occupation"] for p in e1)
    assert held == ["a", "a", "b"]                     # round(2.0), at least 1 if n >= 5, none for 3
    assert len(t1) + len(e1) == len(pairs)
