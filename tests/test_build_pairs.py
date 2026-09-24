"""The pre-registered calibration rule."""
from build_pairs import CALIBRATION_MARGIN, MIN_CALIBRATION_N, calibration_decision


def test_too_few_usable():
    assert calibration_decision(MIN_CALIBRATION_N - 1, 0)[1] is None


def test_near_parity_not_targeted():
    assert calibration_decision(20, 12)[2] == "near_parity"          # p = 0.60


def test_direction_is_towards_minority():
    assert calibration_decision(20, 2)[1] == "female"                # p = 0.10
    assert calibration_decision(20, 18)[1] == "male"                 # p = 0.90


def test_margin_boundary():
    n = 100
    at_margin = int((0.5 + CALIBRATION_MARGIN) * n)                  # p = 0.70
    assert calibration_decision(n, at_margin)[1] == "male"
