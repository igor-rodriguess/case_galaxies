from __future__ import annotations

import numpy as np

from galaxies.modeling.evidence import paired_bootstrap_mae_difference


def test_bootstrap_detects_consistently_better_candidate() -> None:
    actual = np.arange(1.0, 21.0)
    candidate = actual + 0.1
    baseline = actual + 2.0
    evidence = paired_bootstrap_mae_difference(
        actual, candidate, baseline, iterations=1_000
    )
    assert evidence["mae_difference"] < 0
    assert evidence["ci95_upper"] < 0
    assert evidence["statistically_better"] is True
