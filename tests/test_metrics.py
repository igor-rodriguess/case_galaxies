from __future__ import annotations

import pytest

from galaxies.modeling.metrics import regression_metrics, relative_regression_metrics


def test_regression_metrics_include_raw_and_scale_robust_errors() -> None:
    metrics = regression_metrics(
        [100.0, 200.0],
        [110.0, 180.0],
        ["channel-a", "channel-a"],
    )

    assert metrics["mae_raw"] == pytest.approx(15.0)
    assert metrics["rmse_raw"] == pytest.approx(15.8113883008)
    assert metrics["median_ae_raw"] == pytest.approx(15.0)
    assert metrics["bias_raw"] == pytest.approx(-5.0)
    assert metrics["rmsle"] == pytest.approx(metrics["rmse_log"])
    assert metrics["wape"] == pytest.approx(0.1)
    assert metrics["smape"] == pytest.approx(0.1002506266)


def test_regression_metrics_handle_zero_truth_and_prediction() -> None:
    metrics = regression_metrics([0.0, 10.0], [0.0, 5.0], ["a", "a"])

    assert metrics["wape"] == pytest.approx(0.5)
    assert metrics["smape"] == pytest.approx(1 / 3)


def test_relative_metrics_preserve_negative_targets() -> None:
    metrics = relative_regression_metrics(
        [-2.0, -1.0, 1.0, 2.0],
        [-1.5, -0.5, 0.5, 1.5],
        ["a", "a", "b", "b"],
    )
    assert metrics["mae_log"] == pytest.approx(0.5)
    assert metrics["bias_log"] == pytest.approx(0.0)
    assert metrics["r2_log"] is not None
