from __future__ import annotations

import numpy as np
import pandas as pd

from galaxies.modeling.relative_target import AgeChannelBaseline, fit_relative_target


def test_relative_target_is_fitted_only_on_supplied_training_rows() -> None:
    frame = pd.DataFrame(
        {
            "target_age_days": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
            "channel_id": ["known", "known", "known", "known", "new", "new"],
        }
    )
    views = np.array([100.0, 180.0, 300.0, 450.0, 20_000.0, 40_000.0])
    baseline, residual = fit_relative_target(frame, [0, 1, 2, 3], views)
    assert set(baseline.channel_effects) == {"known"}
    assert residual.shape == (6,)
    assert np.all(np.isfinite(residual))


def test_unseen_channel_uses_global_age_curve() -> None:
    train = pd.DataFrame(
        {
            "target_age_days": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
            "channel_id": ["a", "a", "a", "b", "b", "b"],
        }
    )
    model = AgeChannelBaseline().fit(
        train, np.log1p(np.array([100, 170, 260, 500, 800, 1_200], dtype=float))
    )
    unseen = pd.DataFrame({"target_age_days": [4.0], "channel_id": ["c"]})
    prediction = model.predict(unseen)
    assert prediction.shape == (1,)
    assert np.isfinite(prediction[0])
