from __future__ import annotations

import numpy as np
import pandas as pd

from galaxies.agent.embedding_evaluation import (
    BASE_CATEGORICAL,
    BASE_NUMERIC,
    EmbeddingConfig,
    _fit_transform,
)


def _rows(size: int) -> pd.DataFrame:
    frame = pd.DataFrame(index=range(size))
    for position, column in enumerate(BASE_NUMERIC):
        frame[column] = np.arange(size, dtype=float) + position
    for position, column in enumerate(BASE_CATEGORICAL):
        frame[column] = [f"category-{(row + position) % 2}" for row in range(size)]
    frame["_thumb_embedding"] = [np.array([row, row % 3, 1.0], dtype=float) for row in range(size)]
    return frame


def test_cluster_features_are_fit_on_train_and_transform_test() -> None:
    train = _rows(8)
    test = _rows(3)
    config = EmbeddingConfig(
        "cluster-test", cluster_embedding="thumb", cluster_count=3
    )

    x_train, x_test, components = _fit_transform(train, test, config, seed=42)

    assert components == 0
    assert x_train.shape[0] == len(train)
    assert x_test.shape[0] == len(test)
    # As três últimas colunas são o one-hot do K-Means ajustado exclusivamente
    # no treino; o teste só recebe o cluster mais próximo aprendido.
    assert np.allclose(x_train[:, -3:].sum(axis=1), 1.0)
    assert np.allclose(x_test[:, -3:].sum(axis=1), 1.0)
