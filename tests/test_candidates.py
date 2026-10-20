from galaxies.modeling.candidates import make_classifier, make_regressor


def test_requested_regression_candidates_are_registered() -> None:
    names = [
        "linear_regression",
        "gradient_boosting",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "mlp",
    ]
    assert all(make_regressor(name, 42) is not None for name in names)


def test_logistic_regression_is_registered_as_classifier() -> None:
    classifier = make_classifier("logistic_regression", 42)
    assert classifier.__class__.__name__ == "LogisticRegression"
