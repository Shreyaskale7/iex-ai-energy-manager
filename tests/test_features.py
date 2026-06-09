"""Feature engineering tests."""

from iex_forecast.domain.constants import FEATURE_COLUMNS
from iex_forecast.features.builder import FeatureBuilder
from iex_forecast.features.training_matrix import TrainingMatrixBuilder


def test_feature_builder_produces_expected_columns(sample_rtm_df):
    builder = FeatureBuilder()
    result = builder.transform(sample_rtm_df, horizon=1)
    for col in FEATURE_COLUMNS:
        assert col in result.columns


def test_training_matrix_non_empty(sample_rtm_df):
    matrix_builder = TrainingMatrixBuilder()
    X, y = matrix_builder.build_for_horizon(sample_rtm_df, horizon=4)
    assert len(X) > 0
    assert len(y) == len(X)
    assert "horizon" in X.columns
    assert X["horizon"].iloc[0] == 4


def test_compute_metrics():
    import numpy as np

    metrics = TrainingMatrixBuilder.compute_metrics(
        np.array([100.0, 200.0]),
        np.array([110.0, 180.0]),
    )
    assert metrics["mae"] == 15.0
    assert metrics["rmse"] > 0
