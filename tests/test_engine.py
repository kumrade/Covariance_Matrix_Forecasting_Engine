import numpy as np

from main import demonstration_prices
from src.covariance_engine import (
    compare_models,
    covariance_to_correlation,
    dcc_garch_covariance,
    ewma_covariance,
    historical_covariance,
    ledoit_wolf_covariance,
    log_returns,
    stress_correlations,
)


def test_covariance_pipeline():
    returns = log_returns(demonstration_prices(days=350, assets=5, seed=7))
    matrices = [historical_covariance(returns), ewma_covariance(returns)]
    ledoit_wolf, intensity = ledoit_wolf_covariance(returns)
    matrices.append(ledoit_wolf)
    dcc = dcc_garch_covariance(returns)
    matrices.append(dcc.covariance)

    assert 0.0 <= intensity <= 1.0
    assert dcc.a >= 0 and dcc.b >= 0 and dcc.a + dcc.b < 1.0
    for matrix in matrices:
        assert np.allclose(matrix, matrix.T)
        assert np.linalg.eigvalsh(matrix).min() >= -1e-8
        assert np.allclose(np.diag(covariance_to_correlation(matrix)), 1.0)


def test_comparison_and_stress_outputs():
    returns = log_returns(demonstration_prices(days=350, assets=5, seed=9))
    expected_returns = returns.mean().to_numpy() * 252
    comparison = compare_models(returns, expected_returns, maximum_weight=0.4)
    stress = stress_correlations(returns)

    assert set(comparison["model"]) == {"Historical", "EWMA", "Ledoit-Wolf", "DCC-GARCH"}
    assert np.isfinite(comparison[["expected_return", "volatility", "sharpe"]]).all().all()
    assert stress["stress_observations"] > 0
    assert -1.0 <= stress["average_stress"] <= 1.0
