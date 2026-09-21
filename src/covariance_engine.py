"""Reusable covariance forecasting and portfolio-risk research logic."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert an aligned positive price panel into complete log returns."""
    if prices.empty or (prices <= 0).any().any():
        raise ValueError("Prices must be non-empty and strictly positive.")
    aligned = prices.sort_index().ffill().dropna()
    return np.log(aligned / aligned.shift(1)).dropna()


def nearest_psd(matrix: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    return vectors @ np.diag(np.clip(values, floor, None)) @ vectors.T


def covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.clip(np.diag(covariance), 1e-18, None))
    correlation = covariance / np.outer(scale, scale)
    correlation = np.clip(correlation, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def historical_covariance(returns: pd.DataFrame) -> np.ndarray:
    return nearest_psd(returns.cov().to_numpy() * TRADING_DAYS)


def ewma_covariance(returns: pd.DataFrame, decay: float = 0.94) -> np.ndarray:
    if not 0.0 < decay < 1.0:
        raise ValueError("EWMA decay must lie between zero and one.")
    observations = returns.to_numpy(float)
    covariance = np.cov(observations, rowvar=False)
    for observation in observations:
        vector = observation[:, None]
        covariance = decay * covariance + (1.0 - decay) * (vector @ vector.T)
    return nearest_psd(covariance * TRADING_DAYS)


def ledoit_wolf_covariance(returns: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Constant-correlation Ledoit-Wolf shrinkage estimate."""
    x = returns.to_numpy(float)
    x -= x.mean(axis=0)
    sample_count, asset_count = x.shape
    sample = (x.T @ x) / sample_count
    variances = np.clip(np.diag(sample), 1e-18, None)
    standard_deviation = np.sqrt(variances)
    scale = np.outer(standard_deviation, standard_deviation)
    correlation = sample / scale
    average = (correlation.sum() - asset_count) / (asset_count * (asset_count - 1))

    target = average * scale
    np.fill_diagonal(target, variances)
    squared = x**2
    pi_matrix = (squared.T @ squared) / sample_count - sample**2
    pi_hat = pi_matrix.sum()
    third = (x**3).T @ x / sample_count
    theta_i = third - variances[:, None] * sample
    theta_j = third.T - variances[None, :] * sample
    ratio_ji = standard_deviation[None, :] / standard_deviation[:, None]
    ratio_ij = standard_deviation[:, None] / standard_deviation[None, :]
    off_diagonal = 0.5 * average * (ratio_ji * theta_i + ratio_ij * theta_j)
    np.fill_diagonal(off_diagonal, 0.0)
    rho_hat = np.diag(pi_matrix).sum() + off_diagonal.sum()
    gamma_hat = np.sum((target - sample) ** 2)
    intensity = float(np.clip((pi_hat - rho_hat) / max(gamma_hat, 1e-18) / sample_count, 0.0, 1.0))
    covariance = intensity * target + (1.0 - intensity) * sample
    return nearest_psd(covariance * TRADING_DAYS), intensity


@dataclass
class GarchFit:
    omega: float
    alpha: float
    beta: float
    conditional_variance: np.ndarray
    residuals: np.ndarray

    def forecast_variance(self) -> float:
        return float(
            self.omega
            + self.alpha * self.residuals[-1] ** 2
            + self.beta * self.conditional_variance[-1]
        )


def fit_garch11(series: np.ndarray) -> GarchFit:
    residuals = np.asarray(series, float)
    residuals -= residuals.mean()
    initial_variance = max(float(np.var(residuals)), 1e-10)
    count = len(residuals)

    def recursion(parameters: np.ndarray) -> np.ndarray:
        omega, alpha, beta = parameters
        variance = np.empty(count)
        variance[0] = initial_variance
        for index in range(1, count):
            variance[index] = (
                omega + alpha * residuals[index - 1] ** 2 + beta * variance[index - 1]
            )
        return variance

    def negative_log_likelihood(parameters: np.ndarray) -> float:
        omega, alpha, beta = parameters
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e12
        variance = recursion(parameters)
        if np.any(variance <= 0) or not np.all(np.isfinite(variance)):
            return 1e12
        return float(0.5 * np.sum(np.log(2 * np.pi * variance) + residuals**2 / variance))

    initial = np.array([initial_variance * 0.05, 0.05, 0.90])
    result = minimize(
        negative_log_likelihood,
        initial,
        method="L-BFGS-B",
        bounds=[(1e-12, None), (0.0, 0.998), (0.0, 0.998)],
    )
    if result.success and result.x[1] + result.x[2] < 0.999:
        omega, alpha, beta = map(float, result.x)
    else:
        omega, alpha, beta = initial_variance * 0.02, 0.06, 0.92
    variance = recursion(np.array([omega, alpha, beta]))
    return GarchFit(omega, alpha, beta, variance, residuals)


@dataclass
class DccForecast:
    covariance: np.ndarray
    correlation: np.ndarray
    a: float
    b: float
    garch_fits: list[GarchFit]


def dcc_garch_covariance(returns: pd.DataFrame) -> DccForecast:
    x = returns.to_numpy(float)
    sample_count, asset_count = x.shape
    fits = [fit_garch11(x[:, column]) for column in range(asset_count)]
    volatility = np.column_stack([np.sqrt(fit.conditional_variance) for fit in fits])
    standardized = np.column_stack([fit.residuals for fit in fits]) / volatility
    unconditional = nearest_psd(np.cov(standardized, rowvar=False))

    def objective(parameters: np.ndarray) -> float:
        a, b = parameters
        if a < 0 or b < 0 or a + b >= 0.999:
            return 1e12
        q = unconditional.copy()
        loss = 0.0
        for index in range(sample_count):
            scale = np.sqrt(np.clip(np.diag(q), 1e-18, None))
            correlation = nearest_psd(q / np.outer(scale, scale))
            sign, log_determinant = np.linalg.slogdet(correlation)
            if sign <= 0:
                return 1e12
            observation = standardized[index]
            loss += log_determinant + observation @ np.linalg.solve(correlation, observation)
            vector = observation[:, None]
            q = (1.0 - a - b) * unconditional + a * (vector @ vector.T) + b * q
        return float(0.5 * loss)

    result = minimize(
        objective,
        np.array([0.02, 0.95]),
        method="L-BFGS-B",
        bounds=[(0.0, 0.5), (0.0, 0.998)],
    )
    if result.success and result.x.sum() < 0.999:
        a, b = map(float, result.x)
    else:
        a, b = 0.02, 0.95

    q = unconditional.copy()
    for observation in standardized:
        vector = observation[:, None]
        q = (1.0 - a - b) * unconditional + a * (vector @ vector.T) + b * q
    scale = np.sqrt(np.clip(np.diag(q), 1e-18, None))
    correlation = q / np.outer(scale, scale)
    correlation = covariance_to_correlation(nearest_psd(correlation))
    forecast_volatility = np.array(
        [math.sqrt(max(fit.forecast_variance(), 1e-18)) for fit in fits]
    )
    diagonal = np.diag(forecast_volatility)
    covariance = nearest_psd(diagonal @ correlation @ diagonal * TRADING_DAYS)
    return DccForecast(covariance, correlation, a, b, fits)


def stress_correlations(returns: pd.DataFrame, tail_percent: float = 10.0) -> dict:
    if not 0.0 < tail_percent < 50.0:
        raise ValueError("Tail percentage must lie between zero and fifty.")
    x = returns.to_numpy(float)
    market_proxy = x.mean(axis=1)
    threshold = np.percentile(market_proxy, tail_percent)
    stressed = market_proxy <= threshold
    normal = ~stressed
    normal_corr = np.corrcoef(x[normal], rowvar=False)
    stress_corr = np.corrcoef(x[stressed], rowvar=False)

    def average_off_diagonal(matrix: np.ndarray, absolute: bool = False) -> float:
        values = matrix[np.triu_indices_from(matrix, 1)]
        return float(np.mean(np.abs(values) if absolute else values))

    return {
        "normal_correlation": normal_corr,
        "stress_correlation": stress_corr,
        "average_normal": average_off_diagonal(normal_corr),
        "average_stress": average_off_diagonal(stress_corr),
        "average_absolute_normal": average_off_diagonal(normal_corr, True),
        "average_absolute_stress": average_off_diagonal(stress_corr, True),
        "stress_observations": int(stressed.sum()),
        "threshold": float(threshold),
    }


def maximum_sharpe(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    risk_free_rate: float = 0.06,
    maximum_weight: float = 1.0,
) -> dict:
    covariance = nearest_psd(covariance)
    asset_count = len(expected_returns)
    if maximum_weight * asset_count < 1.0:
        raise ValueError("Maximum weight is infeasible for this asset count.")

    def negative_sharpe(weights: np.ndarray) -> float:
        portfolio_return = float(weights @ expected_returns)
        volatility = math.sqrt(max(float(weights @ covariance @ weights), 1e-18))
        return -(portfolio_return - risk_free_rate) / volatility

    result = minimize(
        negative_sharpe,
        np.full(asset_count, 1.0 / asset_count),
        method="SLSQP",
        bounds=[(0.0, maximum_weight)] * asset_count,
        constraints=[{"type": "eq", "fun": lambda weights: weights.sum() - 1.0}],
        options={"maxiter": 500, "ftol": 1e-10},
    )
    weights = result.x if result.success else np.full(asset_count, 1.0 / asset_count)
    weights = np.clip(weights, 0.0, maximum_weight)
    weights /= weights.sum()
    portfolio_return = float(weights @ expected_returns)
    volatility = math.sqrt(max(float(weights @ covariance @ weights), 1e-18))
    return {
        "weights": weights,
        "return": portfolio_return,
        "volatility": volatility,
        "sharpe": (portfolio_return - risk_free_rate) / volatility,
        "success": bool(result.success),
    }


def compare_models(
    returns: pd.DataFrame,
    expected_returns: np.ndarray,
    risk_free_rate: float = 0.06,
    maximum_weight: float = 0.40,
) -> pd.DataFrame:
    ledoit_wolf, shrinkage = ledoit_wolf_covariance(returns)
    dcc = dcc_garch_covariance(returns)
    models = {
        "Historical": historical_covariance(returns),
        "EWMA": ewma_covariance(returns),
        "Ledoit-Wolf": ledoit_wolf,
        "DCC-GARCH": dcc.covariance,
    }
    rows = []
    for name, covariance in models.items():
        portfolio = maximum_sharpe(
            expected_returns, covariance, risk_free_rate, maximum_weight
        )
        rows.append(
            {
                "model": name,
                "expected_return": portfolio["return"],
                "volatility": portfolio["volatility"],
                "sharpe": portfolio["sharpe"],
                "average_correlation": np.mean(
                    covariance_to_correlation(covariance)[
                        np.triu_indices_from(covariance, 1)
                    ]
                ),
                "shrinkage": shrinkage if name == "Ledoit-Wolf" else np.nan,
                "dcc_a": dcc.a if name == "DCC-GARCH" else np.nan,
                "dcc_b": dcc.b if name == "DCC-GARCH" else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
