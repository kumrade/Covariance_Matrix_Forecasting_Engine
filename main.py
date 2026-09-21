"""Run a reproducible demonstration of the covariance-risk research pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.covariance_engine import compare_models, log_returns, stress_correlations


def demonstration_prices(days: int = 600, assets: int = 8, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0003, 0.011, days)
    returns = np.column_stack(
        [rng.uniform(0.7, 1.3) * market + rng.normal(0.0002, 0.010, days) for _ in range(assets)]
    )
    prices = 100.0 * np.exp(np.cumsum(returns, axis=0))
    index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    columns = [f"Asset_{number + 1}" for number in range(assets)]
    return pd.DataFrame(prices, index=index, columns=columns)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, help="Optional CSV with a date column and price columns")
    parser.add_argument("--output", type=Path, default=Path("outputs/model_comparison.csv"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.prices:
        prices = pd.read_csv(arguments.prices, index_col=0, parse_dates=True)
    else:
        prices = demonstration_prices(seed=arguments.seed)

    returns = log_returns(prices)
    expected_returns = returns.mean().to_numpy() * 252
    comparison = compare_models(returns, expected_returns)
    stress = stress_correlations(returns)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(arguments.output, index=False)
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(
        "\nStress correlations: "
        f"signed {stress['average_normal']:.3f} -> {stress['average_stress']:.3f}; "
        f"absolute {stress['average_absolute_normal']:.3f} -> "
        f"{stress['average_absolute_stress']:.3f}"
    )
    print(f"\nSaved {arguments.output}")


if __name__ == "__main__":
    main()
