"""Utilities for constructing financial return correlation matrices."""

import pandas as pd


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate percentage returns from price data."""
    return prices.pct_change().dropna(how="all")


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate pairwise Pearson correlations between asset returns."""
    return returns.corr()
