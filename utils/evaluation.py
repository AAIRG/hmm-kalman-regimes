import numpy as np
import pandas as pd
from typing import Tuple

def calculate_aic_bic(log_likelihood: float, n_params: int, n_samples: int) -> Tuple[float, float]:
    """
    Calculates the Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC).

    Args:
        log_likelihood: The log-likelihood of the model.
        n_params: The number of estimated parameters.
        n_samples: The number of observations.

    Returns:
        A tuple containing (AIC, BIC).
    """
    aic = 2 * n_params - 2 * log_likelihood
    bic = n_params * np.log(n_samples) - 2 * log_likelihood
    return aic, bic

def backtest_regime_strategy(returns: pd.Series, regimes: np.ndarray, regime_positions: dict, transaction_cost: float = 0.0) -> pd.DataFrame:
    """
    Simulates a regime-based long/short trading strategy.

    Args:
        returns: A pandas Series of log returns.
        regimes: An array of predicted regimes (same length as returns).
        regime_positions: A dictionary mapping regime index to target position
                          (e.g., 1.0 for Long, -1.0 for Short, 0 for Cash; supports leverage).
        transaction_cost: Proportional cost per unit of position change (e.g., 0.001 = 10 bps).

    Returns:
        A pandas DataFrame comparing Buy & Hold vs Regime Strategy.
    """
    df = pd.DataFrame({'Log_Return': returns, 'Regime': regimes}, index=returns.index)

    # Map each regime to its corresponding position (defaulting to 0/cash if not found)
    df['Position'] = df['Regime'].map(regime_positions).fillna(0)

    # Shift position by 1 day to avoid lookahead bias (you trade AT the close based on the signal)
    df['Position'] = df['Position'].shift(1).fillna(0)

    # Transaction cost on position changes (e.g., long→short = 2 units of turnover)
    df['Trade'] = df['Position'].diff().abs().fillna(0)
    df['TC'] = df['Trade'] * transaction_cost

    # Accurate leveraged return: convert log→simple, apply position, convert back
    simple_return = np.exp(df['Log_Return']) - 1
    leveraged_simple_return = df['Position'] * simple_return - df['TC']
    # Clip to prevent log of non-positive numbers (total wipeout with extreme leverage)
    leveraged_simple_return = leveraged_simple_return.clip(lower=-0.9999)
    df['Strategy_Return'] = np.log1p(leveraged_simple_return)

    df['Buy_Hold_Equity'] = np.exp(df['Log_Return'].cumsum())
    df['Strategy_Equity'] = np.exp(df['Strategy_Return'].cumsum())

    return df

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculates the annualized Sharpe Ratio based on daily log returns.
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    excess_returns = returns - (risk_free_rate / 252)
    return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculates the Maximum Drawdown of an equity curve.
    """
    if len(equity_curve) == 0:
        return 0.0
    rolling_max = equity_curve.cummax()
    drawdowns = (equity_curve - rolling_max) / rolling_max
    return drawdowns.min()

def calculate_annualized_return(equity_curve: pd.Series) -> float:
    """
    Calculates the Annualized Return from a cumulative equity curve.
    Assumes daily data (252 trading days per year).
    """
    if len(equity_curve) < 2:
        return 0.0
    
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    years = len(equity_curve) / 252.0
    
    if total_return <= 0:
        return -1.0 # Handle extreme edge cases
    
    return (total_return ** (1 / years)) - 1

def apply_regime_filter(regimes: np.ndarray, min_days: int = 1) -> np.ndarray:
    """
    Confirmation filter: only switch regime after the model predicts the new
    regime for at least `min_days` consecutive observations. Prevents whipsaw
    trading from rapid regime flickering.

    Args:
        regimes: Array of predicted regime labels.
        min_days: Number of consecutive days a new regime must be predicted
                  before the switch is confirmed. 1 = no filtering.

    Returns:
        Filtered regime array with the same shape.
    """
    if min_days <= 1:
        return regimes.copy()

    filtered = regimes.copy()
    current = filtered[0]
    candidate = None
    streak = 0

    for i in range(1, len(filtered)):
        if regimes[i] != current:
            if regimes[i] == candidate:
                streak += 1
            else:
                candidate = regimes[i]
                streak = 1

            if streak >= min_days:
                current = candidate
                candidate = None
                streak = 0

            filtered[i] = current
        else:
            candidate = None
            streak = 0
            filtered[i] = current

    return filtered


def calculate_regime_statistics(returns: pd.Series, regimes: np.ndarray) -> pd.DataFrame:
    """
    Calculates the mean annualized return and annualized volatility for each distinct regime.
    Helps evaluate Distribution Separation.
    """
    df = pd.DataFrame({'Return': returns, 'Regime': regimes})
    stats = []
    
    for regime in sorted(df['Regime'].unique()):
        regime_data = df[df['Regime'] == regime]['Return']
        if len(regime_data) > 0:
            ann_ret = regime_data.mean() * 252
            ann_vol = regime_data.std() * np.sqrt(252)
            stats.append({
                'Regime': int(regime),
                'Count': len(regime_data),
                'Ann. Mean Return': ann_ret,
                'Ann. Volatility': ann_vol
            })
            
    return pd.DataFrame(stats).set_index('Regime')
