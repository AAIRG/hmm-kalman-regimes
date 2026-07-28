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

def backtest_regime_strategy(returns: pd.Series, regimes: np.ndarray, regime_positions: dict) -> pd.DataFrame:
    """
    Simulates a regime-based long/short trading strategy.
    
    Args:
        returns: A pandas Series of log returns.
        regimes: An array of predicted regimes (same length as returns).
        regime_positions: A dictionary mapping regime index to target position 
                          (1 for Long, -1 for Short, 0 for Cash).

    Returns:
        A pandas DataFrame comparing Buy & Hold vs Regime Strategy.
    """
    df = pd.DataFrame({'Log_Return': returns, 'Regime': regimes}, index=returns.index)
    
    # Map each regime to its corresponding position (defaulting to 0/cash if not found)
    df['Position'] = df['Regime'].map(regime_positions).fillna(0)
    
    # Shift position by 1 day to avoid lookahead bias (you trade AT the close based on the signal)
    df['Position'] = df['Position'].shift(1).fillna(0)
    
    df['Strategy_Return'] = df['Position'] * df['Log_Return']
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
