import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path
import logging
import plotly.graph_objects as go

from data.csv_loader import CSVLoader
from data.api_loader import YFinanceLoader
from utils.logger import setup_logging
from utils.config import load_config, save_config
from utils.evaluation import (
    backtest_regime_strategy,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_annualized_return,
    calculate_regime_statistics,
    apply_regime_filter,
    build_regime_positions
)
from models.hmm import HiddenMarkovModel
from models.kalman import KalmanFilterModel

logger = logging.getLogger(__name__)

def plot_price_with_regimes(data: pd.DataFrame, regimes: np.ndarray, model_name: str, split_date=None):
    fig = go.Figure()
    
    price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
    if price_col not in data.columns:
        return fig
        
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data[price_col],
        mode='lines',
        name='Price',
        line=dict(color='black')
    ))
    
    colors = ['rgba(255, 0, 0, 0.2)', 'rgba(0, 255, 0, 0.2)', 'rgba(0, 0, 255, 0.2)', 'rgba(255, 165, 0, 0.2)']
    max_price = data[price_col].max() * 1.05
    min_price = data[price_col].min() * 0.95
    
    for i in np.unique(regimes):
        if pd.isna(i): continue
        color = colors[int(i) % len(colors)]
        
        # Add baseline trace for filling
        fig.add_trace(go.Scatter(
            x=data.index, y=[min_price]*len(data), 
            mode='lines', line=dict(width=0), 
            showlegend=False, hoverinfo='skip'
        ))
        
        # Add regime trace
        y_vals = np.where(regimes == i, max_price, min_price)
        fig.add_trace(go.Scatter(
            x=data.index, y=y_vals,
            mode='lines', line_shape='hv',
            fill='tonexty', fillcolor=color,
            line=dict(width=0),
            showlegend=False, hoverinfo='skip'
        ))
        
    if split_date is not None:
        fig.add_vline(x=split_date, line_width=2, line_dash="dash", line_color="black")
        fig.add_annotation(
            x=split_date, y=max_price*0.95, text="Test (OOS) ➔", 
            showarrow=False, xanchor="left", xshift=5,
            font=dict(color="black", size=12), bgcolor="white", opacity=0.8
        )
        
    fig.update_layout(title=f"Price History colored by {model_name} Regimes", template="plotly_white")
    return fig

def plot_convergence(history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=list(history), mode='lines+markers', name='Log-Likelihood'))
    fig.update_layout(title="EM Algorithm Convergence", xaxis_title="Iteration", yaxis_title="Log-Likelihood")
    return fig

def plot_backtest(backtest_df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Buy_Hold_Equity'], name='Buy & Hold'))
    fig.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Strategy_Equity'], name='Regime Strategy'))
    fig.update_layout(title="Strategy Backtest Equity Curve", yaxis_title="Cumulative Return")
    return fig

def main() -> None:
    """
    Main function to construct and run the Streamlit dashboard.
    """
    setup_logging()
    logger.info("Starting Streamlit Dashboard...")

    st.set_page_config(
        page_title="Market Regime Detection",
        page_icon="📈",
        layout="wide"
    )

    st.title("Market Regime Detection Dashboard")
    st.markdown("""
    This dashboard analyzes historical market data to detect hidden regimes (e.g., Bull, Bear, Sideways) 
    using **Hidden Markov Models (HMM)** and **Kalman Filters**.
    """)

    app_config = load_config("app_config.yaml")
    ui_cfg = app_config.get("app", {})
    model_cfg = app_config.get("models", {})

    st.sidebar.header("Data Configuration")
    data_source: str = st.sidebar.radio(
        "Data Source",
        ["Local CSV (Backtest)", "Yahoo Finance API"]
    )
    
    symbol_or_path: str = ""
    if data_source == "Yahoo Finance API":
        popular_tickers = {
            "S&P 500 (SPY)": "SPY",
            "Gold (GLD)": "GLD",
            "Silver (SLV)": "SLV",
            "Bitcoin (BTC-USD)": "BTC-USD",
            "Ethereum (ETH-USD)": "ETH-USD",
            "EUR/USD": "EURUSD=X",
            "GBP/USD": "GBPUSD=X",
            "Apple (AAPL)": "AAPL",
            "Tesla (TSLA)": "TSLA"
        }
        selected_name = st.sidebar.selectbox("Select Asset", list(popular_tickers.keys()))
        symbol_or_path = popular_tickers[selected_name]
    else:
        symbol_or_path = st.sidebar.text_input("CSV File Path", value=ui_cfg.get("csv_file_path", "data/sample_data.csv"))
    
    start_date: date = st.sidebar.date_input("Start Date", value=pd.to_datetime(ui_cfg.get("default_start_date", "2020-01-01")))
    end_date: date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))
    
    st.sidebar.header("Model Selection")
    model_choice: str = st.sidebar.selectbox(
        "Choose a State-Space Model",
        ["Hidden Markov Model (HMM)", "Kalman Filter", "Combined Model (Kalman + HMM)"]
    )

    st.sidebar.header("Strategy Settings")
    trading_direction = st.sidebar.radio(
        "Trading Direction",
        ["Long & Short", "Long Only", "Short Only"]
    )
    leverage = st.sidebar.slider("Leverage", min_value=1.0, max_value=5.0, value=1.0, step=0.5)
    transaction_cost_bps = st.sidebar.slider("Transaction Cost (bps)", min_value=0, max_value=50, value=10, step=1)
    transaction_cost = transaction_cost_bps / 10000.0
    min_regime_days = st.sidebar.slider("Min Regime Confirmation (days)", min_value=1, max_value=10, value=1, step=1)
    st.sidebar.caption("Regime switch requires N consecutive days of the new signal.")

    st.sidebar.header("Regime Quality Filters")
    use_regime_filters = st.sidebar.checkbox("Skip Low-Quality Regimes (stay in Cash)", value=False)
    min_abs_ann_return = 0.0
    max_ann_volatility = float(np.inf)
    min_abs_sharpe = 0.0
    if use_regime_filters:
        min_abs_ann_return = st.sidebar.slider(
            "Min |Ann. Return| to trade (%)", min_value=0.0, max_value=30.0, value=5.0, step=1.0
        ) / 100.0
        max_ann_volatility = st.sidebar.slider(
            "Max Ann. Volatility to trade (%)", min_value=5.0, max_value=150.0, value=60.0, step=5.0
        ) / 100.0
        min_abs_sharpe = st.sidebar.slider(
            "Min |Sharpe| to trade", min_value=0.0, max_value=2.0, value=0.3, step=0.1
        )
        st.sidebar.caption("Regimes failing any test are held in Cash instead of forced Long/Short.")

    st.sidebar.header("Hyperparameter Tuning")
    run_tuning = st.sidebar.checkbox("Run Random Tuning Before Analysis", value=False)
    n_iter_tuning = 0
    if run_tuning:
        n_iter_tuning = st.sidebar.slider("Random Search Iterations", min_value=1, max_value=50, value=10)
        
    st.sidebar.header("Evaluation Settings")
    train_size_pct = st.sidebar.slider("Train/Test Split (%)", min_value=50, max_value=100, value=80, step=5)
    st.sidebar.caption("Set to 100% for In-Sample analysis.")

    if st.sidebar.button("Run Analysis"):
        logger.info(f"User requested analysis with {model_choice} using data from {symbol_or_path}")
        st.info(f"Loading data from **{symbol_or_path}** ({start_date} to {end_date})...")
        
        try:
            if data_source == "Local CSV (Backtest)":
                loader = CSVLoader()
                raw_data = loader.fetch_data(symbol_or_path, str(start_date), str(end_date))
            else:
                loader = YFinanceLoader()
                raw_data = loader.fetch_data(symbol_or_path, str(start_date), str(end_date))
            
            if not raw_data.empty:
                clean_data = loader.preprocess_data(raw_data)
                st.success(f"Successfully loaded and preprocessed {len(clean_data)} records.")
            else:
                st.error("No data loaded. Please check the data source and date range.")
                clean_data = None
                
        except Exception as e:
            st.error(f"Error loading data: {e}")
            clean_data = None

        if clean_data is not None:
            split_idx = int(len(clean_data) * (train_size_pct / 100.0))
            train_data = clean_data.iloc[:split_idx]
            test_data = clean_data.iloc[split_idx:]
            eval_data = test_data if train_size_pct < 100 else clean_data
            split_date = clean_data.index[split_idx] if train_size_pct < 100 else None
            
            st.divider()
            st.subheader(f"Analysis & Backtest Results: {model_choice}")
            
            col1, col2 = st.columns(2)
            
            if model_choice == "Hidden Markov Model (HMM)":
                if run_tuning:
                    with st.spinner(f"Running Random Hyperparameter Tuning for HMM ({n_iter_tuning} iterations)..."):
                        param_grid = {
                            "n_components": [3, 4, 5],
                            "covariance_type": ["full", "diag"],
                            "n_iter": [50, 100, 200]
                        }
                        temp_model = HiddenMarkovModel()
                        best_params = temp_model.random_search_tuning(train_data['Log_Return'], param_grid, n_iter_tuning)
                        
                        if "hmm" not in model_cfg:
                            model_cfg["hmm"] = {}
                        model_cfg["hmm"].update(best_params)
                        app_config["models"] = model_cfg
                        # save_config(app_config, "app_config.yaml") # Disabled to prevent Streamlit rerun
                        st.success(f"Tuning complete! Selected parameters: {best_params}")
                        st.info("Parameters applied to current session (not saved to disk).")

                hmm_params = model_cfg.get("hmm", {})
                n_comp = hmm_params.get("n_components", 2)
                
                with st.spinner(f"Training HMM with {n_comp} regimes..."):
                    model = HiddenMarkovModel(
                        n_components=n_comp,
                        covariance_type=hmm_params.get("covariance_type", "full"),
                        n_iter=hmm_params.get("n_iter", 100)
                    )
                    model.fit(train_data['Log_Return'])
                    # Predict train/test separately to avoid Viterbi look-ahead bias
                    train_regimes = model.predict_regimes(train_data['Log_Return'])
                    if train_size_pct < 100:
                        test_regimes = model.predict_regimes(test_data['Log_Return'])
                        regimes = np.concatenate([train_regimes, test_regimes])
                    else:
                        regimes = train_regimes
                    # Apply regime confirmation filter to prevent whipsaw
                    regimes = apply_regime_filter(regimes, min_regime_days)
                    eval_regimes = regimes[split_idx:] if train_size_pct < 100 else regimes
                    
                    Path("saved_models").mkdir(exist_ok=True)
                    model.save("saved_models/hmm_latest.joblib")
                
                with col1:
                    st.metric("Log-Likelihood", f"{model.log_likelihood:.2f}")
                    st.metric("AIC", f"{model.aic:.2f}")
                with col2:
                    st.metric("BIC", f"{model.bic:.2f}")
                
                st.plotly_chart(plot_price_with_regimes(raw_data.loc[clean_data.index], regimes, "HMM", split_date=split_date), use_container_width=True)
                
                if model.history:
                    st.plotly_chart(plot_convergence(model.history), use_container_width=True)
                
                means = [
                    train_data['Log_Return'][regimes[:split_idx] == i].mean() 
                    if np.sum(regimes[:split_idx] == i) > 0 else -np.inf 
                    for i in range(n_comp)
                ]
                
                regime_positions = {}
                for i, m in enumerate(means):
                    if m > 0:
                        regime_positions[i] = (1 if trading_direction in ["Long & Short", "Long Only"] else 0) * leverage
                    else:
                        regime_positions[i] = (-1 if trading_direction in ["Long & Short", "Short Only"] else 0) * leverage
                
                st.write(f"*(Evaluating on {'**In-Sample**' if train_size_pct == 100 else '**Out-of-Sample Test**'} Data)*")
                bt_df = backtest_regime_strategy(eval_data['Log_Return'], eval_regimes, regime_positions, transaction_cost)
                st.plotly_chart(plot_backtest(bt_df), use_container_width=True)

                st.write("### Backtest Performance")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Sharpe (B&H)", f"{calculate_sharpe_ratio(bt_df['Log_Return']):.2f}")
                m2.metric("Sharpe (Strategy)", f"{calculate_sharpe_ratio(bt_df['Strategy_Return']):.2f}")
                m3.metric("Max Drawdown", f"{calculate_max_drawdown(bt_df['Strategy_Equity']):.2%}")
                m4.metric("Ann. Return", f"{calculate_annualized_return(bt_df['Strategy_Equity']):.2%}")

                st.write("### Regime Distribution Separation")
                regime_stats = calculate_regime_statistics(eval_data['Log_Return'], eval_regimes)
                st.dataframe(regime_stats.style.format({
                    'Ann. Mean Return': '{:.2%}',
                    'Ann. Volatility': '{:.2%}'
                }), use_container_width=True)
                
            elif model_choice == "Combined Model (Kalman + HMM)":
                st.info("Combined Model Strategy: First we apply the causal Kalman Filter to smooth the raw log returns, effectively removing market microstructure noise. Then, we feed these smoothed returns into the Hidden Markov Model (HMM) to detect robust regimes.")
                
                hmm_params = model_cfg.get("hmm", {})
                kf_params = model_cfg.get("kalman", {})
                n_comp = hmm_params.get("n_components", 2)
                
                with st.spinner("Step 1: Filtering noise with Kalman Filter..."):
                    kalman = KalmanFilterModel(
                        initial_state_mean=kf_params.get("initial_state_mean"),
                        initial_state_covariance=kf_params.get("initial_state_covariance"),
                        transition_covariance=kf_params.get("transition_covariance"),
                        observation_covariance=kf_params.get("observation_covariance")
                    )
                    kalman.fit(train_data['Log_Return'])
                    filtered_means, _ = kalman.filter(clean_data['Log_Return'])
                    smoothed_returns = pd.Series(filtered_means.flatten(), index=clean_data.index)
                    
                with st.spinner(f"Step 2: Training HMM ({n_comp} regimes) on smoothed returns..."):
                    hmm = HiddenMarkovModel(
                        n_components=n_comp,
                        covariance_type=hmm_params.get("covariance_type", "full"),
                        n_iter=hmm_params.get("n_iter", 100)
                    )
                    hmm.fit(smoothed_returns.iloc[:split_idx])
                    # Predict train/test separately to avoid Viterbi look-ahead bias
                    train_regimes = hmm.predict_regimes(smoothed_returns.iloc[:split_idx])
                    if train_size_pct < 100:
                        test_regimes = hmm.predict_regimes(smoothed_returns.iloc[split_idx:])
                        regimes = np.concatenate([train_regimes, test_regimes])
                    else:
                        regimes = train_regimes
                    # Apply regime confirmation filter to prevent whipsaw
                    regimes = apply_regime_filter(regimes, min_regime_days)
                    eval_regimes = regimes[split_idx:] if train_size_pct < 100 else regimes
                
                st.plotly_chart(plot_price_with_regimes(raw_data.loc[clean_data.index], regimes, "Combined Model", split_date=split_date), use_container_width=True)
                
                means = [
                    smoothed_returns.iloc[:split_idx][regimes[:split_idx] == i].mean() 
                    if np.sum(regimes[:split_idx] == i) > 0 else -np.inf 
                    for i in range(n_comp)
                ]
                
                regime_positions = {}
                for i, m in enumerate(means):
                    if m > 0:
                        regime_positions[i] = (1 if trading_direction in ["Long & Short", "Long Only"] else 0) * leverage
                    else:
                        regime_positions[i] = (-1 if trading_direction in ["Long & Short", "Short Only"] else 0) * leverage
                
                st.write(f"*(Evaluating on {'**In-Sample**' if train_size_pct == 100 else '**Out-of-Sample Test**'} Data)*")
                bt_df = backtest_regime_strategy(eval_data['Log_Return'], eval_regimes, regime_positions, transaction_cost)
                st.plotly_chart(plot_backtest(bt_df), use_container_width=True)

                st.write("### Backtest Performance")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Sharpe (B&H)", f"{calculate_sharpe_ratio(bt_df['Log_Return']):.2f}")
                m2.metric("Sharpe (Strategy)", f"{calculate_sharpe_ratio(bt_df['Strategy_Return']):.2f}")
                m3.metric("Max Drawdown", f"{calculate_max_drawdown(bt_df['Strategy_Equity']):.2%}")
                m4.metric("Ann. Return", f"{calculate_annualized_return(bt_df['Strategy_Equity']):.2%}")

                st.write("### Regime Distribution Separation (Evaluated on Raw Returns)")
                regime_stats = calculate_regime_statistics(eval_data['Log_Return'], eval_regimes)
                st.dataframe(regime_stats.style.format({
                    'Ann. Mean Return': '{:.2%}',
                    'Ann. Volatility': '{:.2%}'
                }), use_container_width=True)
                
            elif model_choice == "Kalman Filter":
                if run_tuning:
                    with st.spinner(f"Running Random Hyperparameter Tuning for Kalman Filter ({n_iter_tuning} iterations)..."):
                        param_grid = {
                            "initial_state_mean": [[0.0], [0.1], [-0.1]],
                            "initial_state_covariance": [[1.0], [2.0], [5.0]],
                            "transition_covariance": [[0.01], [0.05], [0.1]],
                            "observation_covariance": [[0.5], [1.0], [2.0]]
                        }
                        temp_model = KalmanFilterModel()
                        best_params = temp_model.random_search_tuning(train_data['Log_Return'], param_grid, n_iter_tuning)
                        
                        if "kalman" not in model_cfg:
                            model_cfg["kalman"] = {}
                        model_cfg["kalman"].update(best_params)
                        app_config["models"] = model_cfg
                        # save_config(app_config, "app_config.yaml") # Disabled to prevent Streamlit rerun
                        st.success(f"Tuning complete! Selected parameters: {best_params}")
                        st.info("Parameters applied to current session (not saved to disk).")

                kf_params = model_cfg.get("kalman", {})
                with st.spinner("Training Kalman Filter via EM..."):
                    model = KalmanFilterModel(
                        initial_state_mean=kf_params.get("initial_state_mean"),
                        initial_state_covariance=kf_params.get("initial_state_covariance"),
                        transition_covariance=kf_params.get("transition_covariance"),
                        observation_covariance=kf_params.get("observation_covariance")
                    )
                    model.fit(train_data['Log_Return'])
                    filtered_means, filtered_covs = model.filter(clean_data['Log_Return'])
                    
                    Path("saved_models").mkdir(exist_ok=True)
                    model.save("saved_models/kalman_latest.joblib")
                
                with col1:
                    st.metric("Log-Likelihood", f"{model.log_likelihood:.2f}")
                    st.metric("AIC", f"{model.aic:.2f}")
                with col2:
                    st.metric("BIC", f"{model.bic:.2f}")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=clean_data.index, y=clean_data['Log_Return'], mode='lines', name='Actual Return', opacity=0.3))
                fig.add_trace(go.Scatter(x=clean_data.index, y=filtered_means.flatten(), mode='lines', name='Filtered State (Smoothed)', line=dict(color='red')))
                fig.update_layout(title="Kalman Filter: True Returns vs Smoothed State", template="plotly_white")
                st.plotly_chart(fig, width='stretch')
                
                st.info("Note: The Kalman Filter provides a continuous smoothed state rather than discrete regimes. You can create regimes by thresholding the filtered state.")
                
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; margin-top: 50px;'>
            <p style='color: #888;'>Developed by <b>Kasra Saeid</b></p>
            <a href='https://linkedin.com/in/kasra-saeid' target='_blank'>
                <img src='https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white' alt='LinkedIn'>
            </a>
            &nbsp;
            <a href='https://github.com/Kasra-Saeid' target='_blank'>
                <img src='https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white' alt='GitHub'>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
