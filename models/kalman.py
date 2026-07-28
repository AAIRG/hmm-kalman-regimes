import numpy as np
import pandas as pd
from typing import Tuple, Any
import logging
import joblib
from pykalman import KalmanFilter
from utils.evaluation import calculate_aic_bic

logger = logging.getLogger(__name__)

class KalmanFilterModel:
    """
    A class representing a Kalman Filter for market state estimation.
    Wraps pykalman.KalmanFilter.
    """

    def __init__(
        self, 
        initial_state_mean: Any = None, 
        initial_state_covariance: Any = None,
        transition_matrices: Any = None,
        observation_matrices: Any = None,
        transition_covariance: Any = None,
        observation_covariance: Any = None
    ) -> None:
        self.initial_state_mean = np.array(initial_state_mean) if initial_state_mean is not None else np.zeros(1)
        self.initial_state_covariance = np.array(initial_state_covariance) if initial_state_covariance is not None else np.ones((1, 1))
        
        self.model = KalmanFilter(
            initial_state_mean=self.initial_state_mean,
            initial_state_covariance=self.initial_state_covariance,
            transition_matrices=np.array(transition_matrices) if transition_matrices is not None else np.ones((1, 1)),
            observation_matrices=np.array(observation_matrices) if observation_matrices is not None else np.ones((1, 1)),
            transition_covariance=np.array(transition_covariance) if transition_covariance is not None else np.ones((1, 1)) * 0.01,
            observation_covariance=np.array(observation_covariance) if observation_covariance is not None else np.ones((1, 1))
        )
        self.is_fitted = False
        self.log_likelihood = None
        self.aic = None
        self.bic = None

    def fit(self, data: pd.Series) -> None:
        X = data.values
        logger.info("Fitting Kalman Filter parameters via EM...")
        self.model = self.model.em(X, n_iter=20)
        self.is_fitted = True
        
        self.log_likelihood = self.model.loglikelihood(X).sum()
        
        n_params = 6
        self.aic, self.bic = calculate_aic_bic(self.log_likelihood, n_params, len(data))
        logger.info(f"Kalman Filter fit complete. Log-Likelihood: {self.log_likelihood:.2f}, AIC: {self.aic:.2f}, BIC: {self.bic:.2f}")

    def filter(self, data: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        X = data.values
        filtered_state_means, filtered_state_covariances = self.model.filter(X)
        return filtered_state_means, filtered_state_covariances

    def predict(self, steps: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Prediction logic for future steps requires manual looping in pykalman.")

    def random_search_tuning(self, data: pd.Series, param_grid: dict, n_iter: int = 10) -> dict:
        logger.info(f"Starting random search tuning for Kalman Filter over {n_iter} iterations...")
        X = data.values
        best_ll = -np.inf
        best_params = {}
        
        for i in range(n_iter):
            params = {}
            for key, values in param_grid.items():
                if isinstance(values, list) and len(values) > 0:
                    idx = np.random.randint(0, len(values))
                    params[key] = values[idx]
                else:
                    params[key] = values
            
            temp_model = KalmanFilter(
                initial_state_mean=np.array(params.get('initial_state_mean', [0.0])),
                initial_state_covariance=np.array(params.get('initial_state_covariance', [[1.0]])),
                transition_covariance=np.array(params.get('transition_covariance', [[0.01]])),
                observation_covariance=np.array(params.get('observation_covariance', [[1.0]]))
            )
            
            try:
                ll = temp_model.loglikelihood(X).sum()
                if ll > best_ll:
                    best_ll = ll
                    best_params = params
            except Exception as e:
                logger.debug(f"Kalman Tuning iteration failed: {e}")
                
        logger.info(f"Found best Kalman Filter parameters (LL={best_ll:.2f}): {best_params}")
        if not best_params:
            best_params = {k: v[0] if isinstance(v, list) else v for k, v in param_grid.items()}
        return best_params

    def update_online(self, new_data: float, current_mean: np.ndarray, current_cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
             raise ValueError("Model must be fitted/initialized first.")
        new_mean, new_cov = self.model.filter_update(
            filtered_state_mean=current_mean,
            filtered_state_covariance=current_cov,
            observation=new_data
        )
        return new_mean, new_cov

    def save(self, filepath: str) -> None:
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted model.")
        joblib.dump(self.model, filepath)
        logger.info(f"Kalman Filter saved to {filepath}")

    def load(self, filepath: str) -> None:
        self.model = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Kalman Filter loaded from {filepath}")
