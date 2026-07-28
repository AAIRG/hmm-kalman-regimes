import numpy as np
import pandas as pd
import logging
import joblib
from hmmlearn.hmm import GaussianHMM
from utils.evaluation import calculate_aic_bic

logger = logging.getLogger(__name__)

class HiddenMarkovModel:
    """
    A class representing a Hidden Markov Model (HMM) for market regime detection.
    Wraps hmmlearn.hmm.GaussianHMM.
    """

    def __init__(self, n_components: int = 2, covariance_type: str = "full", n_iter: int = 100) -> None:
        """
        Initializes the Hidden Markov Model.

        Args:
            n_components: The number of hidden states (regimes).
            covariance_type: String describing the type of covariance parameters to use.
            n_iter: Maximum number of iterations to perform.
        """
        self.n_components = int(n_components)
        self.covariance_type = covariance_type
        self.n_iter = int(n_iter)
        
        self.model = GaussianHMM(
            n_components=self.n_components, 
            covariance_type=self.covariance_type, 
            n_iter=self.n_iter, 
            random_state=42
        )
        self.is_fitted = False
        self.log_likelihood = None
        self.history = []
        self.aic = None
        self.bic = None

    def fit(self, data: pd.Series) -> None:
        """
        Fits the HMM to the historical data.

        Args:
            data: A pandas Series of observations (e.g., log returns).
        """
        X = data.values.reshape(-1, 1)
        
        logger.info(f"Fitting HMM with {self.n_components} components...")
        self.model.fit(X)
        self.is_fitted = True
        self.log_likelihood = self.model.score(X)
        
        n_params = self.n_components - 1 + self.n_components * (self.n_components - 1) + 2 * self.n_components
        self.aic, self.bic = calculate_aic_bic(self.log_likelihood, n_params, len(data))
        
        if hasattr(self.model, "monitor_"):
            self.history = self.model.monitor_.history
            
        logger.info(f"HMM fit complete. Log-Likelihood: {self.log_likelihood:.2f}, AIC: {self.aic:.2f}, BIC: {self.bic:.2f}")

    def predict_regimes(self, data: pd.Series) -> np.ndarray:
        """
        Predicts the most likely sequence of hidden states (regimes).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting regimes.")
        X = data.values.reshape(-1, 1)
        return self.model.predict(X)

    def get_state_probabilities(self, data: pd.Series) -> np.ndarray:
        """
        Calculates the probability of being in each regime at each time step.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting probabilities.")
        X = data.values.reshape(-1, 1)
        return self.model.predict_proba(X)

    def random_search_tuning(self, data: pd.Series, param_grid: dict, n_iter: int = 10) -> dict:
        """
        Performs random search for optimal hyperparameters based on BIC.
        """
        logger.info(f"Starting random search tuning for HMM over {n_iter} iterations...")
        X = data.values.reshape(-1, 1)
        best_bic = np.inf
        best_params = {}
        
        for i in range(n_iter):
            params = {}
            for key, values in param_grid.items():
                if isinstance(values, list) and len(values) > 0:
                    idx = np.random.randint(0, len(values))
                    params[key] = values[idx]
                else:
                    params[key] = values
            
            temp_model = GaussianHMM(
                n_components=int(params.get('n_components', 2)),
                covariance_type=params.get('covariance_type', 'full'),
                n_iter=int(params.get('n_iter', 100)),
                random_state=i
            )
            
            try:
                temp_model.fit(X)
                ll = temp_model.score(X)
                n_c = int(params.get('n_components', 2))
                n_p = n_c - 1 + n_c * (n_c - 1) + 2 * n_c
                _, bic = calculate_aic_bic(ll, n_p, len(data))
                
                if bic < best_bic:
                    best_bic = bic
                    best_params = params
                    best_params['n_components'] = int(best_params['n_components'])
                    best_params['n_iter'] = int(best_params['n_iter'])
            except Exception as e:
                logger.debug(f"Tuning iteration failed: {e}")
                
        logger.info(f"Found best HMM parameters (BIC={best_bic:.2f}): {best_params}")
        if not best_params:
            best_params = {k: v[0] if isinstance(v, list) else v for k, v in param_grid.items()}
        return best_params

    def update_online(self, new_data: float) -> None:
        raise NotImplementedError("Online update is not straightforward for batch EM hmmlearn. Requires custom implementation.")

    def save(self, filepath: str) -> None:
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted model.")
        joblib.dump(self.model, filepath)
        logger.info(f"HMM saved to {filepath}")

    def load(self, filepath: str) -> None:
        self.model = joblib.load(filepath)
        self.is_fitted = True
        self.n_components = self.model.n_components
        logger.info(f"HMM loaded from {filepath}")
