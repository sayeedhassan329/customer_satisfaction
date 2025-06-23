import logging
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

class Evaluation(ABC):
    """
    Abstract base class for evaluation strategy
    """
    @abstractmethod
    def calculate_score(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        pass


class MSE(Evaluation):
    """
    concrete class for calculating Mean Squared Error
    """
    def calculate_score(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """
        Method to calculate the mse
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            mse: float
        """
        try:
            logging.info('start mean squared error calculation with MSE class')
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f'finish mean squared error calculation and value is {str(mse)}')
            return mse
        except Exception as e:
            logging.error(f'Could not calculate MSE with Exception: {str(e)}')
            raise e


class RMSE(Evaluation):
    """
    Concrete class for calculating RMSE
    """

    def calculate_score(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """
        Method to calculate the RMSE
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            RMSE: float
        """
        try:
            logging.info('start Root mean squared error calculation with RMSE class ')
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            logging.info(f'finish Root Mean Squared Error calculation and value is {str(rmse)}')
            return rmse
        except Exception as e:
            logging.error(f'Could not calculate RMSE with Exception: {str(e)}')
            raise e


class R2Score(Evaluation):
    """
    Concrete class for calculating r2 score
    """

    def calculate_score(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """
        Method to calculate the R2 score
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            R2 score: float
        """
        try:
            logging.info('start R2 score calculation with R2Score class ')
            r2 = r2_score(y_true, y_pred)
            logging.info(f'finish R2 score calculation and value is {str(r2)}')
            return r2
        except Exception as e:
            logging.error(f'Could not calculate r2_score with Exception: {str(e)}')
            raise e
