from zenml import step
import mlflow
import pandas as pd
import logging
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
from zenml.client import Client
from model.evaluation import(
    MSE,
    RMSE,
    R2Score,
)

#experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker = 'mlflow_tracker')
def evaluation(model: RegressorMixin, x_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[Annotated[float, 'r2_score'], Annotated[float, 'rmse'], Annotated[float, 'mse']]:

    """
    Zenml step for model evaluation.
    Args:
        model: RegressorMixin
        x_test: pd.DataFrame
        y_test: pd.Series
    Returns:
        r2_score: float
        mse: float
        rmse: float
    """
    try:
        logging.info('start predicting in zenml step')
        y_pred = model.predict(x_test)

        mse_class = MSE()
        mse = mse_class.calculate_score(y_test, y_pred)
        mlflow.log_metric('mse', mse)

        r2_class = R2Score()
        r2_score = r2_class.calculate_score(y_test, y_pred)
        mlflow.log_metric('r2_score', r2_score)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_score(y_test, y_pred)
        mlflow.log_metric('rmse', rmse)
        return mse, rmse, r2_score

    except Exception as e:
       logging.error(f"exception error while evaluating scores with exception: {str(e)}")
       raise e
