import logging
import mlflow
import pandas as pd
from model.model_dev import(
    HyperparameterTuner,
    RandomForestModel,
)

from zenml import step
from zenml.client import Client


experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker = experiment_tracker.name)
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
):
    """
    Zenml train_model step:
    Args:
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    """
    try:
        logging.info('start model training')
        mlflow.sklearn.autolog()
        model = RandomForestModel
        tuner = HyperparameterTuner(model, x_train, y_train, x_test, y_test)
        trained_model = tuner.train()
        logging.info('finish model training')
        return trained_model
    except Exception as e:
        logging.error(f'could not train model with error: {str(e)}')
        raise e
