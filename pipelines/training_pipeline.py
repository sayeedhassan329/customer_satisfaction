from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml.pipelines import pipeline
from steps.clean_data import clean_data
from steps.ingest_data import ingest_data
from steps.model_train import train_model
from steps.evaluation import evaluation

docker_settings = DockerSettings(required_integration=[MLFLOW])

@pipeline(enable_cache=False, settings={'docker': docker_settings})
def train_pipeline(data_path: str):
    """
    pipeline to train a model
    Args:
        data_path (str): path to the data
    Returns:
        mse: float
        rmse: float
        r2: float
    """
    data = ingest_data(data_path)
    x_train, x_test, y_train, y_test = clean_data(data)
    model = train_model(x_train, x_test, y_train, y_test)
    mse, rmse, r2 = evaluation(model, x_test ,y_test)
