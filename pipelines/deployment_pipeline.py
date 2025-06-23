import json
import os
import numpy as np
import pandas as pd
import logging

from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluation

from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW

from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from pydantic import BaseModel

from .utils import get_data_for_test



docker_settings = DockerSettings(required_integrations=[MLFLOW])

requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")




from zenml import step
from zenml.logger import get_logger
import pandas as pd

logger = get_logger(__name__)

@step(enable_cache=False)
def dynamic_importer() -> pd.DataFrame:
    """
    Downloads the latest data from mock API.
    Returns:
        str: JSON string of the data or empty string if no data
    """
    try:
        data = get_data_for_test()

        if data is None:
            logger.warning("No data received from API")
            return ""

        if isinstance(data, pd.DataFrame) and data.empty:
            logger.warning("Empty DataFrame received")
            return ""

        return data

    except Exception as e:
        logger.error(f"Error in dynamic_importer: {str(e)}")
        raise


class DeploymentTriggerConfig(BaseModel):
    """
    parameters that are used to trigger the deployment
    """
    min_accuracy: float = 0.5


@step
def deployment_trigger(
    config: DeploymentTriggerConfig,
    accuracy: float,
) -> bool:
    """Implements a simple model deployment trigger that looks at the
    input model accuracy and decides if it is good enough to deploy"""

    return float(accuracy) > float(config.min_accuracy)




class MLFlowDeploymentLoaderStepParameters(BaseModel):
    """MLflow deployment getter parameters

    Attributes:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """

    pipeline_name: str
    step_name: str
    running: bool = False
    model_name: str = "model"

@step(enable_cache=False)
def prediction_service_loader(
    params: MLFlowDeploymentLoaderStepParameters
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline.

    Args:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """
    # get the MLflow model deployer stack component
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # fetch existing services with same pipeline name, step name and model name
    existing_services = model_deployer.find_model_server(
        pipeline_name=params.pipeline_name,
        pipeline_step_name=params.step_name,
        model_name=params.model_name,
        running=params.running,
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{params.step_name} step in the {params.pipeline_name} "
            f"pipeline for the '{params.model_name}' model is currently "
            f"running."
        )
    print('Existing_services: ', existing_services)
    print('Type of existing services: ', type(existing_services))
    service = existing_services[0]

       # Ensure the service is running
    if not service.is_running:
        logger.info(f"Starting MLflow service {service.uuid}...")
        service.start(timeout=120)

        if not service.is_running:
            raise RuntimeError(
                f"Failed to start MLflow prediction service {service.uuid}. "
                f"Current state: {service.status.state.value}"
            )

    logger.info(f"MLflow service is running at: {service.prediction_url}")
    return service



@step
def predictor(
    service: MLFlowDeploymentService,
    data: pd.DataFrame,
) -> np.ndarray:
    """Run an inference request against a prediction service"""

    if not service.is_running:
            raise ValueError(
                f"Service not running. Current status: {service.status}\n"
                f"Prediction URL: {service.prediction_url}\n"
                f"Service info: {service.get_service_info()}"
            )

    prediction = service.predict(data)
    return prediction


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    data_path: str,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    # Link all the steps artifacts together
    df = ingest_data(data_path=data_path)
    x_train, x_test, y_train, y_test = clean_data(data=df)  # Explicitly name the parameter
    model = train_model(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
    )
    mse, rmse , r2 = evaluation(model=model, x_test=x_test, y_test=y_test)
    deployment_decision = deployment_trigger(
        config=DeploymentTriggerConfig(),
        accuracy=r2
    )
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
    )

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    # Link all the steps artifacts together
    batch_data = dynamic_importer()
    loader_params = MLFlowDeploymentLoaderStepParameters(
            pipeline_name=pipeline_name,
            step_name=pipeline_step_name,  # Matches class definition
            running=False,)

    model_deployment_service = prediction_service_loader(loader_params)


    predictor(service=model_deployment_service, data=batch_data)
