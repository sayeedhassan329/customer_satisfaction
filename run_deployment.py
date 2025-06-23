from typing import cast

import click
from pipelines.deployment_pipeline import(
    continuous_deployment_pipeline,
    inference_pipeline
)
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService

DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help="Optionally you can choose to only run the deployment "
    "pipeline to train and deploy a model (`deploy`), or to "
    "only run a prediction against the deployed model "
    "(`predict`). By default both will be run "
    "(`deploy_and_predict`).",
)
@click.option(
    "--min-accuracy",
    default=0.5,
    help="Minimum accuracy required to deploy the model",
)
def main(config: str, min_accuracy: float):
    """Run the MLflow example pipeline."""
    # get the MLflow model deployer stack component
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT

    if deploy:
        # Initialize a continuous deployment pipeline run
        continuous_deployment_pipeline(
            data_path = "data/olist_customers_dataset.csv",
            workers=3,
            timeout=60,

        )

    if predict:
        # Initialize an inference pipeline run
        inference_pipeline(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
        )

    print(
        "You can run:\n "
        f"[italic green]    mlflow ui --backend-store-uri '{get_tracking_uri()}"
        "[/italic green]\n ...to inspect your experiment runs within the MLflow"
        " UI.\nYou can find your runs tracked within the "
        "`mlflow_example_pipeline` experiment. There you'll also be able to "
        "compare two or more runs.\n\n"
    )

    # fetch existing services with same pipeline name, step name and model name
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="model",
    )

    if existing_services:
        service = cast(MLFlowDeploymentService, existing_services[0])
        if service.is_running:
            print(
                f"The MLflow prediction server is running locally as a daemon "
                f"process service and accepts inference requests at:\n"
                f"    {service.prediction_url}\n"
                f"To stop the service, run "
                f"[italic green]`zenml model-deployer models delete "
                f"{str(service.uuid)}`[/italic green]."
            )
        elif service.is_failed:
            print(
                f"The MLflow prediction server is in a failed state:\n"
                f" Last state: '{service.status.state.value}'\n"
                f" Last error: '{service.status.last_error}'"
            )
    else:
        print(
            "No MLflow prediction server is currently running. The deployment "
            "pipeline must run first to train a model and deploy it. Execute "
            "the same command with the `--deploy` argument to deploy a model."
        )


if __name__ == "__main__":
    main()



# Run the below code  in terminal after removing # from all lines to check the prediction: its original review_score is 4:
#
# curl -X POST -H "Content-Type:application/json" \
#   --data '{
#     "dataframe_split": {
#       "columns": [
#         "order_status", "order_purchase_timestamp",
#         "order_approved_at", "order_delivered_carrier_date",
#         "order_delivered_customer_date", "order_estimated_delivery_date",
#         "payment_sequential", "payment_type", "payment_installments",
#         "payment_value", "customer_unique_id", "customer_zip_code_prefix",
#         "customer_city", "customer_state", "order_item_id",
#         "seller_id", "shipping_limit_date", "price", "freight_value",
#         "product_category_name", "product_name_lenght",
#         "product_description_lenght", "product_photos_qty", "product_weight_g",
#         "product_length_cm", "product_height_cm", "product_width_cm",
#         "product_category_name_english"
#       ],
#       "data": [
#         [

#           "delivered", "2017-10-02 10:56:33", "2017-10-02 11:07:15",
#           "2017-10-04 19:55:00", "2017-10-10 21:25:13", "2017-10-18 00:00:00",
#           1, "credit_card", 1, 18.12, "7c396fd4830fd04220f754e42b4e5bff",
#           3149, "sao paulo", "SP", 1,
#           "3504c0cb71d7fa48d967e0e4c94d59d9", "2017-10-06 11:07:15", 29.99,
#           8.72, "utilidades_domesticas", 40.0, 268.0, 4.0, 500.0, 19.0,
#           8.0, 13.0, "housewares"
#         ]
#       ]
#     }
#   }' \
#   http://127.0.0.1:8000/invocations
