import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import main as run_main


def main():
    st.title("End to End Customer Satisfaction Pipeline with ZenML")

    high_level_image = Image.open("_assets/high_level_overview.png")
    st.image(high_level_image, caption="High Level Pipeline")

    whole_pipeline_image = Image.open("_assets/training_and_deployment_pipeline_updated.png")

    st.markdown(
        """
    #### Problem Statement
     The objective here is to predict the customer satisfaction score for a given order based on features like order status, price, payment, etc. We will be using [ZenML](https://zenml.io/) to build a production-ready pipeline to predict the customer satisfaction score for the next order or purchase.    """
    )
    st.image(whole_pipeline_image, caption="Whole Pipeline")
    st.markdown(
        """
    Above is a figure of the whole pipeline, we first ingest the data, clean it, train the model, and evaluate the model, and if data source changes or any hyperparameter values changes, deployment will be triggered, and (re) trains the model and if the model meets minimum accuracy requirement, the model will be deployed.
    """
    )

    st.markdown(
        """
    #### Description of Features
    This app is designed to predict the customer satisfaction score for a given customer. You can input the features of the product listed below and get the customer satisfaction score.""")

    col1, col2 = st.columns([1, 1])
    col1.markdown(
        """
    | Models        | Description   |
    | ------------- | -     |
    | Payment Sequential | Customer may pay an order with more than one payment method. If he does so, a sequence will be created to accommodate all payments. |
    | Payment Installments   | Number of installments chosen by the customer. |
    | Payment Value |       Total amount paid by the customer. |
    | Price |       Price of the product. |
    | Freight Value |    Freight value of the product.  |
    | Product Name length |    Length of the product name. |""")
    col2.markdown(
        """
    | Models        | Description   |
    | ------------- | -     |
    | Product Description length |    Length of the product description. |
    | Product photos Quantity |    Number of product published photos |
    | Product weight measured in grams |    Weight of the product measured in grams. |
    | Product length (CMs) |    Length of the product measured in centimeters. |
    | Product height (CMs) |    Height of the product measured in centimeters. |
    | Product width (CMs) |    Width of the product measured in centimeters. |
    """
    )
    payment_sequential = st.sidebar.slider("Payment Sequential")
    payment_installments = st.sidebar.slider("Payment Installments")
    payment_value = st.sidebar.number_input("Payment Value")
    price = st.sidebar.number_input("Price")
    freight_value = st.sidebar.number_input("Freight Value")
    product_name_length = st.sidebar.number_input("Product name length")
    product_description_length = st.sidebar.number_input("Product Description length")
    product_photos_qty = st.sidebar.number_input("Product photos Quantity ")
    product_weight_g = st.sidebar.number_input("Product weight measured in grams")
    product_length_cm = st.sidebar.number_input("Product length (CMs)")
    product_height_cm = st.sidebar.number_input("Product height (CMs)")
    product_width_cm = st.sidebar.number_input("Product width (CMs)")

    pred = None

    if st.sidebar.button("Predict"):
        params = {
            "pipeline_name": "continuous_deployment_pipeline",
            "step_name": "mlflow_model_deployer_step",
            "model_name": "model",
            "running": True
        }

        try:
            service = prediction_service_loader(params)

            if service is None:
                st.warning("No service found. Running deployment pipeline...")
                with st.spinner("Deploying model..."):
                    run_main()
                st.info("Deployment completed! Please click Predict again.")
            else:
                # Prepare input data
                df = pd.DataFrame({
                    "payment_sequential": [payment_sequential],
                    "payment_installments": [payment_installments],
                    "payment_value": [payment_value],
                    "price": [price],
                    "freight_value": [freight_value],
                    "product_name_lenght": [product_name_length],
                    "product_description_lenght": [product_description_length],
                    "product_photos_qty": [product_photos_qty],
                    "product_weight_g": [product_weight_g],
                    "product_length_cm": [product_length_cm],
                    "product_height_cm": [product_height_cm],
                    "product_width_cm": [product_width_cm],
                })

                # Convert to model input format
                json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
                data = np.array(json_list)

                # Make prediction
                pred = service.predict(data)

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

    # Show results if prediction was made
    if pred is not None:
        st.success(
            f"Your Customer Satisfaction rate (range 0-5) is: {pred[0] if isinstance(pred, (list, np.ndarray)) else pred}"
        )
    if st.button("Metrics of Model"):
        st.write(
            "We have experimented with  tree based model (Random Forest). The results are as follows:"
        )

        df = pd.DataFrame(
            {
                "Model": ["Random Forest"],
                "MSE": [0.93],
                "RMSE": [0.96],
                "R2_score": [0.50]
            }
        )
        st.dataframe(df)

        st.write(
            "Following figure shows how important each feature is in the model that contributes to the target variable or contributes in predicting customer satisfaction rate."
        )
        image = Image.open("_assets/feature_importance_gain.png")
        st.image(image, caption="Feature Importance Gain")


if __name__ == "__main__":
    main()
