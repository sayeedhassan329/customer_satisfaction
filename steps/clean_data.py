from zenml import step
import pandas as pd
from typing import Tuple, Annotated
import logging
from model.data_cleaning import (DataPreprocessStrategy,
    DataStandardizeStrategy,
    DataEncodingStrategy,
    DataDivideStrategy,
    DataCleaning)


from zenml.materializers.pandas_materializer import PandasMaterializer


@step(output_materializers=PandasMaterializer)
def clean_data(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]]:

    """
    Zenml step which cleans data and returns x_train, x_test, y_train, y_test datasets.
    """
    if data is None:
        raise ValueError("Input data cannot be None")
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(data)}")
    if data.empty:
        raise ValueError("Input DataFrame is empty")

    logging.info(f"Received data with shape: {data.shape}")

    try:
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(data, data, preprocess_strategy)
        preprocessed_data= data_cleaning.handle_data()

        standardize = DataStandardizeStrategy()
        data_cleaning = DataCleaning(preprocessed_data, preprocessed_data, standardize)
        standardized_data = data_cleaning.handle_data()

        encoding_strategy = DataEncodingStrategy()
        data_cleaning = DataCleaning(standardized_data, standardized_data, encoding_strategy)
        encoded_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(encoded_data, encoded_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info(f'Returning cleaned data- x_train:{X_train.shape}, x_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}')
        return X_train, X_test, y_train, y_test

    except Exception as e:
        logging.error(f'Error in clean_data: {str(e)}')
        raise e


if __name__ == "__main__":
    data = pd.read_csv("/home/sayeed-hassan/Desktop/customer_satisfaction_project/data/olist_customers_dataset.csv")
    print(data.head())
    x_train, x_test, y_train, y_test = clean_data(data)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
