import pandas as pd
from abc import ABC, abstractmethod
import logging
from typing import Union, Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    Abstract base class for defining data cleaning strategies.
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreprocessStrategy(DataStrategy):
    """
    Concrete strategy for preprocessing data.
    """
    def handle_data(self, data: pd.DataFrame, subset_data: pd.DataFrame) -> pd.DataFrame:
        # Implement data preprocessing logic here
        """
        Removes columns which are not required, fills missing values with mdeian, mean values, and converts the data type to float
        """
        try:
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                    "customer_zip_code_prefix",
                    "order_item_id",

                    "order_id", "customer_id", "product_id","review_comment_message",
                ],
                axis=1,
            )
            data["product_weight_g"].fillna(data["product_weight_g"].median())
            data["product_length_cm"].fillna(data["product_length_cm"].median())
            data["product_height_cm"].fillna(data["product_height_cm"].median())
            data["product_width_cm"].fillna(data["product_width_cm"].median())
            return data
        except Exception as e:
            logging.error(f"Error occurred while preprocessing data: {e}")


#
class DataStandardizeStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame, subset_data: pd.DataFrame) -> pd.DataFrame:
        try:
            # Preserve original index for realignment

            original_index = data.index

            # Separate components (maintain as DataFrames)
            target = data[['review_score']]
            non_numeric = data.select_dtypes(exclude=[np.number])
            numeric_cols = data.select_dtypes(include=[np.number])
            numeric_cols = numeric_cols.columns.difference(['review_score']).tolist()

            if not numeric_cols:
                return data

            scaler = StandardScaler()
            # Convert scaled array to DataFrame WITH ORIGINAL INDEX
            scaled_numeric = pd.DataFrame(
                scaler.fit_transform(data[numeric_cols]),
                columns=numeric_cols,
                index=original_index  # Critical fix
            )

            # Combine using original index
            return pd.concat(
                [scaled_numeric, non_numeric, target],
                axis=1
            ).reset_index(drop=True)  # Optional: clean index

        except Exception as e:
            logging.error(f"Data standardization failed: {e}")
            raise

class DataEncodingStrategy(DataStrategy):
    """
    Concrete strategy for encoding categorical data with one-hot encoding.
    """
    def handle_data(self, data: pd.DataFrame, subset_data: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes categorical data using OneHotEncoder.
        """
        try:
            df_cat = data.select_dtypes(include=[object])
            encoder = OneHotEncoder(sparse_output=False, drop='first')
            encoded_array = encoder.fit(df_cat[['order_status', 'payment_type','customer_state']])

            def encode_subset(data: pd.DataFrame) -> pd.DataFrame:
                # Transform categorical columns
                encoded_array = encoder.transform(data[['order_status', 'payment_type','customer_state']])
                encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(), index=data.index)

                # Return aligned DataFrame
                return pd.concat([data.drop(data.select_dtypes(include=[object]).columns, axis=1), encoded_df], axis=1)
            data = encode_subset(subset_data)

            # df_cat_encoded = pd.DataFrame(
            #     encoded_array,
            #     columns=encoder.get_feature_names_out(['order_status', 'payment_type','customer_state']))
            # data = pd.concat([data.drop(data.select_dtypes(include=[object]).columns, axis=1), df_cat_encoded], axis=1)
            return data
        except Exception as e:
            logging.error(f"Error occurred while encoding data: {e}")
            raise e



class DataDivideStrategy(DataStrategy):
    """
    Data divide strategy which divides data into training and testing sets.
    """

    def handle_data(self, data: pd.DataFrame, subset_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Divides data into training and testing sets.
        """
        try:
            X = data.drop(['review_score'], axis=1)
            y = data['review_score']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error occurred while dividing data: {e}")
            raise e


class DataCleaning:
    """
    Data cleaning class which preprocesses data, standardizes numerical features, and encodes categorical features
    and divides data into training and testing sets.
    """
    def __init__(self, data:pd.DataFrame, subset_data: pd.DataFrame, strategy:DataStrategy) -> None:
        """Initialize the DataCleaning class with specific data and strategy.
        """
        self.df = data
        self.strategy = strategy
        self.subset = subset_data
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data cleaning, preprocessing, and splitting into training and testing sets.
        """
        try:

            return self.strategy.handle_data(self.df, self.subset)

        except Exception as e:
            logging.error(f"Error occurred while cleaning data: {e}")
            raise e



if __name__ == "__main__":
    data = pd.read_csv('/home/sayeed-hassan/Desktop/customer_satisfaction_project/data/olist_customers_dataset.csv')
    print(data.head())
    print(data.shape)
    preprocess_strategy = DataPreprocessStrategy()
    data_cleaning = DataCleaning(data, data, preprocess_strategy)
    preprocessed_data= data_cleaning.handle_data()
    print('preprocessed_data', preprocessed_data.head())

    standardize = DataStandardizeStrategy()
    data_cleaning = DataCleaning(preprocessed_data, preprocessed_data, standardize)
    standardized_data = data_cleaning.handle_data()

    encoding_strategy = DataEncodingStrategy()
    data_cleaning = DataCleaning(standardized_data, standardized_data, encoding_strategy)
    encoded_data = data_cleaning.handle_data()

    divide_strategy = DataDivideStrategy()
    data_cleaning = DataCleaning(encoded_data, encoded_data, divide_strategy)
    X_train, X_test, y_train, y_test = data_cleaning.handle_data()
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print(X_train.head(),X_test.head(), y_train.head(), y_test.head())
