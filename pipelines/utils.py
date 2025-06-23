import logging
import pandas as pd


from model.data_cleaning import (DataPreprocessStrategy,
    DataStandardizeStrategy,
    DataEncodingStrategy,
    DataCleaning)


def get_data_for_test():
    try:
        dataframe = pd.read_csv("/home/sayeed-hassan/Desktop/customer_satisfaction_project/data/olist_customers_dataset.csv")
        df = dataframe.sample(n=100)

        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(df, df, preprocess_strategy)
        preprocessed_data = data_cleaning.handle_data()


        standardize = DataStandardizeStrategy()
        data_cleaning = DataCleaning(preprocessed_data,preprocessed_data, standardize)
        standardized_data = data_cleaning.handle_data()
        print('Standardized data null value: ', standardized_data.isnull().sum().sum())



        encoding_strategy = DataEncodingStrategy()
        data_cleaning = DataCleaning(dataframe, standardized_data, encoding_strategy)
        df = data_cleaning.handle_data()

        df.drop(["review_score"], axis=1, inplace=True)
        result = df#.to_json(orient="split")
        return result
    except Exception as e:
        logging.error(e)
        raise e

if __name__ == "__main__":
    import json
    json_data = get_data_for_test()
    data = json.loads(json_data)
    columns=data["columns"]
    print(len(columns))
    print(columns)
