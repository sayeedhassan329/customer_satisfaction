
import logging
import pandas as pd
from zenml import step

class IngestData:
    """
    Data ingestion class which can ingest data from source and return dataframe.
    """
    def __init__(self) -> None:
        """
        Initialize the IngestData class.
        """
        pass
    def get_data(self, data_path: str) -> pd.DataFrame:
        """
        Get data from source and return dataframe.
        """
        df = pd.read_csv(data_path)
        return df


from zenml.materializers.pandas_materializer import PandasMaterializer


@step(output_materializers=PandasMaterializer)
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Ingest data from source and return dataframe.

    Args:
        data_path: Path to the data file.
    Returns:
        DataFrame: Ingested data.
    """
    try:
        ingest_data = IngestData()
        df = ingest_data.get_data(data_path)
        if df.empty:
            raise ValueError("Dataframe is empty")
        return df
    except Exception as e:
        logging.error(f"Error occurred while ingesting data: {e}")
        raise e


if __name__ == "__main__":
    df = ingest_data("/home/sayeed-hassan/Desktop/customer_satisfaction_project/data/olist_customers_dataset.csv")
    print(df.head())
