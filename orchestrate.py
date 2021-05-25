from prefect import Flow

from src.data.make_dataset import read_database_and_store_in_parquet
from src.features.feature_engineering import main_feature_engineering

with Flow("Chaîne de traitement") as flow:
    first_task = read_database_and_store_in_parquet("data/raw/batch_1.db", "data/parquet/train_data")
    second_task = main_feature_engineering()
    flow.add_edge(first_task, second_task)

flow.register(project_name="OCTOFLY")



