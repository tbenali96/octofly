from prefect import Flow

from src.data.extract_files_from_database import read_database_and_store_in_parquet
from src.features.feature_engineering import main_feature_engineering

with Flow("Chaîne de traitement") as flow:
    read_first_batch = read_database_and_store_in_parquet("data/raw/batch_1.db", "data/extracted/train_data/batch_1")
    read_second_batch = read_database_and_store_in_parquet("data/raw/batch_2.db", "data/extracted/train_data/batch_2")
    read_test_data = read_database_and_store_in_parquet("data/raw/test.db", "data/extracted/test_data")
    second_task = main_feature_engineering()
    flow.add_edge(read_first_batch, read_second_batch)

flow.register(project_name="OCTOFLY")



