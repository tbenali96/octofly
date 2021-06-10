import logging

import pandas as pd
import os


def aggregate_data(input_filepath: str, output_filepath: str) -> None:
    """ Aggregates all the data extracted from the different databases
    and make it ready to be used in the feature engineering process. It is stored in ../aggregated_data.
    """
    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)
    batches = os.listdir(input_filepath)
    if ".DS_Store" in batches:
        batches.remove(".DS_Store")
    tables = ['aeroports', 'compagnies', 'vols']
    for table in tables:
        logging.info("Aggrégation des données de la table " + str(table))
        df1 = pd.read_parquet(f'{input_filepath}/{batches[0]}/{table}.gzip')
        df2 = pd.read_parquet(f'{input_filepath}/{batches[1]}/{table}.gzip')
        df = concat_dataframes(df1,df2)
        df.to_parquet(output_filepath + "/" + str(table) + '.gzip', compression='gzip')
    # For the dataframe containing the price of the fuel, we keep the parquet raw file
    logging.info("Aggrégation des données de la table prix_fuel")
    df_fuel = pd.read_parquet("../../data/raw/fuel.parquet")
    df_fuel.to_parquet(f'{output_filepath}/prix_fuel.gzip', compression='gzip')


def concat_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([df1, df2])
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    return df


if __name__ == '__main__':
    aggregate_data("../../data/extracted/train_data", "../../data/aggregated_data")
