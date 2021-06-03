import pandas as pd
import os


def aggregate_data(input_filepath, output_filepath):
    """ Aggregates all the tables from the different batches.
    """
    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)
    batches = os.listdir(input_filepath)
    tables = ['aeroports', 'compagnies', 'vols']
    for table in tables:
        print("Aggrégation des données de la table " + str(table))
        df1 = pd.read_parquet(input_filepath + "/" + str(batches[0]) + "/" + str(table) + ".gzip")
        df2 = pd.read_parquet(input_filepath + "/" + str(batches[1]) + "/" + str(table) + ".gzip")
        df = concat_dataframes(df1, df2)
        df.to_parquet(output_filepath + "/" + str(table) + '.gzip', compression='gzip')
    # For the dataframe containing the price of the fuel, we keep the parquet raw file
    print("Aggrégation des données de la table prix_fuel")
    df_fuel = pd.read_parquet("../../data/raw/fuel.parquet")
    df_fuel.to_parquet(output_filepath + "/prix_fuel.gzip", compression='gzip')


def concat_dataframes(df1, df2):
    df = pd.concat([df1, df2])
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    return df


if __name__ == '__main__':
    aggregate_data("../../data/extracted/train_data", "../../data/aggregated_data")