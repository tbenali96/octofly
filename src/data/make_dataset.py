# -*- coding: utf-8 -*-
import logging
import sqlite3
import pandas as pd


def read_database_and_store_in_parquet(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Creating data from raw data')
    con = sqlite3.connect(input_filepath)
    cursor = con.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    for table in tables:
        for name in table:
            print("Extraction des données de la table " + str(name) + " ...")
            if name == "vols":
                df = pd.read_sql_query("SELECT * from " + str(name), con, parse_dates=['DATE'])
            else:
                df = pd.read_sql_query("SELECT * from " + str(name), con)
            df.to_parquet(output_filepath + "/" + str(name) + '.gzip', compression='gzip')
            print("Fin de l'extraction des données")
    con.close()


if __name__ == '__main__':
    read_database_and_store_in_parquet("../../data/raw/batch_1.db", "../../data/processed/train_data")
    read_database_and_store_in_parquet("../../data/raw/test.db", "../../data/processed/test_data")
