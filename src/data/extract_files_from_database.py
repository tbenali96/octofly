# -*- coding: utf-8 -*-
import logging
import os
import sqlite3

import pandas as pd
from prefect import task

@task
def read_database_and_store_in_parquet(input_filepath: str, output_filepath: str):
    """ Extracts raw data from the database (../raw) into
        cleaned data ready to be processed (saved in ../extracted).
    """
    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)
    logging.info('Création de données à partir de données bruts')
    con = sqlite3.connect(input_filepath)
    cursor = con.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    for table in tables:
        for name in table:
            #FIXME Utiliser des f string au lieu de str
            logging.info(f'Extraction des données de la table {name}...')
            if name == "vols":
                df = pd.read_sql_query(f'SELECT * from  {name}', con, parse_dates=['DATE'])
            else:
                df = pd.read_sql_query(f'SELECT * from  {name}', con)
            df.to_parquet(f'{output_filepath}/{name}.gzip', compression='gzip')
            logging.info("Fin de l'extraction des données")
    con.close()


if __name__ == '__main__':
    read_database_and_store_in_parquet("../../data/raw/batch_1.db", "../../data/extracted/train_data/batch_1")
    read_database_and_store_in_parquet("../../data/raw/batch_2.db", "../../data/extracted/train_data/batch_2")
    read_database_and_store_in_parquet("../../data/raw/test.db", "../../data/extracted/test_data")
