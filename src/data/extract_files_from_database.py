# -*- coding: utf-8 -*-
import logging
import os
import sqlite3

import pandas as pd
from prefect import task


@task
def read_database_and_store_in_parquet(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../parquet).
    """
    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)
    #FIXME Choisir soit l'anglais soit le français
    logging.info('Création de données à partir de données bruts')
    con = sqlite3.connect(input_filepath)
    cursor = con.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    for table in tables:
        for name in table:
            #FIXME Utiliser des f string au lieu de str
            logging.info("Extraction des données de la table " + str(name) + " ...")
            if name == "vols":
                df = pd.read_sql_query("SELECT * from " + str(name), con, parse_dates=['DATE'])
            else:
                df = pd.read_sql_query("SELECT * from " + str(name), con)
            df.to_parquet(output_filepath + "/" + str(name) + '.gzip', compression='gzip')
            logging.info("Fin de l'extraction des données")
    con.close()


if __name__ == '__main__':
    read_database_and_store_in_parquet("../../data/raw/batch_1.db", "../../data/extracted/train_data/batch_1")
    read_database_and_store_in_parquet("../../data/raw/batch_2.db", "../../data/extracted/train_data/batch_2")
    read_database_and_store_in_parquet("../../data/raw/test.db", "../../data/extracted/test_data")
