import logging
import os
import pickle

import pandas as pd
from prefect import Flow, task, Parameter

from src.data.extract_files_from_database import read_database_and_store_in_parquet
from src.features.feature_engineering import build_features, list_features_to_scale
from src.models.predict_classification_model import predict_classifier
from src.models.predict_regression_model import predict_regressor
from src.data.aggregate_data import concat_dataframes
from src.models.train_classification_model import train_classifier
from src.models.train_regression_model import train_regressor

SCALERS_MODEL_PATH = os.path.join("models/train_features_scalers")

@task
def aggregate_data(input_filepath: str, output_filepath: str):
    """ Aggregates all the tables from the different batches.
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
        df = concat_dataframes(df1, df2)
        df.to_parquet(output_filepath + "/" + str(table) + '.gzip', compression='gzip')
    # For the dataframe containing the price of the fuel, we keep the parquet raw file
    logging.info("Aggrégation des données de la table prix_fuel")
    df_fuel = pd.read_parquet("data/raw/fuel.parquet")
    df_fuel.to_parquet(f'{output_filepath}/prix_fuel.gzip', compression='gzip')


@task
def main_feature_engineering(delay: int):
    logging.info("Début de la lecture des datasets utilisés pour la phase d'entraînement...")
    flights = pd.read_parquet("data/aggregated_data/vols.gzip")
    fuel = pd.read_parquet("data/aggregated_data/prix_fuel.gzip")
    flights, target = build_features(flights, fuel, list_features_to_scale, SCALERS_MODEL_PATH, "TRAIN", delay_param=delay)
    logging.info("Création du jeu d'entraînement ...")
    flights.to_parquet("data/processed/train_data/train.gzip", compression='gzip')
    target.to_parquet("data/processed/train_data/train_target.gzip", compression='gzip')
    logging.info("Fin")


@task
def main_training_classifier():
    X = pd.read_parquet("data/processed/train_data/train.gzip")
    y = pd.read_parquet("data/processed/train_data/train_target.gzip")
    logging.info("Entraînement du modèle de classification")
    model = train_classifier(X, y)
    logging.info("Fin de l'entraînement")
    filename = 'models/model_classification.sav'
    pickle.dump(model, open(filename, 'wb'))


@task
def main_training_regressor():
    X = pd.read_parquet("data/processed/train_data/train.gzip")
    y = pd.read_parquet("data/processed/train_data/train_target.gzip")
    logging.info("Entraînement du modèle de regression")
    model = train_regressor(X, y)
    logging.info("Fin de l'entraînement")
    filename = 'models/model_regression.sav'
    pickle.dump(model, open(filename, 'wb'))

@task
def main_prediction_classifier():
    flights = pd.read_parquet("data/extracted/test_data/vols.gzip")
    fuel = pd.read_parquet("data/aggregated_data/prix_fuel.gzip")
    logging.info("Construction des features du dataset de test")
    flights = build_features(flights, fuel, list_features_to_scale, SCALERS_MODEL_PATH, "TEST")
    logging.info("Prédiction du retard ou du non-retard")
    preds = predict_classifier(flights, "models/model_classification.sav")
    preds.to_parquet("data/predictions/predictions_classification.gzip", compression='gzip')
    logging.info("Fin")


@task
def main_prediction_regressor():
    predictions_retard = pd.read_parquet("data/predictions/predictions_classification.gzip")
    logging.info("Prédiction des minutes de retard")
    preds = predict_regressor(predictions_retard, "models/model_regression.sav")
    preds.to_parquet("data/predictions/predictions_regression.gzip", compression='gzip')
    logging.info("Fin")


with Flow("Chaîne de traitement") as flow:
    read_first_batch = read_database_and_store_in_parquet("data/raw/batch_1.db", "data/extracted/train_data/batch_1")
    read_second_batch = read_database_and_store_in_parquet("data/raw/batch_2.db", "data/extracted/train_data/batch_2")
    aggregate_data_task = aggregate_data("data/extracted/train_data", "data/aggregated_data")

    delay_param = Parameter('delay_param')
    feature_engineering = main_feature_engineering(delay_param)

    train_classification_model = main_training_classifier()
    train_regression_model = main_training_regressor()

    read_test_data = read_database_and_store_in_parquet("data/raw/test.db", "data/extracted/test_data")
    predict_class = main_prediction_classifier()
    predict_delay_duration = main_prediction_regressor()

    flow.add_edge(read_first_batch, aggregate_data_task)
    flow.add_edge(read_second_batch, aggregate_data_task)
    flow.add_edge(aggregate_data_task, feature_engineering)
    flow.add_edge(feature_engineering, train_classification_model)
    flow.add_edge(feature_engineering, train_regression_model)
    flow.add_edge(read_test_data, predict_class)
    flow.add_edge(train_classification_model, predict_class)
    flow.add_edge(predict_class, predict_delay_duration)
    flow.add_edge(train_regression_model, predict_delay_duration)

flow.register(project_name="OCTOFLY")
