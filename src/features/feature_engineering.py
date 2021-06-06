import datetime
import logging
import os
from datetime import timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd
from joblib import load, dump
from sklearn.preprocessing import MinMaxScaler, StandardScaler

SCALERS_MODEL_PATH = os.path.join("../../models/train_features_scalers")

target_columns = ["ANNULATION", "ATTERRISSAGE", "DECOLLAGE", "DETOURNEMENT",
                  "HEURE D'ARRIVEE", "HEURE DE DEPART", "RAISON D'ANNULATION",
                  "RETARD A L'ARRIVEE", "RETARD AVION", "RETARD COMPAGNIE",
                  "RETARD METEO", "RETARD SECURITE", "RETARD SYSTEM", "RETART DE DEPART",
                  "TEMPS DE VOL", "TEMPS PASSE"]

list_features_to_scale = ['TEMPS PROGRAMME', 'DISTANCE', 'TEMPS DE DEPLACEMENT A TERRE AU DECOLLAGE',
                          "TEMPS DE DEPLACEMENT A TERRE A L'ATTERRISSAGE", "NOMBRE DE PASSAGERS", "PRIX DU BARIL"]


#FIXME Ajouter des typehinting
def build_features_for_train(df_flights: pd.DataFrame, df_fuel: pd.DataFrame, features_to_scale: List[str],
                             path_for_scaler: str, delay_param=0) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    """
    function that do the all preprocessing to the dataframe df_flights that will be used for the model
    """
    df_flights = add_price_fuel(df_flights, df_fuel)
    df_flights = delete_irrelevant_columns(df_flights)
    df_target = df_flights[target_columns]
    df_without_target = df_flights.drop(columns=target_columns)
    df_without_target, deleted_indexes = handle_missing_values(df_without_target)
    df_target = df_target.drop(deleted_indexes).reset_index(drop=True)

    add_night_flight_binary_feature(df_without_target)
    df_without_target = extracting_time_features_from_date(df_without_target)
    change_hour_format(df_without_target)

    # Scaling
    df_without_target = scale_features(df_without_target, features_to_scale, path=path_for_scaler,
                                       is_train_dataset=True)

    # Create RETARD binary target
    add_delay_binary_target(df_target, delay_param=delay_param)
    df_target["CATEGORIE RETARD"] = df_target["RETARD A L'ARRIVEE"].apply(lambda x: add_categorical_delay_target(x))
    df_without_target = df_without_target.drop(
        columns=["DEPART PROGRAMME", "ARRIVEE PROGRAMMEE", "IDENTIFIANT", "DATE", "VOL", "CODE AVION"])
    return df_without_target, df_target


def build_features_for_test(df_flights: pd.DataFrame, df_fuel: pd.DataFrame, features_to_scale: List[str],
                            path_for_scaler: str) -> pd.DataFrame:
    """
    function that do the all preprocessing to the dataframe df_flights that will be used for the model
    """
    df_without_target = add_price_fuel(df_flights, df_fuel)
    df_without_target = delete_irrelevant_columns(df_without_target)
    df_without_target, deleted_indexes = handle_missing_values(df_without_target)
    add_night_flight_binary_feature(df_without_target)
    df_without_target = extracting_time_features_from_date(df_without_target)
    change_hour_format(df_without_target)

    # Scaling
    df_without_target = scale_features(df_without_target, features_to_scale, path=path_for_scaler,
                                       is_train_dataset=False)

    # Create RETARD binary target
    df_without_target = df_without_target.drop(
        columns=["DEPART PROGRAMME", "ARRIVEE PROGRAMMEE", "IDENTIFIANT", "DATE", "VOL", "CODE AVION"])
    return df_without_target


def build_features(df_flights: pd.DataFrame, df_fuel: pd.DataFrame, features_to_scale: List[str], path_for_scaler: str,
                   TRAIN_OR_TEST: str, delay_param: int=0):
    if TRAIN_OR_TEST == "TRAIN":
        return build_features_for_train(df_flights, df_fuel, features_to_scale, path_for_scaler, delay_param)
    if TRAIN_OR_TEST == "TEST":
        return build_features_for_test(df_flights, df_fuel, features_to_scale, path_for_scaler)


def add_price_fuel(df_flights: pd.DataFrame, df_fuel: pd.DataFrame) -> pd.DataFrame:
    df_fuel["DATE"] = pd.to_datetime(df_fuel["DATE"])
    df_flights = pd.merge(df_flights, df_fuel, on="DATE", how="left")
    df_flights["PRIX DU BARIL"] = df_flights["PRIX DU BARIL"].fillna(df_flights["PRIX DU BARIL"].mean())
    return df_flights


def add_night_flight_binary_feature(df_without_target: pd.DataFrame):
    create_is_night_flight_feature('DEPART PROGRAMME', "DEPART DE NUIT", df_without_target)
    create_is_night_flight_feature('ARRIVEE PROGRAMMEE', "ARRIVEE DE NUIT", df_without_target)


def create_is_night_flight_feature(feature: str, is_night_flight_feature: str, df_without_target: pd.DataFrame) -> pd.DataFrame:
    df_without_target[is_night_flight_feature] = 0
    df_without_target.loc[
        (df_without_target[feature] >= 2300) | (
                df_without_target[feature] <= 600), is_night_flight_feature] = 1
    return df_without_target


def change_hour_format(df_without_target: pd.DataFrame) -> None:
    df_without_target["ARRIVEE PROGRAMMEE"] = df_without_target["ARRIVEE PROGRAMMEE"].astype(str).apply(
        lambda x: format_hour(x))
    df_without_target["DEPART PROGRAMME"] = df_without_target["DEPART PROGRAMME"].astype(str).apply(
        lambda x: format_hour(x))


def add_delay_binary_target(df_target: pd.DataFrame, delay_param: int = 0) -> None:
    df_target["RETARD"] = 0
    df_target.loc[df_target["RETARD A L'ARRIVEE"] > delay_param, 'RETARD'] = 1


def add_categorical_delay_target(retard_a_larrivee_du_vol: float) -> None:
    if retard_a_larrivee_du_vol <= 0:
        return 0
    elif retard_a_larrivee_du_vol <= 180:
        return 1
    else:
        return 2


def get_category_delay_target_in_string(x) -> str:
    if x == 0:
        return "A l'heure"
    elif x == 1:
        return "Retard <= 3h"
    else:
        return "Retard > 3h"


def extracting_time_features_from_date(df_without_target: pd.DataFrame) -> pd.DataFrame:
    df_without_target['DAY OF THE WEEK'] = df_without_target['DATE'].dt.dayofweek + 1
    df_without_target['WEEKEND'] = df_without_target['DAY OF THE WEEK'].apply(lambda x: check_weekend(x))
    df_without_target['MONTH'] = df_without_target['DATE'].dt.month
    df_without_target['DAY OF THE MONTH'] = df_without_target['DATE'].dt.day
    df_without_target["HEURE DE DEPART"] = df_without_target['DEPART PROGRAMME'].apply(
        lambda x: convert_time_into_datetime(x).hour)
    df_without_target["HEURE D'ARRIVEE"] = df_without_target['ARRIVEE PROGRAMMEE'].apply(
        lambda x: convert_time_into_datetime(x).hour)
    return df_without_target


def delete_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["NIVEAU DE SECURITE"])


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    indexes = df.index
    df = df.dropna()
    deleted_indexes = indexes.difference(df.index)
    return df.reset_index(drop=True), deleted_indexes


def format_hour(x: str) -> pd.to_timedelta:
    while len(x) < 4:
        x = '0' + x
    return pd.to_timedelta(x[:-2] + ':' + x[-2:] + ':00')


def convert_time_into_datetime(time_val):
    if pd.isnull(time_val):
        return np.nan
    else:
        if time_val == 2400: time_val = 0
        time_val = "{0:04d}".format(int(time_val))  # transform : 0612
        time_formatted = datetime.time(int(time_val[0:2]), int(time_val[2:4]))
    return time_formatted


def check_weekend(x: int) -> int:
    return 1 if x > 5 else 0


def save_scaler(sc: StandardScaler, path: str, feature: str) -> None:
    if os.path.exists(path + f'/{feature}_std_scaler.bin'):
        dump(sc, path + f'/{feature}_std_scaler.bin', compress=True)
    else :
        with open(path + f'/{feature}_std_scaler.bin', 'x') as f:
            dump(sc, path + f'/{feature}_std_scaler.bin', compress=True)


def load_scaler(path: str, feature: str) -> None:
    return load(path + f'/{feature}_std_scaler.bin')


def scale_feature_in_df(df: pd.DataFrame, feature: str, path: str, is_train_dataset: bool = True) -> pd.Series:
    if is_train_dataset:
        scaler_feature = StandardScaler()
        scaler_feature = scaler_feature.fit(np.array(df[feature]).reshape(-1, 1))
        save_scaler(scaler_feature, path, feature)
    else:
        scaler_feature = load_scaler(path, feature)
    return scaler_feature.transform(np.array(df[feature]).reshape(-1, 1))


def scale_features(df: pd.DataFrame, features_to_scale: List[str], path: str,
                   is_train_dataset: bool = True) -> pd.DataFrame:
    for feature in features_to_scale:
        scale_feature_in_df(df, feature, path, is_train_dataset)
    return df


def format_date(df_flights, df_fuel) -> pd.DataFrame:
    period_of_flights = df_flights['DATE'].max() - df_flights['DATE'].min()
    scaler = MinMaxScaler()
    df_fuel["DATE"] = (scaler.fit_transform(np.array(df_fuel["DATE"]).reshape(-1, 1)) * period_of_flights.days).astype(
        int)
    df_fuel['DATE FORMATTE'] = df_fuel.apply(lambda x: calculate_date(x, df_flights['DATE'].min()), axis=1)
    return df_fuel.drop(columns="DATE")


def calculate_date(x, first_date) -> datetime:
    return first_date + timedelta(days=int(x['DATE']))


def main_feature_engineering(delay):
    logging.info("Début de la lecture des datasets utilisés pour la phase d'entraînement...")
    flights = pd.read_parquet("../../data/aggregated_data/vols.gzip")
    fuel = pd.read_parquet("../../data/aggregated_data/prix_fuel.gzip")
    flights, target = build_features(flights, fuel, list_features_to_scale, SCALERS_MODEL_PATH, "TRAIN", delay_param=delay)
    logging.info("Création du jeu d'entraînement ...")
    flights.to_parquet("../../data/processed/train_data/train.gzip", compression='gzip')
    target.to_parquet("../../data/processed/train_data/train_target.gzip", compression='gzip')
    logging.info("Fin")


if __name__ == '__main__':
    main_feature_engineering(10)
