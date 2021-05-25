from typing import List, Tuple

import pandas as pd
import numpy as np
from datetime import timedelta
import datetime

from joblib import load, dump
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os

from config import SCALERS_MODEL_PATH

target_columns = ["ANNULATION", "ATTERRISSAGE", "DECOLLAGE", "DETOURNEMENT",
                  "HEURE D'ARRIVEE", "HEURE DE DEPART", "RAISON D'ANNULATION",
                  "RETARD A L'ARRIVEE", "RETARD AVION", "RETARD COMPAGNIE",
                  "RETARD METEO", "RETARD SECURITE", "RETARD SYSTEM", "RETART DE DEPART",
                  "TEMPS DE VOL", "TEMPS PASSE"]

list_features_to_scale = ['TEMPS PROGRAMME', 'DISTANCE', 'TEMPS DE DEPLACEMENT A TERRE AU DECOLLAGE',
                          "TEMPS DE DEPLACEMENT A TERRE A L'ATTERRISSAGE", "NOMBRE DE PASSAGERS"]


def build_features(df_vols: pd.DataFrame, features_to_scale: List[str], path_for_scaler: str) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    """
    function that do the all preprocessing to the dataframe df_vols that will be used for the model
    """
    # Drop irrelevant columns and handling missing values
    df_vols = delete_irrelevant_columns(df_vols)
    df_target = df_vols[target_columns]
    df_without_target = df_vols.drop(columns=target_columns)
    df_without_target, deleted_indexes = handle_missing_values(df_without_target)
    df_target = df_target.drop(deleted_indexes).reset_index(drop=True)

    add_night_flight_binary_features(df_without_target)
    df_without_target = extracting_time_features_from_date(df_without_target)
    change_hour_format(df_without_target)

    # Scaling
    df_without_target = scale_features(df_without_target, features_to_scale, path=path_for_scaler,
                                       is_train_dataset=True)

    # Create RETARD binary target
    add_delay_binary_target(df_target)
    df_target["CATEGORIE RETARD"] = df_target["RETARD A L'ARRIVEE"].apply(lambda x: add_categorical_delay_target(x))
    df_without_target = df_without_target.drop(
        columns=["DEPART PROGRAMME", "ARRIVEE PROGRAMMEE", "IDENTIFIANT", "DATE", "VOL", "CODE AVION"])
    return df_without_target, df_target


def add_night_flight_binary_features(df_without_target: pd.DataFrame):
    create_is_night_flight_feature('DEPART PROGRAMME', "DEPART DE NUIT", df_without_target)
    create_is_night_flight_feature('ARRIVEE PROGRAMMEE', "ARRIVEE DE NUIT", df_without_target)


def create_is_night_flight_feature(feature: str, is_night_flight_feature: str, df_without_target: pd.DataFrame):
    df_without_target[is_night_flight_feature] = 0
    df_without_target.loc[
        (df_without_target[feature] >= 2300) | (
                df_without_target[feature] <= 600), is_night_flight_feature] = 1
    return df_without_target


def change_hour_format(df_without_target: pd.DataFrame):
    df_without_target["ARRIVEE PROGRAMMEE"] = df_without_target["ARRIVEE PROGRAMMEE"].astype(str).apply(
        lambda x: format_hour(x))
    df_without_target["DEPART PROGRAMME"] = df_without_target["DEPART PROGRAMME"].astype(str).apply(
        lambda x: format_hour(x))


def add_delay_binary_target(df_target: pd.DataFrame):
    df_target["RETARD"] = 0
    df_target.loc[df_target["RETARD A L'ARRIVEE"] > 0, 'RETARD'] = 1


def add_categorical_delay_target(retard_a_larrivee_du_vol: float):
    if retard_a_larrivee_du_vol <= 0:
        return 0
    elif retard_a_larrivee_du_vol <= 180:
        return 1
    else:
        return 2


def get_category_delay_target_in_string(x):
    if x == 0:
        return 'on time'
    elif x == 1:
        return "delay <= 3h"
    else:
        return "delay > 3h"


def extracting_time_features_from_date(df_without_target: pd.DataFrame):
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


def handle_missing_values(df: pd.DataFrame):
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


def save_scaler(sc: StandardScaler, path: str, feature: str):
    dump(sc, path + f'/{feature}_std_scaler.bin', compress=True)


def load_scaler(path: str, feature: str):
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


def format_date(df_vols, df_fuel):
    period_of_flights = df_vols['DATE'].max() - df_vols['DATE'].min()
    scaler = MinMaxScaler()
    df_fuel["DATE"] = (scaler.fit_transform(np.array(df_fuel["DATE"]).reshape(-1, 1)) * period_of_flights.days).astype(
        int)
    df_fuel['DATE FORMATTE'] = df_fuel.apply(lambda x: calculate_date(x, df_vols['DATE'].min()), axis=1)
    return df_fuel.drop(columns="DATE")


def calculate_date(x, first_date):
    return first_date + timedelta(days=int(x['DATE']))


def main():
    print("Début de la lecture des datasets utilisés pour la phase d'entraînement...")
    vols = pd.read_parquet("../../data/parquet_format/train_data/vols.gzip")
    vols, target = build_features(vols, list_features_to_scale, SCALERS_MODEL_PATH)
    print("Création du jeu d'entraînement ...")
    vols.to_parquet("../../data/processed/train_data/train.gzip", compression='gzip')
    target.to_parquet("../../data/processed/train_data/train_target.gzip", compression='gzip')
    print("Fin")


if __name__ == '__main__':
    main()
