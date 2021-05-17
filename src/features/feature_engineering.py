import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler

target_columns = ["ANNULATION", "ATTERRISSAGE", "DECOLLAGE", "DETOURNEMENT",
                  "HEURE D'ARRIVEE", "HEURE DE DEPART", "RAISON D'ANNULATION",
                  "RETARD A L'ARRIVEE", "RETARD AVION", "RETARD COMPAGNIE",
                  "RETARD METEO", "RETARD SECURITE", "RETARD SYSTEM", "RETART DE DEPART",
                  "TEMPS DE VOL", "TEMPS PASSE"]


def build_features(df_vols):
    # Drop irrelevant columns and handling missing values
    df_vols = delete_irrelevant_columns(df_vols)
    df_target = df_vols[target_columns]
    df_without_target = df_vols.drop(columns=target_columns)
    df_without_target, deleted_indexes = handle_missing_values(df_without_target)
    df_target = df_target.drop(deleted_indexes).reset_index(drop=True)

    df_without_target["DEPART DE NUIT"] = 0
    df_without_target.loc[
        (df_vols['DEPART PROGRAMME'] >= 2300) | (df_without_target['DEPART PROGRAMME'] <= 600), "DEPART DE NUIT"] = 1
    df_without_target["ARRIVEE DE NUIT"] = 0
    df_without_target.loc[
        (df_vols['ARRIVEE PROGRAMMEE'] >= 2300) | (
                    df_without_target['ARRIVEE PROGRAMMEE'] <= 600), "ARRIVEE DE NUIT"] = 1

    # Changing format
    df_without_target["ARRIVEE PROGRAMMEE"] = df_without_target["ARRIVEE PROGRAMMEE"].astype(str).apply(
        lambda x: format_hour(x))
    df_without_target["DEPART PROGRAMME"] = df_without_target["DEPART PROGRAMME"].astype(str).apply(
        lambda x: format_hour(x))

    # Scaling
    df_without_target = scale(df_without_target)

    # Extracting data from date
    df_without_target['DAY OF THE WEEK'] = df_without_target['DATE'].dt.dayofweek + 1
    df_without_target['WEEKEND'] = df_without_target['DAY OF THE WEEK'].apply(lambda x: check_weekend(x))
    df_without_target['MONTH'] = df_without_target['DATE'].dt.month
    df_without_target['DAY OF THE MONTH'] = df_without_target['DATE'].dt.day

    df_without_target["HEURE DE DEPART"] = df_without_target['DEPART PROGRAMME'].dt.components['hours']
    df_without_target["HEURE D'ARRIVEE"] = df_without_target['ARRIVEE PROGRAMMEE'].dt.components['hours']

    df_target["RETARD"] = 0
    df_target.loc[df_target["RETARD A L'ARRIVEE"] > 0, 'RETARD'] = 1

    return df_without_target, df_target


def delete_irrelevant_columns(df):
    return df.drop(columns=["NIVEAU DE SECURITE"])


def handle_missing_values(df):
    indexes = df.index
    df = df.dropna()
    deleted_indexes = indexes.difference(df.index)
    return df.reset_index(drop=True), deleted_indexes


def format_hour(x):
    while len(x) < 4:
        x = '0' + x
    return pd.to_timedelta(x[:-2] + ':' + x[-2:] + ':00')


def check_weekend(x):
    return 1 if x > 5 else 0


def scale(df):
    scaler_temps_programme = StandardScaler()
    scaler_temps_programme = scaler_temps_programme.fit(np.array(df['TEMPS PROGRAMME']).reshape(-1, 1))
    df['TEMPS PROGRAMME'] = scaler_temps_programme.transform(np.array(df['TEMPS PROGRAMME']).reshape(-1, 1))

    scaler_distance = StandardScaler()
    scaler_distance = scaler_distance.fit(np.array(df['DISTANCE']).reshape(-1, 1))
    df['DISTANCE'] = scaler_distance.transform(np.array(df['DISTANCE']).reshape(-1, 1))

    scaler_temps_deplacement_decollage = StandardScaler()
    scaler_temps_deplacement_decollage = scaler_temps_deplacement_decollage.fit(
        np.array(df['TEMPS DE DEPLACEMENT A TERRE AU DECOLLAGE']).reshape(-1, 1))
    df['TEMPS DE DEPLACEMENT A TERRE AU DECOLLAGE'] = scaler_temps_deplacement_decollage.transform(
        np.array(df['TEMPS DE DEPLACEMENT A TERRE AU DECOLLAGE']).reshape(-1, 1))

    scaler_temps_deplacement_atterrissage = StandardScaler()
    scaler_temps_deplacement_atterrissage = scaler_temps_deplacement_atterrissage.fit(
        np.array(df["TEMPS DE DEPLACEMENT A TERRE A L'ATTERRISSAGE"]).reshape(-1, 1))
    df["TEMPS DE DEPLACEMENT A TERRE A L'ATTERRISSAGE"] = scaler_temps_deplacement_atterrissage.transform(
        np.array(df["TEMPS DE DEPLACEMENT A TERRE A L'ATTERRISSAGE"]).reshape(-1, 1))

    scaler_nombre_passagers = StandardScaler()
    scaler_nombre_passagers = scaler_nombre_passagers.fit(np.array(df["NOMBRE DE PASSAGERS"]).reshape(-1, 1))
    df["NOMBRE DE PASSAGERS"] = scaler_nombre_passagers.transform(np.array(df["NOMBRE DE PASSAGERS"]).reshape(-1, 1))

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


if __name__ == '__main__':
    print("Début de la lecture des datasets utilisés pour la phase d'entraînement...")
    vols = pd.read_parquet("../../data/processed/train_data/vols.gzip")
    prix_fuel = pd.read_parquet("../../data/processed/train_data/prix_fuel.gzip")
    print("Lecture des datasets terminée")
    vols, target = build_features(vols)
    vols = vols.drop(columns=["DEPART PROGRAMME", "ARRIVEE PROGRAMMEE", "IDENTIFIANT", "DATE", "VOL", "CODE AVION"])
    print("Création du jeu d'entraînement ...")
    vols.to_parquet("../../data/processed/train_data/train.gzip", compression='gzip')
    target.to_parquet("../../data/processed/train_data/train_target.gzip", compression='gzip')
    print("Fin")
