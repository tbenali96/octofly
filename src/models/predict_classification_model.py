import logging
import pickle
import pandas as pd
from src.features.feature_engineering import build_features
import os
import streamlit as st

list_features_to_scale = ['TEMPS PROGRAMME', 'DISTANCE', 'TEMPS DE DEPLACEMENT A TERRE AU DECOLLAGE',
                          "TEMPS DE DEPLACEMENT A TERRE A L'ATTERRISSAGE", "NOMBRE DE PASSAGERS", "PRIX DU BARIL"]

SCALERS_MODEL_PATH = os.path.join("../../models/train_features_scalers")


def calculate_prediction(x: float, threshold: float) -> int:
    """
    Returns a class depending on the value of the threshold.
    """
    if x <= threshold:
        return 0
    else:
        return 1


def predict_classifier(X: pd.DataFrame, model_file_name: str, threshold: float=0.2) -> pd.DataFrame:
    """
    Predicts a delay or not for each record of the test dataset.
    """
    model = pickle.load(open(model_file_name, 'rb'))
    preds_proba = model.predict_proba(X)
    predictions = []
    for i in range(preds_proba.shape[0]):
        predictions.append(preds_proba[i][1])
    X["RETARD"] = predictions
    X["RETARD"] = X["RETARD"].apply(lambda x: calculate_prediction(x, threshold))
    return X


def main_prediction(flight_path, fuel_path, scaler_model_path, model_save_path, pred_classif_path):
    flights = pd.read_parquet(flight_path)
    fuel = pd.read_parquet(fuel_path)
    logging.info("Construction des features du dataset de test")
    flights = build_features(flights, fuel, list_features_to_scale, scaler_model_path, "TEST")
    logging.info("Prédiction du retard ou du non-retard")
    preds = predict_classifier(flights, model_save_path)
    preds.to_parquet(pred_classif_path, compression='gzip')
    logging.info("Fin")


def st_main_prediction(flight_path, fuel_path, scaler_model_path, model_save_path, pred_classif_path):
    flights = pd.read_parquet(flight_path)
    fuel = pd.read_parquet(fuel_path)
    st.text("Construction des features du dataset de test")
    flights = build_features(flights, fuel, list_features_to_scale, scaler_model_path, "TEST")
    st.text("Prédiction du retard ou du non-retard")
    preds = predict_classifier(flights, model_save_path)
    preds.to_parquet(pred_classif_path, compression='gzip')
    st.text("Fin")


if __name__ == '__main__':
    flight_path = "../../data/extracted/test_data/vols.gzip"
    fuel_path = "../../data/aggregated_data/prix_fuel.gzip"
    model_save_path = "../../models/model_classification.sav"
    pred_classif_path = "../../data/predictions/predictions_classification.gzip"
    main_prediction(flight_path, fuel_path, SCALERS_MODEL_PATH)
