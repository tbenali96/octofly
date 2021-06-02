import pickle
import pandas as pd
from src.features.feature_engineering import build_features
import os

list_features_to_scale = ['TEMPS PROGRAMME', 'DISTANCE', 'TEMPS DE DEPLACEMENT A TERRE AU DECOLLAGE',
                          "TEMPS DE DEPLACEMENT A TERRE A L'ATTERRISSAGE", "NOMBRE DE PASSAGERS", "PRIX DU BARIL"]

SCALERS_MODEL_PATH = os.path.join("../features/models/train_features_scalers")

def calculate_prediction(x, threshold):
    if x <= threshold:
        return 0
    else:
        return 1


def predict(X, model_file_name, threshold=0.2):
    model = pickle.load(open(model_file_name, 'rb'))
    preds_proba = model.predict_proba(X)
    predictions = []
    for i in range(preds_proba.shape[0]):
        predictions.append(preds_proba[i][1])
    X["RETARD"] = predictions
    X["RETARD"] = X["RETARD"].apply(lambda x: calculate_prediction(x, threshold))
    return X


if __name__ == '__main__':
    vols = pd.read_parquet("../../data/extracted/test_data/vols.gzip")
    prix_fuel = pd.read_parquet("../../data/aggregated_data/prix_fuel.gzip")
    print("Construction des features du dataset de test")
    vols = build_features(vols, prix_fuel, list_features_to_scale, SCALERS_MODEL_PATH, "TEST")
    print("Prédiction du retard ou du non-retard")
    preds = predict(vols, "../../models/model_classification.sav")
    preds.to_parquet("../../data/predictions/predictions_classification.gzip", compression='gzip')
    print("Fin")

