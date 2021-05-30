import pickle
import pandas as pd
from src.features.feature_engineering import build_features
from config import SCALERS_MODEL_PATH


list_features_to_scale = ['TEMPS PROGRAMME', 'DISTANCE', 'TEMPS DE DEPLACEMENT A TERRE AU DECOLLAGE',
                          "TEMPS DE DEPLACEMENT A TERRE A L'ATTERRISSAGE", "NOMBRE DE PASSAGERS", "PRIX DU BARIL"]


def predict(X, model_file_name) -> pd.DataFrame:
    model = pickle.load(open(model_file_name, 'rb'))
    return model.predict(X)


if __name__ == '__main__':
    vols = pd.read_parquet("../../data/processed/test_data/vols.gzip")
    prix_fuel = pd.read_parquet("../../data/aggregated_data/prix_fuel.gzip")
    vols = build_features(vols, prix_fuel, list_features_to_scale, SCALERS_MODEL_PATH, "TEST")
    preds = predict(vols, "../../models/model_prediction.sav")
    #preds.to_parquet("../../data/predictions/predictions.gzip", compression='gzip')
