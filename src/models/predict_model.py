import pickle
import pandas as pd
from src.features.feature_engineering import build_features


def predict(X, model_file_name) -> pd.DataFrame:
    model = pickle.load(open(model_file_name, 'rb'))
    return model.predict(X)


if __name__ == '__main__':
    X = pd.read_parquet("../../data/processed/test_data/vols.gzip")
    #X = build_features(X)
    preds = predict(X, "../../models/model_prediction.sav")
    preds.to_csv("")
