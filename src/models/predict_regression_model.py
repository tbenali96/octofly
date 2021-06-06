import logging
import pickle
import pandas as pd


def predict_regressor(X: pd.DataFrame, model_file_name: str) -> pd.DataFrame:
    model = pickle.load(open(model_file_name, 'rb'))
    X = X[X["RETARD"] == 1]
    X = X.drop(columns=["RETARD"])

    cat_features = [0, 1, 6, 11, 13, 14, 15, 16]
    for i in cat_features:
        X.iloc[:, i] = X.iloc[:, i].astype('category')

    predictions = model.predict(X)
    X["RETARD MINUTES"] = predictions
    return X


if __name__ == '__main__':
    predictions_retard = pd.read_parquet("../../data/predictions/predictions_classification.gzip")
    logging.info("Prédiction des minutes de retard")
    preds = predict_regressor(predictions_retard, "../../models/model_regression.sav")
    preds.to_parquet("../../data/predictions/predictions_regression.gzip", compression='gzip')
    logging.info("Fin")
