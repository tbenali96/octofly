import logging
import pickle
import pandas as pd


def predict_delay_duration(X, model_file_name):
    model = pickle.load(open(model_file_name, 'rb'))
    X = X[X["RETARD"] == 1]
    X = X.drop(columns=["RETARD"])
    predictions = model.predict_delay_duration(X, )
    X["RETARD MINUTES"] = predictions
    return X


if __name__ == '__main__':
    predictions_retard = pd.read_parquet("../../data/predictions/predictions_classification.gzip")
    logging.info("Prédiction des minutes de retard")
    preds = predict_delay_duration(predictions_retard, "../../models/model_regression.sav")
    preds.to_parquet("../../data/predictions/predictions_regression.gzip", compression='gzip')
    logging.info("Fin")
