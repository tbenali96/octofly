import logging
import pickle
import pandas as pd
import streamlit as st


def predict_regressor(X: pd.DataFrame, model_file_name: str) -> pd.DataFrame:
    """
    Predicts the delay duration for each record of the test dataset.
    """
    model = pickle.load(open(model_file_name, 'rb'))
    X = X[X["RETARD"] == 1]
    X = X.drop(columns=["RETARD"])

    cat_features = [0, 1, 6, 11, 13, 14, 15, 16]
    for i in cat_features:
        X.iloc[:, i] = X.iloc[:, i].astype('category')

    predictions = model.predict(X)
    X["RETARD MINUTES"] = predictions
    return X


def main_predict_delay(classif_preds_path, reg_preds_path, model_regression_path):
    predictions_retard = pd.read_parquet(classif_preds_path)
    logging.info("Prédiction des minutes de retard")
    preds = predict_regressor(predictions_retard, model_regression_path)
    preds.to_parquet(reg_preds_path, compression='gzip')
    logging.info("Fin")


def st_main_predict_delay(classif_preds_path, reg_preds_path, model_regression_path):
    predictions_retard = pd.read_parquet(classif_preds_path)
    st.text("Prédiction des minutes de retard")
    preds = predict_regressor(predictions_retard, model_regression_path)
    preds.to_parquet(reg_preds_path, compression='gzip')
    st.text("Fin")


if __name__ == '__main__':
    classif_preds_path = "../../data/predictions/predictions_classification.gzip"
    reg_preds_path = "../../data/predictions/predictions_regression.gzip"
    model_regression_path = "../../models/model_regression.sav"
    main_predict_delay(classif_preds_path, reg_preds_path, model_regression_path)
