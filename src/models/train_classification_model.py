import pickle
import logging
import pandas as pd
from catboost import CatBoostClassifier
import streamlit as st


def train_classifier(X: pd.DataFrame, y: pd.DataFrame) -> CatBoostClassifier:
    y = y[["RETARD"]]
    cat_features = [0, 1, 6, 11, 13, 14, 15, 16]
    model = CatBoostClassifier(iterations=500,
                               learning_rate=0.05,
                               eval_metric="Recall",
                               depth=10,
                               random_seed=0,
                               auto_class_weights="Balanced")
    model.fit(X, y, cat_features)
    return model


def main_training(X_path, y_path, filename):
    X = pd.read_parquet(X_path)
    y = pd.read_parquet(y_path)
    logging.info("Entraînement du modèle de classification")
    model = train_classifier(X, y)
    logging.info("Fin de l'entraînement")
    pickle.dump(model, open(filename, 'wb'))


def st_main_training(X_path, y_path, filename):
    X = pd.read_parquet(X_path)
    y = pd.read_parquet(y_path)
    st.text("Entraînement du modèle de classification")
    model = train_classifier(X, y)
    st.text("Fin de l'entraînement")
    pickle.dump(model, open(filename, 'wb'))


if __name__ == '__main__':
    X_path = "../../data/processed/train_data/train.gzip"
    y_path = "../../data/processed/train_data/train_target.gzip"
    filename = '../../models/model_classification.sav'
    main_training(X_path, y_path, filename)
