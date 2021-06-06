import pickle
import logging
import pandas as pd
from catboost import CatBoostClassifier


def train_classifier(X, y):
    y = y[["RETARD"]]
    cat_features = [0, 1, 6, 11, 13, 14, 15, 16]
    model = CatBoostClassifier(iterations=500,
                               learning_rate=0.03,
                               eval_metric="Recall",
                               depth=10,
                               random_seed=0,
                               auto_class_weights="Balanced")
    model.fit(X, y, cat_features)
    return model


if __name__ == '__main__':
    X = pd.read_parquet("../../data/processed/train_data/train.gzip")
    y = pd.read_parquet("../../data/processed/train_data/train_target.gzip")
    logging.info("Entraînement du modèle de classification")
    model = train_classifier(X, y)
    logging.info("Fin de l'entraînement")
    filename = '../../models/model_classification.sav'
    pickle.dump(model, open(filename, 'wb'))
