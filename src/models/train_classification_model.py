import pickle
import pandas as pd
from catboost import CatBoostClassifier


def train(X, y):
    y = y[["RETARD"]]
    cat_features = [0, 1, 6, 10, 12, 13, 14, 15]
    model = CatBoostClassifier(iterations=150,
                               learning_rate=0.1,
                               eval_metric="Recall",
                               random_seed=0,
                               auto_class_weights="Balanced")
    model.fit(X, y, cat_features)
    return model


if __name__ == '__main__':
    X = pd.read_parquet("../../data/processed/train_data/train.gzip")
    y = pd.read_parquet("../../data/processed/train_data/train_target.gzip")
    model = train(X, y)
    filename = '../../models/model_prediction.sav'
    pickle.dump(model, open(filename, 'wb'))
