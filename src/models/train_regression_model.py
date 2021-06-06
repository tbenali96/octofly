import logging
import pickle
import pandas as pd
import lightgbm as lgb
import streamlit as st


def train_regressor(X: pd.DataFrame, y: pd.DataFrame) -> lgb.LGBMRegressor:
    y_late = y[y["RETARD"] == 1]
    y_late = y_late[["RETARD A L'ARRIVEE"]]
    indexes_retard = y_late.index
    indexes_non_retard = X.index.difference(indexes_retard)
    X_late = X.drop(indexes_non_retard)

    X_late = X_late.reset_index(drop=True)
    y_late = y_late.reset_index(drop=True)

    cat_features = [0, 1, 6, 11, 13, 14, 15, 16]
    for i in cat_features:
        X_late.iloc[:, i] = X_late.iloc[:, i].astype('category')

    model = lgb.LGBMRegressor(num_leaves=50, max_depth=-1,
                              random_state=314,
                              silent=True,
                              metric='None',
                              n_jobs=4,
                              n_estimators=2000,
                              colsample_bytree=0.9,
                              subsample=0.9,
                              learning_rate=0.05)
    model.fit(X_late, y_late)
    return model


def main_train_regressor_model(X_path, y_path, filename):
    X = pd.read_parquet(X_path)
    y = pd.read_parquet(y_path)
    logging.info("Entraînement du modèle de regression")
    model = train_regressor(X, y)
    logging.info("Fin de l'entraînement")

    pickle.dump(model, open(filename, 'wb'))


def st_main_train_regressor_model(X_path, y_path, filename):
    X = pd.read_parquet(X_path)
    y = pd.read_parquet(y_path)
    st.text("Entraînement du modèle de regression")
    model = train_regressor(X, y)
    st.text("Fin de l'entraînement")

    pickle.dump(model, open(filename, 'wb'))


if __name__ == '__main__':
    X_path = "../../data/processed/train_data/train.gzip"
    y_path = "../../data/processed/train_data/train_target.gzip"
    filename = '../../models/model_regression.sav'
    main_train_regressor_model(X_path, y_path, filename)
