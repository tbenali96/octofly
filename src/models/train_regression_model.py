import pickle
import pandas as pd
import lightgbm as lgb


def train_regressor(X, y):
    y_retard = y[y["RETARD"] == 1]
    y_retard = y_retard[["RETARD A L'ARRIVEE"]]
    indexes_retard = y_retard.index
    indexes_non_retard = X.index.difference(indexes_retard)
    X_retard = X.drop(indexes_non_retard)

    X_retard = X_retard.reset_index(drop=True)
    y_retard = y_retard.reset_index(drop=True)

    cat_features = [0, 1, 6, 11, 13, 14, 15, 16]
    for i in cat_features:
        X_retard.iloc[:, i] = X_retard.iloc[:, i].astype('category')

    model = lgb.LGBMRegressor(num_leaves=50, max_depth=-1,
                              random_state=314,
                              silent=True,
                              metric='None',
                              n_jobs=4,
                              n_estimators=2000,
                              colsample_bytree=0.9,
                              subsample=0.9,
                              learning_rate=0.05)
    model.fit(X_retard, y_retard)
    return model


if __name__ == '__main__':
    X = pd.read_parquet("../../data/processed/train_data/train.gzip")
    y = pd.read_parquet("../../data/processed/train_data/train_target.gzip")
    print("Entraînement du modèle de regression")
    model = train_regressor(X, y)
    print("Fin de l'entraînement")
    filename = '../../models/model_regression.sav'
    pickle.dump(model, open(filename, 'wb'))
