import pickle
import pandas as pd


def predict(X, model_file_name):
    model = pickle.load(open(model_file_name, 'rb'))
    X = X[X["RETARD"] == 1]
    X = X.drop(columns=["RETARD"])
    predictions = model.predict(X)
    X["RETARD MINUTES"] = predictions
    return X


if __name__ == '__main__':
    predictions_retard = pd.read_parquet("../../data/predictions/predictions_classification.gzip")
    print("Prédiction des minutes de retard")
    preds = predict(predictions_retard, "../../models/model_classification.sav")
    preds.to_parquet("../../data/predictions/predictions_regression.gzip", compression='gzip')
    print("Fin")
