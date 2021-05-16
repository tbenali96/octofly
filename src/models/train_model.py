import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def train(X,y):
    y = y[["RETARD"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    cat_features = [0, 1, 6, 10, 12, 13, 14, 15]
    model = CatBoostClassifier(iterations=200, learning_rate=0.05)
    model.fit(X_train, y_train, cat_features)
    return model, X_test, y_test



if __name__ == '__main__':
    X = pd.read_parquet("../../data/processed/train_data/train.gzip")
    y = pd.read_parquet("../../data/processed/train_data/train_target.gzip")
    model, X_test, y_test = train(X,y)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
