import streamlit as st

from src.features.feature_engineering import st_main_feature_engineering
from src.models.train_classification_model import st_main_training
from src.models.train_regression_model import st_main_train_regressor_model


def write():
    st.header("Entrainement du modèle")
    delay_parameter = st.text_input("A partir de combien de minutes un vol est considéré en retard")
    flights_path = "data/aggregated_data/vols.gzip"
    fuel_path = "data/aggregated_data/prix_fuel.gzip"
    flights_processed_path = "data/processed/train_data/train.gzip"
    target_path = "data/processed/train_data/train_target.gzip"
    scaler_model_path = "models/train_features_scalers"
    if delay_parameter:
        st.text("Lancement du feature engineering")
        st_main_feature_engineering(flights_path, fuel_path, flights_processed_path, target_path, scaler_model_path,
                                    delay_param=delay_parameter)
    st.subheader("1. Entrainement du classifier : prédit si le vol est en retard")
    X_path = "data/processed/train_data/train.gzip"
    y_path = "data/processed/train_data/train_target.gzip"
    classif_filename = 'models/model_classification.sav'
    if st.button("Entrainer le modèle de classification"):
        st_main_training(X_path, y_path, classif_filename)
    st.subheader("2. Entrainement du modèle de regression : prédit le temps de retard en minutes")
    regressor_filename = 'models/model_regression.sav'
    if st.button("Entrainer le modèle de régression"):
        st_main_train_regressor_model(X_path, y_path, regressor_filename)


if __name__ == "__main__":
    write()
