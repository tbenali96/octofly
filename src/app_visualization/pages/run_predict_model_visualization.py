import os

import pandas as pd
import streamlit as st

from src.models.predict_classification_model import st_main_prediction
from src.models.predict_regression_model import st_main_predict_delay


def write():
    st.header("Inférence du modèle")
    st.subheader("Sur quelles données faire l'inférence ?")
    flight_path = "data/extracted/test_data/vols.gzip"
    fuel_path = "data/aggregated_data/prix_fuel.gzip"
    scaler_model_path = "models/train_features_scalers"
    model_save_path = "models/model_classification.sav"
    pred_classif_path = "data/predictions/predictions_classification.gzip"
    reg_preds_path = "data/predictions/predictions_regression.gzip"
    model_regression_path = "models/model_regression.sav"
    st.markdown('Les données à inférer sont le Dataframe de vols auquel on ajoute les données du fuels :')
    df_flight = pd.read_parquet(flight_path)
    st.dataframe(df_flight.head(10))
    st.markdown('Donnée correspondant au fuel')
    st.dataframe(pd.read_parquet(fuel_path))
    st.subheader("Inférence !")
    st.markdown("On prédit dans un premier temps si le vol est en retard ou non (avec le modèle de classification"
                "et dans un second temps parmis tous les vols en retard, on prédit le temps de retard en minutes.")
    if st.button("Inférer les données"):
        st_main_prediction(flight_path, fuel_path, scaler_model_path, model_save_path, pred_classif_path)
        st_main_predict_delay(pred_classif_path, reg_preds_path, model_regression_path)
        st.text('Inférence terminée')
        st.markdown('Tableau de prédiction:')
        prediction_df = pd.read_parquet(reg_preds_path)
        #st.dataframe(prediction_df)
        st.write(prediction_df.astype('object').head(50))
        st.text('Allez à la page suivante pour voir les KPIs')


if __name__ == "__main__":
    write()
