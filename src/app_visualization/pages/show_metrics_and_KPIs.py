import os

import pandas as pd
import streamlit as st

from src.app_visualization.data_viz_components.metrics_and_KPIs import get_prediction_with_all_cost_id_df, \
    plot_turnover_of_airlines_and_the_total_to_be_paid, plot_former_turnover_of_airlines_and_the_new_one, \
    plot_breakdown_total_payable_and_turnover


def write():
    st.header("Interprétation métriques et KPIs")
    add_selectbox = st.sidebar.selectbox(
        "Comment souhaitez-vous recevoir les résultats ?",
        ("Email", "SMS")
    )
    if add_selectbox == "SMS":
        phone_number = st.sidebar.text_input("Entrez votre numéro de téléphone")
        if phone_number:
            st.sidebar.markdown(f"Merci un {add_selectbox} vous sera envoyé")
    elif add_selectbox == "Email":
        email = st.sidebar.text_input("Entrez votre email")
        if email:
            st.sidebar.markdown(f"Merci un {add_selectbox} vous sera envoyé")

    st.subheader("Tableau présentant les coûts des retards pour chaque compagnies")
    if st.button("Besoin de recalculer les KPIs (s'il y a eu réentrainement du modèle)"):
        reg_preds = pd.read_parquet("data/predictions/predictions_regression.gzip")
        df_compagnies = pd.read_parquet("data/aggregated_data/compagnies.gzip")
        df_aeroports = pd.read_parquet("data/aggregated_data/aeroports.gzip")
        class_preds = pd.read_parquet("data/predictions/predictions_classification.gzip")
        prediction_with_cost_gb_airline = get_prediction_with_all_cost_id_df(reg_preds, df_compagnies,
                                                                             df_aeroports, class_preds)
        prediction_with_cost_gb_airline.to_csv('src/app_visualization/prediction_with_cost_kpi.csv', index=False)
        st.dataframe(prediction_with_cost_gb_airline)
    if st.button("Afficher les prédictions avec les KPI"):
        prediction_with_cost_gb_airline = pd.read_csv('src/app_visualization/prediction_with_cost_kpi.csv')
        st.dataframe(prediction_with_cost_gb_airline)

    prediction_with_cost_gb_airline = pd.read_csv('src/app_visualization/prediction_with_cost_kpi.csv')
    st.subheader("Interprétation et Hypothèses")
    st.subheader("1. Le prix du retard à payer aux aéroports")
    st.markdown(
        "- après 10min : la compagnie paye toutes les minutes le prix indiqué dans la colonne 'PRIS RETARD POUR CHAQUE MINUTE APRES 10 MINUTES'" \
        "- après 20min : la compagnie paye un supplément qui est le prix indiqué dans la colonne 'PRIX RETARD PREMIERE 20 MINUTES'")

    st.subheader("2. Indemnisation des clients ")
    st.markdown("- 10% des clients vont demander à être indemnisé pour un retard compris entre 10min et 45min"
                "   - Indemnité à payer : 1/4 du prix du billet"
                "- 20% des clients vont demander à être indemnisé pour un retard supérieur à 1h"
                "   - Indemnité à payer : 1/2 du prix du billet"
                "- 50% des clients vont demander à être indemnisé pour un retard supérieur à 3h"
                "   - Indemnité à payer : totalité du prix du billet"
                "On fait l'hypothèse d'un prix maximal fixe du billet : 300€")

    st.subheader("3. Perte de client (taux d'attrition) ")
    st.markdown("On fait l'hypothèse d'un taux d'attrition à 3% pour un retard de plus de 3h"
                "Donc pour connaitre le cout d'une perte de client du aux retard : \n"
                "On suppose qu'un client prend en moyenne 3 fois l'avion par an avec la même compagnie "
                "(on suppose une fidéité total des clients auprès de leur compagnie).\n"
                "Donc Donc si la compagnie perd un client, elle perd un cout de 3x'prix du billet' par client\n"
                "On suppose le prix du billet à 300€ par défaut")
    if st.button("Afficher le total à payer et le chiffre d'affaire par compagnie"):
        plot_turnover_of_airlines_and_the_total_to_be_paid(prediction_with_cost_gb_airline)
    if st.button("Afficher le nouveau chiffre d'affaire et l'ancien chiffre d'affaire par compagnie"):
        plot_former_turnover_of_airlines_and_the_new_one(prediction_with_cost_gb_airline)
    if st.button("Afficher la répartition du total à payer et du chiffre d'affaire restant"):
        plot_breakdown_total_payable_and_turnover(prediction_with_cost_gb_airline)

    if __name__ == "__main__":
        write()
