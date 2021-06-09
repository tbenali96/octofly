import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytest
import streamlit as st


def get_airline_turnover(reg_preds: pd.DataFrame, df_compagnies: pd.DataFrame) -> pd.DataFrame:
    new_prediction_avec_retard = pd.merge(reg_preds,
                                          df_compagnies[['CODE', 'CHIFFRE D AFFAIRE']].rename(
                                              columns={'CODE': 'COMPAGNIE AERIENNE'}),
                                          on='COMPAGNIE AERIENNE', how='left')
    return new_prediction_avec_retard


def add_cost_20min_delay(df_aeroports: pd.DataFrame, prediction_avec_retard: pd.DataFrame) -> pd.DataFrame:
    new_prediction_avec_retard = pd.merge(prediction_avec_retard,
                                          df_aeroports[['CODE IATA', 'PRIX RETARD PREMIERE 20 MINUTES']].rename(
                                              columns={'CODE IATA': 'AEROPORT ARRIVEE'}),
                                          on='AEROPORT ARRIVEE', how='left')
    return new_prediction_avec_retard


def add_cost_10min_delay(df_aeroports: pd.DataFrame, prediction_avec_retard: pd.DataFrame) -> pd.DataFrame:
    new_prediction_avec_retard = pd.merge(prediction_avec_retard,
                                          df_aeroports[
                                              ['CODE IATA', 'PRIS RETARD POUR CHAQUE MINUTE APRES 10 MINUTES']].rename(
                                              columns={'CODE IATA': 'AEROPORT ARRIVEE'}),
                                          on='AEROPORT ARRIVEE', how='left')
    return new_prediction_avec_retard


def cost_of_delay(pred_vol: pd.Series) -> float:
    delay = pred_vol['RETARD MINUTES']
    twenty_first_min_cost = pred_vol['PRIX RETARD PREMIERE 20 MINUTES']
    ten_min_delay_cost = pred_vol['PRIS RETARD POUR CHAQUE MINUTE APRES 10 MINUTES']

    cost = 0
    if delay > 10:
        cost += ten_min_delay_cost * (delay - 10)
    if delay >= 20:
        cost += twenty_first_min_cost
    return cost


def get_airport_delay_cost(prediction_avec_retard: pd.DataFrame, df_aeroports: pd.DataFrame) -> pd.DataFrame:
    prediction_avec_retard = add_cost_10min_delay(df_aeroports, prediction_avec_retard)
    prediction_avec_retard = add_cost_20min_delay(df_aeroports, prediction_avec_retard)
    prediction_avec_retard['COUT DU RETARD'] = prediction_avec_retard.apply(cost_of_delay, axis=1)
    return prediction_avec_retard.drop(
        columns=['PRIX RETARD PREMIERE 20 MINUTES', 'PRIS RETARD POUR CHAQUE MINUTE APRES 10 MINUTES'])


def get_number_of_indemnities_asked(pred_vol: pd.Series):
    delay = pred_vol.loc['RETARD MINUTES']
    nb_of_passenger = pred_vol.loc['NOMBRE DE PASSAGERS']
    nb_of_indemnities_asked = 0
    if 10 < delay < 45:
        nb_of_indemnities_asked = 20 * nb_of_passenger // 100
    elif 60 < delay < 180:
        nb_of_indemnities_asked = 50 * nb_of_passenger // 100
    elif delay > 180:
        nb_of_indemnities_asked = 75 * nb_of_passenger // 100
    return nb_of_indemnities_asked


def compensation_due(pred_vol, ticket_price=300):
    delay = pred_vol.loc['RETARD MINUTES']
    nb_of_indemnities_asked = pred_vol.loc["NOMBRE D'INDEMNITES DEMANDEES"]
    compensation_due_to_clients = 0
    if 10 < delay < 45:
        compensation_due_to_clients = (ticket_price / 3) * nb_of_indemnities_asked
    elif 60 < delay < 180:
        compensation_due_to_clients = (ticket_price / 2) * nb_of_indemnities_asked
    elif delay > 180:
        compensation_due_to_clients = ticket_price * nb_of_indemnities_asked
    return compensation_due_to_clients


def get_number_of_indemnities_asked_and_compensation_due(prediction_with_delay: pd.DataFrame):
    prediction_with_delay[
        "NOMBRE D'INDEMNITES DEMANDEES"] = prediction_with_delay.apply(get_number_of_indemnities_asked, axis=1)
    prediction_with_delay[
        "INDEMNITES A PAYER"] = prediction_with_delay.apply(compensation_due, axis=1)
    return prediction_with_delay


def get_number_of_lost_customer(delay: float, passenger_nb: int) -> int:
    if delay > 180:
        return passenger_nb * 3 // 100
    else:
        return 0


def get_cost_of_lost_customer(nb_of_lost_customers: int, ticket_price: int = 300, flight_frequency: int = 3) -> int:
    return flight_frequency * ticket_price * nb_of_lost_customers


def cost_of_delay_gb_airlines(prediction_with_delay: pd.DataFrame):
    return prediction_with_delay[["RETARD MINUTES", 'COMPAGNIE AERIENNE', "CHIFFRE D AFFAIRE",
                                  "COUT DU RETARD", "INDEMNITES A PAYER",
                                  "NOMBRE DE CLIENTS PERDUS",
                                  "COUT DES CLIENTS PERDUS"]] \
        .groupby(['COMPAGNIE AERIENNE'], as_index=False) \
        .agg({
        "RETARD MINUTES": "count",
        "CHIFFRE D AFFAIRE": 'first',
        "COUT DU RETARD": 'sum',
        "INDEMNITES A PAYER": 'sum',
        "NOMBRE DE CLIENTS PERDUS": "sum",
        "COUT DES CLIENTS PERDUS": "sum"
    }).rename(columns={"RETARD MINUTES": "NOMBRE DE RETARD"})


def get_total_to_be_paid(prediction_with_cost_gb_airline: pd.DataFrame) -> pd.DataFrame:
    prediction_with_cost_gb_airline["TOTAL A PAYER"] = prediction_with_cost_gb_airline["COUT DU RETARD"] \
                                                       + prediction_with_cost_gb_airline["INDEMNITES A PAYER"] \
                                                       + prediction_with_cost_gb_airline["COUT DES CLIENTS PERDUS"]
    return prediction_with_cost_gb_airline


def get_new_turnover_for_each_airline(prediction_with_cost_gb_airline: pd.DataFrame) -> pd.DataFrame:
    prediction_with_cost_gb_airline["NV CHIFFRE D'AFFAIRE"] = prediction_with_cost_gb_airline[
                                                                  "CHIFFRE D AFFAIRE"] \
                                                              - prediction_with_cost_gb_airline["TOTAL A PAYER"]
    return prediction_with_cost_gb_airline


def get_percentage_of_lost_sales(prediction_with_cost_gb_airline: pd.DataFrame) -> pd.DataFrame:
    prediction_with_cost_gb_airline["%CHIFFRE D'AFFAIRE LOST"] = (prediction_with_cost_gb_airline["TOTAL A PAYER"] /
                                                                  prediction_with_cost_gb_airline[
                                                                      "CHIFFRE D AFFAIRE"]) * 100
    return prediction_with_cost_gb_airline


def get_percentage_of_delay_by_company(prediction_with_cost_gb_airline: pd.DataFrame,
                                       class_preds: pd.DataFrame) -> pd.DataFrame:
    class_preds_with_delay = class_preds[['COMPAGNIE AERIENNE', 'RETARD']].rename(columns={'RETARD': 'NB DE RETARD'})
    class_preds_with_delay['NB DE VOLS'] = 1
    flight_gb_airlines = class_preds_with_delay.groupby(['COMPAGNIE AERIENNE'], as_index=False).sum()
    flight_gb_airlines['POURCENTAGE DE RETARD'] = (flight_gb_airlines['NB DE RETARD'] / flight_gb_airlines[
        'NB DE VOLS']) * 100
    prediction_with_cost_gb_airline_with_percentage_of_flight = pd.merge(prediction_with_cost_gb_airline,
                                                                         flight_gb_airlines, on='COMPAGNIE AERIENNE',
                                                                         how='left')
    return prediction_with_cost_gb_airline_with_percentage_of_flight


@pytest.mark.skip(reason="no need to test this function")
def plot_turnover_of_airlines_and_the_total_to_be_paid(
        prediction_with_cost_gb_airline: pd.DataFrame):  # pragma: no cover
    fig = px.bar(prediction_with_cost_gb_airline,
                 x="COMPAGNIE AERIENNE",
                 y=["CHIFFRE D AFFAIRE", "TOTAL A PAYER"],
                 barmode='group',
                 title="Repartition du Chiffre d'affaire et cout total du retard par Compagnie")
    st.plotly_chart(fig)


@pytest.mark.skip(reason="no need to test this function")
def plot_turnover_of_airlines_and_the_different_cost_to_be_paid(
        prediction_with_cost_gb_airline: pd.DataFrame):  # pragma: no cover
    prediction_with_cost_gb_airline = prediction_with_cost_gb_airline.rename(
        columns={"INDEMNITES A PAYER": "INDEMNITES A PAYER AUX CLIENTS",
                 "COUT DU RETARD": "COUT DU RETARD (AEROPORTS)"})
    fig = px.bar(prediction_with_cost_gb_airline,
                 x="COMPAGNIE AERIENNE",
                 y=["CHIFFRE D AFFAIRE", "TOTAL A PAYER", "INDEMNITES A PAYER AUX CLIENTS",
                    "COUT DU RETARD (AEROPORTS)",
                    "COUT DES CLIENTS PERDUS"],
                 barmode='group',
                 title="Repartition du Chiffre d'affaire et cout total du retard par Compagnie")
    st.plotly_chart(fig)


@pytest.mark.skip(reason="no need to test this function")
def plot_former_turnover_of_airlines_and_the_new_one(prediction_with_cost_gb_airline: pd.DataFrame):  # pragma: no cover
    fig = px.bar(prediction_with_cost_gb_airline,
                 x="COMPAGNIE AERIENNE",
                 y=["CHIFFRE D AFFAIRE", "NV CHIFFRE D'AFFAIRE"],
                 barmode='group',
                 title="Repartition du Chiffre d'affaire et cout total du retard par Compagnie")
    st.plotly_chart(fig)


@pytest.mark.skip(reason="no need to test this function")
def plot_breakdown_total_payable_and_turnover(prediction_with_cost_gb_airline):  # pragma: no cover
    for idx, company in enumerate(prediction_with_cost_gb_airline["COMPAGNIE AERIENNE"]):
        labels = ["NV CHIFFRE D'AFFAIRE", "TOTAL A PAYER"]
        values = [prediction_with_cost_gb_airline.iloc[idx]["NV CHIFFRE D'AFFAIRE"],
                  prediction_with_cost_gb_airline.iloc[idx]["TOTAL A PAYER"]]

        fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0, 0.2])])
        fig.update_layout(title_text=company)
        st.plotly_chart(fig)


@pytest.mark.skip(reason="no need to test this function")
def plot_breakdown_all_different_payables_and_turnover(prediction_with_cost_gb_airline):  # pragma: no cover
    for idx, company in enumerate(prediction_with_cost_gb_airline["COMPAGNIE AERIENNE"]):
        labels = ["NV CHIFFRE D'AFFAIRE", "INDEMNITES A PAYER AUX CLIENTS", "COUT DES CLIENTS PERDUS",
                  "COUT DU RETARD (AEROPORTS)"
                  ]
        values = [prediction_with_cost_gb_airline.iloc[idx]["NV CHIFFRE D'AFFAIRE"],
                  prediction_with_cost_gb_airline.iloc[idx]["INDEMNITES A PAYER"],
                  prediction_with_cost_gb_airline.iloc[idx]["COUT DES CLIENTS PERDUS"],
                  prediction_with_cost_gb_airline.iloc[idx]["COUT DU RETARD"]]

        fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0, 0.2, 0.2, 0.2])])
        fig.update_layout(title_text=company)
        st.plotly_chart(fig)


def get_prediction_with_all_cost_id_df(reg_preds, df_compagnies, df_aeroports, class_preds):
    prediction_with_delay = get_airline_turnover(reg_preds, df_compagnies)
    # Cost of delay by airports
    prediction_with_delay = get_airport_delay_cost(prediction_with_delay, df_aeroports)

    # Compensation to clients
    prediction_with_delay = get_number_of_indemnities_asked_and_compensation_due(prediction_with_delay)

    # Client loss
    prediction_with_delay['NOMBRE DE CLIENTS PERDUS'] = prediction_with_delay.apply(
        lambda x: get_number_of_lost_customer(x["RETARD MINUTES"], x['NOMBRE DE PASSAGERS']), axis=1)

    # Get the cost of all the lost clients
    prediction_with_delay['COUT DES CLIENTS PERDUS'] = prediction_with_delay["NOMBRE DE CLIENTS PERDUS"].map(
        lambda x: get_cost_of_lost_customer(x))

    # Get total to be paid by airlines
    prediction_with_cost_gb_airline = cost_of_delay_gb_airlines(prediction_with_delay)
    prediction_with_cost_gb_airline = get_total_to_be_paid(prediction_with_cost_gb_airline)
    prediction_with_cost_gb_airline = get_new_turnover_for_each_airline(prediction_with_cost_gb_airline)
    prediction_with_cost_gb_airline = get_percentage_of_lost_sales(prediction_with_cost_gb_airline)
    prediction_with_cost_gb_airline = get_percentage_of_delay_by_company(prediction_with_cost_gb_airline, class_preds)
    return prediction_with_cost_gb_airline


if __name__ == '__main__':  # pragma: no cover
    reg_preds = pd.read_csv("../../../data/predictions/predictions_regression_with_turnover.csv")
    df_compagnies = pd.read_parquet("../../../data/aggregated_data/compagnies.gzip")
    df_aeroports = pd.read_parquet("../../../data/aggregated_data/aeroports.gzip")
    class_preds = pd.read_parquet("../../../data/predictions/predictions_classification.gzip")
    prediction_with_cost_gb_airline = get_prediction_with_all_cost_id_df(reg_preds, df_compagnies,
                                                                         df_aeroports, class_preds)
    prediction_with_cost_gb_airline.to_csv('../prediction_with_cost_kpi.csv', index=False)
    logging.Logger('Dataframe with KPIs is created and saved')
