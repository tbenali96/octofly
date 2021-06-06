import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def get_ailine_turnover(reg_preds: pd.DataFrame) -> pd.DataFrame:
    df_compagnies = pd.read_parquet("../../../data/aggregated_data/compagnies.gzip")
    reg_preds["CHIFFRE D'AFFAIRE COMPAGNIE"] = reg_preds['COMPAGNIE AERIENNE'].map(
        lambda x: df_compagnies[df_compagnies['CODE'] == x]['CHIFFRE D AFFAIRE'].values[0])
    return reg_preds


def add_cost_20min_delay(df_aeroports: pd.DataFrame, airport: str) -> int:
    twenty_first_min_cost = df_aeroports[
        df_aeroports['CODE IATA'] == airport]['PRIX RETARD PREMIERE 20 MINUTES'].values[0]
    return twenty_first_min_cost


def add_cost_10min_delay(df_aeroports: pd.DataFrame, airport: str) -> int:
    ten_min_delay_cost = df_aeroports[
        df_aeroports['CODE IATA'] == airport]['PRIS RETARD POUR CHAQUE MINUTE APRES 10 MINUTES'].values[0]
    return ten_min_delay_cost


def cost_of_delay(pred_vol: pd.Series) -> float:
    delay = pred_vol['RETARD']
    twenty_first_min_cost = pred_vol['PRIX RETARD PREMIERE 20 MINUTES']
    ten_min_delay_cost = pred_vol['PRIS RETARD CHAQUE MINUTE APRES 10 MINUTES']

    cost = 0
    if delay > 10:
        cost += ten_min_delay_cost * (delay - 10)
    if delay >= 20:
        cost += twenty_first_min_cost
    return cost


def get_first_twenty_min_cost_for_airlines(prediction_avec_retard: pd.DataFrame, df_aeroports: pd.DataFrame):
    prediction_avec_retard['PRIX RETARD PREMIERE 20 MINUTES'] = prediction_avec_retard['AEROPORTS'] \
        .map(lambda x: add_cost_20min_delay(df_aeroports, x))


def get_first_ten_min_cost_for_airlines(prediction_avec_retard: pd.DataFrame, df_aeroports: pd.DataFrame):
    prediction_avec_retard['PRIX RETARD PREMIERE 20 MINUTES'] = prediction_avec_retard['AEROPORTS'] \
        .map(lambda x: add_cost_20min_delay(df_aeroports, x))


def get_airport_delay_cost(prediction_avec_retard: pd.DataFrame) -> pd.DataFrame:
    df_aeroports = pd.read_parquet("../../../data/aggregated_data/aeroports.gzip")
    get_first_twenty_min_cost_for_airlines(prediction_avec_retard, df_aeroports)
    get_first_ten_min_cost_for_airlines(prediction_avec_retard, df_aeroports)
    prediction_avec_retard['COUT DU RETARD'] = prediction_avec_retard.apply(cost_of_delay, axis=1)
    return prediction_avec_retard.drop(
        columns=['PRIX RETARD PREMIERE 20 MINUTES', 'PRIS RETARD CHAQUE MINUTE APRES 10 MINUTES'])


def get_number_of_indemnities_asked(pred_vol: pd.Series):
    delay = pred_vol.loc['RETARD']
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
    delay = pred_vol.loc['RETARD']
    nb_of_indemnities_asked = pred_vol.loc["NOMBRE D'INDEMNITES DEMANDEES"]
    compensation_due_to_clients = 0
    if 10 < delay < 45:
        compensation_due_to_clients = (ticket_price / 3) * nb_of_indemnities_asked
    elif 60 < delay < 180:
        compensation_due_to_clients = (ticket_price / 2) * nb_of_indemnities_asked
    elif delay > 180:
        compensation_due_to_clients = ticket_price * nb_of_indemnities_asked
    return compensation_due_to_clients


def get_number_of_indemnities_asked_and_compensation_due(prediction_avec_retard: pd.DataFrame):
    prediction_avec_retard[
        "NOMBRE D'INDEMNITES DEMANDEES"] = prediction_avec_retard.apply(get_number_of_indemnities_asked, axis=1)
    prediction_avec_retard[
        "INDEMNITES A PAYER"] = prediction_avec_retard.apply(compensation_due, axis=1)


def get_number_of_lost_customer(delay: float, passenger_nb: int) -> int:
    if delay > 180:
        return passenger_nb * 3 // 100
    else:
        return 0


def get_cost_of_lost_customer(nb_of_lost_customers: int, ticket_price: int = 300, flight_frequency: int = 3) -> int:
    return flight_frequency * ticket_price * nb_of_lost_customers


def cost_of_delay_gb_airlines(prediction_avec_retard: pd.DataFrame):
    return prediction_avec_retard[["RETARD", 'COMPAGNIE AERIENNE', "CHIFFRE D'AFFAIRE COMPAGNIE",
                                   "COUT DU RETARD", "INDEMNITES A PAYER",
                                   "NOMBRE DE CLIENTS PERDUS",
                                   "COUT DES CLIENTS PERDUS"]] \
        .groupby(['COMPAGNIE AERIENNE'], as_index=False) \
        .agg({
        "RETARD": "count",
        "CHIFFRE D'AFFAIRE COMPAGNIE": 'first',
        "COUT DU RETARD": 'sum',
        "INDEMNITES A PAYER": 'sum',
        "NOMBRE DE CLIENTS PERDUS": "sum",
        "COUT DES CLIENTS PERDUS": "sum"
    }).rename(columns={"RETARD": "NOMBRE DE RETARD"})


def get_total_to_be_paid(prediction_with_cost_gb_airline: pd.DataFrame):
    prediction_with_cost_gb_airline["TOTAL A PAYER"] = prediction_with_cost_gb_airline["COUT DU RETARD"] \
                                                       + prediction_with_cost_gb_airline["INDEMNITES A PAYER"] \
                                                       + prediction_with_cost_gb_airline["COUT DES CLIENTS PERDUS"]


def get_new_turnover_for_each_airline(prediction_with_cost_gb_airline: pd.DataFrame):
    prediction_with_cost_gb_airline["NV CHIFFRE D'AFFAIRE"] = prediction_with_cost_gb_airline[
                                                                  "CHIFFRE D'AFFAIRE COMPAGNIE"] \
                                                              - prediction_with_cost_gb_airline["TOTAL A PAYER"]


def get_percentage_of_lost_sales(prediction_with_cost_gb_airline: pd.DataFrame):
    prediction_with_cost_gb_airline["%CHIFFRE D'AFFAIRE LOST"] = (prediction_with_cost_gb_airline["TOTAL A PAYER"] /
                                                                  prediction_with_cost_gb_airline[
                                                                      "CHIFFRE D'AFFAIRE COMPAGNIE"]) * 100


def plot_turnover_of_airlines_and_the_total_to_be_paid(prediction_with_cost_gb_airline: pd.DataFrame):
    print(prediction_with_cost_gb_airline)
    fig = px.bar(prediction_with_cost_gb_airline,
                 x="'COMPAGNIE AERIENNE'",
                 y=["CHIFFRE D'AFFAIRE COMPAGNIE", "TOTAL A PAYER"],
                 barmode='group',
                 title="Repartition du Chiffre d'affaire et cout total du retard par Compagnie")
    fig.show()


def plot_former_turnover_of_airlines_and_the_new_one(prediction_with_cost_gb_airline: pd.DataFrame):
    fig = px.bar(prediction_with_cost_gb_airline,
                 x="'COMPAGNIE AERIENNE'",
                 y=["CHIFFRE D'AFFAIRE COMPAGNIE", "NV CHIFFRE D'AFFAIRE"],
                 barmode='group',
                 title="Repartition du Chiffre d'affaire et cout total du retard par Compagnie")
    fig.show()


def get_prediction_with_all_cost_id_df():
    reg_preds = pd.read_parquet("../../../data/predictions/predictions_regression.gzip")
    reg_preds = get_ailine_turnover(reg_preds)
    print(reg_preds.keys())
    prediction_with_delay = reg_preds #reg_preds[reg_preds['RETARD'] > 0].copy()

    # Cost of delay by airports
    get_airport_delay_cost(prediction_with_delay)

    # Compensation to clients
    get_number_of_indemnities_asked_and_compensation_due(prediction_with_delay)

    # Client loss
    prediction_with_delay['NOMBRE DE CLIENTS PERDUS'] = prediction_with_delay.apply(
        lambda x: get_number_of_lost_customer(x["RETARD"], x['NOMBRE DE PASSAGERS']), axis=1)

    # Get the cost of all the lost clients
    prediction_with_delay['COUT DES CLIENTS PERDUS'] = prediction_with_delay["NOMBRE DE CLIENTS PERDUS"].map(
        lambda x: get_cost_of_lost_customer(x))

    # Get total to be paid by airlines
    prediction_with_cost_gb_airline = cost_of_delay_gb_airlines(prediction_with_delay)
    get_total_to_be_paid(prediction_with_cost_gb_airline)
    get_percentage_of_lost_sales(prediction_with_cost_gb_airline)
    return prediction_with_cost_gb_airline


if __name__ == '__main__':
    prediction_with_cost_gb_airline = get_prediction_with_all_cost_id_df()
    plot_turnover_of_airlines_and_the_total_to_be_paid(prediction_with_cost_gb_airline)
    plot_former_turnover_of_airlines_and_the_new_one(prediction_with_cost_gb_airline)
