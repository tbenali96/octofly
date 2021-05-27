import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
from config import DATA_PATH
from src.features.feature_engineering import add_delay_binary_target, add_categorical_delay_target, \
    get_category_delay_target_in_string, convert_time_into_datetime


def get_vols_dataframe_with_target_defined():
    df_vols = pd.read_parquet(DATA_PATH + "/parquet_format/train_data/vols.gzip")
    add_delay_binary_target(df_vols)
    df_vols["CATEGORIE RETARD"] = df_vols["RETARD A L'ARRIVEE"].apply(lambda x: add_categorical_delay_target(x))
    return df_vols


def get_scatter_plot_delay_at_arrival_wrt_distance(df_vols):
    df_vols_avec_retard = df_vols[df_vols["RETARD A L'ARRIVEE"] > 0].reset_index(drop=True)
    fig = px.scatter(df_vols_avec_retard, x="DISTANCE", y="RETARD A L'ARRIVEE",
                     title="Répartition des vols en retard par rapport à la distance parcourue")
    st.plotly_chart(fig)


def get_histogram_nbins100_plot_delay_at_arrival_wrt_distance(df_vols):
    df_vols_avec_retard = df_vols[df_vols["RETARD A L'ARRIVEE"] > 0].reset_index(drop=True)
    df_vols_avec_retard['Category retard str'] = df_vols_avec_retard['CATEGORIE RETARD'].map(
        lambda x: "delay <= 3h" if x == 1 else "delay > 3h")
    fig = px.histogram(df_vols_avec_retard, x="DISTANCE", nbins=100, color="Category retard str",
                       labels={"CATEGORIE RETARD": "CATEGORY DELAY"},
                       title="Répartition des vols en retard par rapport à la distance parcourue (vue avec histogramme)")
    st.plotly_chart(fig)
    st.markdown(
        "Nous pouvons remarquer sur ce graphe que la plupart des vols en retard concernent les vols à faibles distances")


def get_pie_chart_that_display_the_category_delay_distribution(df_vols):
    df_vols_avec_retard_wth_count = df_vols[['CATEGORIE RETARD']]
    df_vols_avec_retard_wth_count['count'] = 1
    df_vols_avec_retard_wth_count_gb = df_vols_avec_retard_wth_count.groupby(['CATEGORIE RETARD'], as_index=False).sum()
    df_vols_avec_retard_wth_count_gb['Category retard str'] = df_vols_avec_retard_wth_count_gb['CATEGORIE RETARD'].map(
        lambda x: get_category_delay_target_in_string(x))
    fig = px.pie(df_vols_avec_retard_wth_count_gb, values='count', names='Category retard str',
                 title='Distribution of the category delay')
    st.plotly_chart(fig)


def create_dataframe_to_plot_the_number_of_delay_sorted_per_nb_of_flight_per_airline(
        df_vols: pd.DataFrame) -> pd.DataFrame:
    # Get airline list and their number of flight sorted
    airline_sorted_by_nb_of_flights = list(df_vols["COMPAGNIE AERIENNE"].value_counts().index)
    number_flights_sorted = list(df_vols["COMPAGNIE AERIENNE"].value_counts().values)

    # Create index to sort wrt the airline sorted list
    sorterIndex = dict(zip(airline_sorted_by_nb_of_flights, range(len(airline_sorted_by_nb_of_flights))))

    # count number of delay flight per airline
    number_flight_with_delay = df_vols[['RETARD', "COMPAGNIE AERIENNE"]] \
        .groupby(["COMPAGNIE AERIENNE"], as_index=False) \
        .sum()
    number_flight_with_delay['COMPAGNIE AERIENNE SORTED'] = number_flight_with_delay["COMPAGNIE AERIENNE"] \
        .map(sorterIndex)

    # sort airline by number of flight by airline
    number_flight_with_delay_sorted = number_flight_with_delay \
        .sort_values(by='COMPAGNIE AERIENNE SORTED', ascending=True) \
        .drop(['COMPAGNIE AERIENNE SORTED'], axis=1)
    # add total number of flight per airline
    number_flight_with_delay_sorted['NOMBRE VOLS TOTAL'] = number_flights_sorted

    # add total number of flight on time per airline
    number_flight_with_delay_sorted["VOL A l'HEURE"] = number_flight_with_delay_sorted \
        .apply(lambda x: x['NOMBRE VOLS TOTAL'] - x['RETARD'], axis=1)
    # Get mean delay per airline
    number_flight_with_delay_sorted['RETARD MOYEN'] = number_flight_with_delay_sorted \
        .apply(lambda x: x['RETARD'] / x['NOMBRE VOLS TOTAL'],
               axis=1)
    return number_flight_with_delay_sorted


def plot_bar_of_number_of_delay_and_on_time_flight_per_airline(df_vols: pd.DataFrame):
    df_processed = create_dataframe_to_plot_the_number_of_delay_sorted_per_nb_of_flight_per_airline(df_vols)
    fig1 = px.bar(df_processed, x='COMPAGNIE AERIENNE', y=['RETARD', "VOL A l'HEURE"],
                  title="Repartition des vols à l'heure et des vols en retard par compagnie")

    fig2 = px.bar(df_processed,
                  y='RETARD MOYEN',
                  x='COMPAGNIE AERIENNE',
                  title='Nombre moyen de vols en retard par Compagnie Aérienne (trié par nb de vol par compagnie)')
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)
    st.markdown(
        "Les Compagnies qui ont le plus de vols en retard sont celles qui ont le moins de vols au total")


def get_bar_plot_that_give_the_mean_number_of_delay_given_the_number_of_passagers(df_vols: pd.DataFrame):
    df_delay_wrt_nb_of_passengers = df_vols[['NOMBRE DE PASSAGERS', "RETARD"]].copy()

    df_delay_wrt_nb_of_passengers['Nombre de vols'] = 1
    df_delay_wrt_nb_of_passengers = df_delay_wrt_nb_of_passengers[
        df_delay_wrt_nb_of_passengers['NOMBRE DE PASSAGERS'] > 0].sort_values(by='NOMBRE DE PASSAGERS', ascending=True)

    df_delay_wrt_nb_of_passengers_gb = df_delay_wrt_nb_of_passengers.groupby(['NOMBRE DE PASSAGERS'],
                                                                             as_index=False).sum().rename(
        columns={'RETARD': 'NB RETARD'})
    df_delay_wrt_nb_of_passengers_gb['RETARD MOYEN'] = df_delay_wrt_nb_of_passengers_gb.apply(
        lambda x: x['NB RETARD'] / x['Nombre de vols'], axis=1)

    fig = px.bar(df_delay_wrt_nb_of_passengers_gb,
                 y='RETARD MOYEN',
                 x='NOMBRE DE PASSAGERS',
                 title='Retard moyen par nombre de passagers')
    st.plotly_chart(fig)
    st.markdown(
        "Il semblerait que le nombre important de passagers dans un vol (2500) influence peu le temps de vol de l'avion"
        "(30% de chance que le vol soit en retard),"
        "On remarque cependant que les vols avec peu de passagers (<500) ont plus de chance d'être en retard (plus de 40%)."
        "On pourrait faire l'hypothèse que le nombre de passager est lié à la distance parcourue: "
        "plus la distance est petite moins il y a de passagers."
        "Or nous avons vu dans le premier graphe que plus la distance parcourue est courte plus il a de chance que le vol soit en retard.")


def get_bar_chart_with_the_delay_type_cumuluated_by_airline(df_vols: pd.DataFrame, time_cumulated: bool = True):
    delay_type = ['RETARD SYSTEM', 'RETARD SECURITE',
                  'RETARD COMPAGNIE', 'RETARD AVION', 'RETARD METEO']
    bool_delay_type = ['is_' + delay for delay in delay_type]
    delays_type_by_airline = df_vols[['COMPAGNIE AERIENNE', 'RETARD SYSTEM', 'RETARD SECURITE',
                                      'RETARD COMPAGNIE', 'RETARD AVION', 'RETARD METEO']].fillna(0)
    for delay in delay_type:
        delays_type_by_airline['is_' + delay] = delays_type_by_airline[delay].map(lambda x: True if x > 0 else False)
    delays_type_by_airline_gb = delays_type_by_airline.groupby(["COMPAGNIE AERIENNE"], as_index=False).sum()

    if time_cumulated:
        fig = px.bar(delays_type_by_airline_gb, x="COMPAGNIE AERIENNE", y=delay_type,
                     title="Repartition du tps de retards par compagnie")
        st.plotly_chart(fig)
    else:
        fig = px.bar(delays_type_by_airline_gb,
                     x="COMPAGNIE AERIENNE",
                     y=bool_delay_type,
                     title="Repartition du nombre de retard par type de retard par compagnie")
        st.plotly_chart(fig)


def plot_mean_delay_wrt_hour_of_departure(df_vols: pd.DataFrame):
    df_vols['DEPART PROGRAMME'] = df_vols['DEPART PROGRAMME'].apply(convert_time_into_datetime)
    df_vols["HEURE DE DEPART"] = df_vols['DEPART PROGRAMME'].map(lambda x: str(x)[:2])
    df_vols_delay_wrt_hour = df_vols[["HEURE DE DEPART", "RETARD"]].copy()
    df_vols_delay_wrt_hour['Nombre de vols'] = 1

    df_vols_delay_wrt_hour_gb = df_vols_delay_wrt_hour.groupby(["HEURE DE DEPART"], as_index=False).sum()
    df_vols_delay_wrt_hour_gb['RETARD MOYEN'] = df_vols_delay_wrt_hour_gb.apply(
        lambda x: x['RETARD'] / x['Nombre de vols'], axis=1)
    fig = px.bar(df_vols_delay_wrt_hour_gb,
                 y='RETARD MOYEN',
                 x="HEURE DE DEPART",
                 title="Retard moyen par rapport à l'heure de départ")
    st.plotly_chart(fig)
    st.markdown("On remarque une légère diminition du retard moyen entre 00h et 10h du matin. "
                "On aurait moins de chance d'avoir du retard si le départ était programmé dans cette tranche horraire")


def get_df_to_plot_the_airport_mapping_wrt_delay(df_vols: pd.DataFrame) -> pd.DataFrame:
    df_aeroports = pd.read_parquet(DATA_PATH + "/parquet_format/train_data/aeroports.gzip")
    delays_by_airport = df_vols[['AEROPORT ARRIVEE', "RETARD A L'ARRIVEE"]]
    delays_by_airport['NB VOLS'] = 1

    df_aeroports_and_country = df_aeroports[['CODE IATA', 'LONGITUDE', 'LATITUDE', 'LIEU']] \
        .rename(columns={'CODE IATA': 'AEROPORT ARRIVEE'})
    delays_by_airport_iso_alpha = pd.merge(delays_by_airport, df_aeroports_and_country, on='AEROPORT ARRIVEE')

    delays_by_airport_iso_alpha = delays_by_airport_iso_alpha.fillna(0)

    delays_by_airport_iso_alpha["NB RETARD A L'ARRIVEE"] = delays_by_airport_iso_alpha["RETARD A L'ARRIVEE"] \
        .map(lambda x: True if x > 0 else False)

    delays_by_airport_iso_alpha_gb = delays_by_airport_iso_alpha.groupby(['AEROPORT ARRIVEE'], as_index=False) \
        .agg({"NB RETARD A L'ARRIVEE": 'sum',
              'NB VOLS': 'sum',
              'LATITUDE': 'first',
              'LONGITUDE': 'first',
              'LIEU': 'first'})
    delays_by_airport_iso_alpha_gb['NB RETARD MOYEN'] = delays_by_airport_iso_alpha_gb.apply(
        lambda x: x["NB RETARD A L'ARRIVEE"] / x['NB VOLS'], axis=1)
    return delays_by_airport_iso_alpha_gb


def plot_mapping_size_of_airport_wrt_their_delay_and_nb_of_flight(df_vols: pd.DataFrame, scale: int = 300):
    scale = scale
    delays_by_airport_iso_alpha_gb = get_df_to_plot_the_airport_mapping_wrt_delay(df_vols)
    delays_by_airport_iso_alpha_gb['text_nb_retard'] = delays_by_airport_iso_alpha_gb.apply(
        lambda x: 'Aeroport : ' + x['LIEU'] + '| Nombre de retard : ' + str(x["NB RETARD A L'ARRIVEE"])
        , axis=1)
    delays_by_airport_iso_alpha_gb['text_nb_vol'] = delays_by_airport_iso_alpha_gb.apply(
        lambda x: 'Aeroport : ' + x['LIEU'] + '| Nombre de vols: ' + str(x['NB VOLS'])
        , axis=1)
    fig = go.Figure()

    fig.add_trace(go.Scattergeo(
        locationmode='USA-states',
        lon=delays_by_airport_iso_alpha_gb['LONGITUDE'],
        lat=delays_by_airport_iso_alpha_gb['LATITUDE'],
        text=delays_by_airport_iso_alpha_gb['text_nb_vol'],
        marker=dict(
            size=delays_by_airport_iso_alpha_gb['NB VOLS'] / scale,
            line_color='rgb(0,0,255)',
            line_width=0.5,
            sizemode='area'
        ),
        name='NB VOLS'))

    fig.add_trace(go.Scattergeo(
        locationmode='USA-states',
        lon=delays_by_airport_iso_alpha_gb['LONGITUDE'],
        lat=delays_by_airport_iso_alpha_gb['LATITUDE'],
        text=delays_by_airport_iso_alpha_gb['text_nb_retard'],
        marker=dict(
            size=delays_by_airport_iso_alpha_gb["NB RETARD A L'ARRIVEE"] / scale,
            line_color='rgb(0,0,255)',
            line_width=0.5,
            sizemode='area'
        ),
        name="NB DE RETARD A L'ARRIVEE"))
    fig.update_layout(
        title='Cartographie des aeroports en fonction de leur nombre de retard et nombre de vols',
    )

    st.plotly_chart(fig)


def main():
    df_vols = get_vols_dataframe_with_target_defined()
    get_scatter_plot_delay_at_arrival_wrt_distance(df_vols)
    get_histogram_nbins100_plot_delay_at_arrival_wrt_distance(df_vols)
    get_pie_chart_that_display_the_category_delay_distribution(df_vols)
    plot_bar_of_number_of_delay_and_on_time_flight_per_airline(df_vols)


if __name__ == '__main__':
    main()
