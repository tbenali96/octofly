import streamlit as st

from src.app_visualization.data_viz_components.data_viz_and_stats import get_vols_dataframe_with_target_defined, \
    get_scatter_plot_delay_at_arrival_wrt_distance, get_histogram_nbins100_plot_delay_at_arrival_wrt_distance, \
    get_pie_chart_that_display_the_category_delay_distribution, \
    plot_bar_of_number_of_delay_and_on_time_flight_per_airline, \
    get_bar_plot_that_give_the_mean_number_of_delay_given_the_number_of_passagers, \
    plot_bar_chart_with_the_delay_type_cumuluated_by_airline, \
    plot_mapping_size_of_airport_wrt_their_delay_and_nb_of_flight, plot_mean_delay_wrt_hour_of_departure, \
    plot_mean_delay_wrt_month_of_departure, plot_mean_delay_wrt_night_flight, \
    plot_number_of_delay_and_on_time_flight_wrt_the_night_flight, plot_number_of_flights_wrt_hour_of_departure, \
    plot_bar_of_number_of_delay_and_on_time_flight_per_airport_of_departure, \
    plot_bar_of_number_of_delay_and_on_time_flight_per_arrival_airport


def write():
    st.header("PAGE SUR LA VISUALISATION DES DONNÉES ET STATISTIQUES")
    df_vols = get_vols_dataframe_with_target_defined()
    st.subheader("1. Informations relatives aux compagnies aériennes")
    get_pie_chart_that_display_the_category_delay_distribution(df_vols)
    plot_bar_of_number_of_delay_and_on_time_flight_per_airline(df_vols)
    plot_bar_chart_with_the_delay_type_cumuluated_by_airline(df_vols, time_cumulated=True)
    if st.button("Afficher la répartition du NOMBRE de retard par type de retard par compagnie"):
        plot_bar_chart_with_the_delay_type_cumuluated_by_airline(df_vols, time_cumulated=False)
    st.subheader("2. Informations relatives aux Aéroport de départ et d'arrivée")
    st.subheader("2.1 Aéroport de départ:")
    plot_bar_of_number_of_delay_and_on_time_flight_per_airport_of_departure(df_vols)
    st.subheader("2.2 Aéroport d'arrivée:")
    plot_bar_of_number_of_delay_and_on_time_flight_per_arrival_airport(df_vols)
    st.subheader("3. Informations relatives au vols")
    get_scatter_plot_delay_at_arrival_wrt_distance(df_vols)
    if st.button("Afficher le graphique ci-dessus avec la vue en histogramme"):
        get_histogram_nbins100_plot_delay_at_arrival_wrt_distance(df_vols)
    get_bar_plot_that_give_the_mean_number_of_delay_given_the_number_of_passagers(df_vols)
    st.subheader("4. Informations relatives au vols et à la date de départ")
    st.subheader("4.1 Heure de départ")
    plot_number_of_flights_wrt_hour_of_departure(df_vols)
    plot_mean_delay_wrt_hour_of_departure(df_vols)
    plot_number_of_delay_and_on_time_flight_wrt_the_night_flight(df_vols)
    plot_mean_delay_wrt_night_flight(df_vols)
    st.subheader("4.2 Mois")
    plot_mean_delay_wrt_month_of_departure(df_vols)
    st.subheader("5. Cartographie des aeroports (carte intéractive)")
    scale_choice = st.selectbox('Select a values to scale the map', (300, 500, 700, 800, 1000))
    plot_mapping_size_of_airport_wrt_their_delay_and_nb_of_flight(df_vols, scale=scale_choice)


if __name__ == "__main__":
    write()
