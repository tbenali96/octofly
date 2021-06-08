from unittest.mock import patch

import pandas as pd
import plotly.express as px
from pandas._testing import assert_frame_equal

from src.app_visualization.data_viz_components.data_viz_and_stats import get_vols_dataframe_with_target_defined, \
    get_scatter_plot_delay_at_arrival_wrt_distance, get_histogram_nbins100_plot_delay_at_arrival_wrt_distance, \
    get_pie_chart_that_display_the_category_delay_distribution, get_airport_list_and_their_number_of_flight_sorted, \
    get_index_of_the_arline_sorted_list, create_df_with_number_of_delay_gb_feature, \
    add_the_columns_with_the_airport_index_sorted_list, sort_airport_df_by_number_of_flight, \
    add_total_number_of_flight_on_time_per_airline_column, add_mean_delay_column

TESTED_MODULE = 'src.app_visualization.data_viz_components.data_viz_and_stats'


@patch(f'{TESTED_MODULE}.pd.read_parquet', return_value=pd.DataFrame({"RETARD A L'ARRIVEE": [12, 9]}))
def test_get_vols_dataframe_with_target_defined__add_categorie_retard_equal_1_1_with_delay_12_and_9(m_read_parquet):
    # Given
    expected = pd.DataFrame({"RETARD A L'ARRIVEE": [12, 9],
                             "RETARD": [1, 1],
                             "CATEGORIE RETARD": [1, 1]})
    # When
    actual = get_vols_dataframe_with_target_defined()
    # Then
    assert_frame_equal(actual, expected)


@patch(f'{TESTED_MODULE}.st.plotly_chart')
@patch(f'{TESTED_MODULE}.px.scatter', return_value=px.scatter())
def testget_scatter_plot_delay_at_arrival_wrt_distance__check_that_scatter_is_called(m_scatter, m_st_plot):
    # Given
    df_vols = pd.DataFrame({"RETARD A L'ARRIVEE": [12, 9, -19, 5],
                            'DISTANCE': [23, 12, 90, 15]})
    # When
    get_scatter_plot_delay_at_arrival_wrt_distance(df_vols)
    # Then
    m_scatter.assert_called()


@patch(f'{TESTED_MODULE}.st.plotly_chart')
@patch(f'{TESTED_MODULE}.px.histogram', return_value=px.histogram())
def test_get_histogram_nbins100_plot_delay_at_arrival_wrt_distance__check_histogram_is_called(m_hist, m_st_plot):
    # Given
    df_vols = pd.DataFrame({"RETARD A L'ARRIVEE": [12, 9, 5],
                            'DISTANCE': [23, 12, 15],
                            "CATEGORIE RETARD": [1, 1, 1]})
    # When
    get_histogram_nbins100_plot_delay_at_arrival_wrt_distance(df_vols)
    # Then
    m_hist.assert_called()


@patch(f'{TESTED_MODULE}.st.plotly_chart')
@patch(f'{TESTED_MODULE}.px.pie', return_value=px.pie())
def test_get_pie_chart_that_display_the_category_delay_distribution__check_px_pie_is_called(m_pie, m_st_plot):
    # Given
    df_vols = pd.DataFrame({"RETARD A L'ARRIVEE": [12, 9, 5],
                            'DISTANCE': [23, 12, 15],
                            "CATEGORIE RETARD": [1, 1, 1]})
    # When
    get_pie_chart_that_display_the_category_delay_distribution(df_vols)
    # Then
    m_pie.assert_called()


def test_get_airport_list_and_their_number_of_flight_sorted(get_df_vols):
    # Given
    feature = 'AEROPORT ARRIVEE'
    airport_sorted_expected = ['AAL', 'JNB', 'LTK']
    nb_flight_sorted_expected = [2, 1, 1]
    # When
    airport_index_sorted, nb_flight_sorted = get_airport_list_and_their_number_of_flight_sorted(get_df_vols, feature)
    # Then
    assert len(airport_index_sorted) == len(airport_sorted_expected)
    assert len(nb_flight_sorted) == len(nb_flight_sorted_expected)
    assert airport_sorted_expected[0] == airport_index_sorted[0]


def test_get_index_of_the_arline_sorted_list__give_a_dict_with_the_index_as_value_and_airport_as_key():
    # Given
    airport_sorted_by_nb_of_flights = ['JNB', 'AAL', 'LTK', 'BOB']
    expected_sorterIndex = {'AAL': 1, 'BOB': 3, 'JNB': 0, 'LTK': 2}
    # When
    actual_sorterIndex = get_index_of_the_arline_sorted_list(airport_sorted_by_nb_of_flights)
    # Then
    assert actual_sorterIndex == expected_sorterIndex


def test_create_df_with_number_of_delay_gb_feature__gb_feature_and_add_nb_of_delay_column(get_df_vols_with_delay):
    # Given
    feature = 'AEROPORT ARRIVEE'
    expected = pd.DataFrame({'AEROPORT ARRIVEE': ['AAL', 'JNB', 'LTK'],
                             'RETARD': [2, 0, 1]})
    # When
    actual = create_df_with_number_of_delay_gb_feature(get_df_vols_with_delay, feature)
    assert_frame_equal(actual, expected)
    assert len(actual.RETARD) == len(expected.RETARD)


def test_add_the_columns_with_the_airport_index_sorted_list__give_to_the_each_airport_its_index_from_the_sorted_list_by_nb_of_flight():
    # Given
    feature = 'AEROPORT ARRIVEE'
    number_flight_with_delay = pd.DataFrame({'AEROPORT ARRIVEE': ['BOB', 'AAL', 'LTK', 'JNB'],
                                             'RETARD': [2, 0, 1, 5]})
    sorterIndex = {'AAL': 1, 'BOB': 3, 'JNB': 0, 'LTK': 2}
    expected = pd.DataFrame({'AEROPORT ARRIVEE': ['BOB', 'AAL', 'LTK', 'JNB'],
                             'RETARD': [2, 0, 1, 5],
                             'AEROPORT ARRIVEE SORTED': [3, 1, 2, 0]})
    # When
    add_the_columns_with_the_airport_index_sorted_list(feature, number_flight_with_delay, sorterIndex)
    # Then
    assert_frame_equal(number_flight_with_delay, expected)


def test_sort_airport_df_by_number_of_flight__reorder_the_dataframe_according_to_the_index_of_the_feature_sorted_column():
    # Given
    feature = 'AEROPORT ARRIVEE'
    number_flight_with_delay = pd.DataFrame({'AEROPORT ARRIVEE': ['BOB', 'AAL', 'LTK', 'JNB'],
                                             'RETARD': [2, 0, 1, 5],
                                             'AEROPORT ARRIVEE SORTED': [3, 1, 2, 0]})
    expected = pd.DataFrame({'AEROPORT ARRIVEE': ['JNB', 'AAL', 'LTK', 'BOB'],
                             'RETARD': [5, 0, 1, 2]}, index=[3, 1, 2, 0])
    # When
    actual = sort_airport_df_by_number_of_flight(feature, number_flight_with_delay)
    # Then
    assert_frame_equal(actual, expected)


def test_add_total_number_of_flight_on_time_per_airline_column__give_the_numbre_of_flight_on_time_for_each_airport_of_the_df(
        get_df_vols_with_delay):
    # Given
    expected = pd.DataFrame({'AEROPORT ARRIVEE': ['JNB', 'AAL', 'LTK', 'AAL'],
                             'VOL': [4661, 5026, 2021, 3809],
                             'NOMBRE DE PASSAGERS': [10, 40, 30, 490],
                             'DISTANCE	': [187, 234, 2288, 835],
                             'RETARD': [0, 1, 1, 1],
                             'NOMBRE VOLS TOTAL': [34, 12, 8, 10],
                             "VOL A l'HEURE": [34, 11, 7, 9]})
    # When
    add_total_number_of_flight_on_time_per_airline_column(get_df_vols_with_delay)
    # Then
    assert_frame_equal(get_df_vols_with_delay, expected)


def test_add_mean_delay_column__compute_the_ratio_between_delay_and_nb_of_flight(get_df_vols_with_delay):
    # Given
    expected = pd.DataFrame({'AEROPORT ARRIVEE': ['JNB', 'AAL', 'LTK', 'AAL'],
                             'VOL': [4661, 5026, 2021, 3809],
                             'NOMBRE DE PASSAGERS': [10, 40, 30, 490],
                             'DISTANCE	': [187, 234, 2288, 835],
                             'RETARD': [0, 1, 1, 1],
                             'NOMBRE VOLS TOTAL': [34, 12, 8, 10],
                             "RETARD MOYEN": [0, 1 / 12, 1 / 8, 1 / 10]})
    # When
    actual = add_mean_delay_column(get_df_vols_with_delay)
    # Then
    assert_frame_equal(get_df_vols_with_delay, expected)
