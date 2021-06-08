from unittest.mock import patch

import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

from src.app_visualization.data_viz_components.metrics_and_KPIs import get_airline_turnover, add_cost_20min_delay, \
    add_cost_10min_delay, cost_of_delay, get_airport_delay_cost, get_number_of_indemnities_asked, compensation_due, \
    get_number_of_indemnities_asked_and_compensation_due, get_cost_of_lost_customer, cost_of_delay_gb_airlines, \
    get_total_to_be_paid, get_new_turnover_for_each_airline, get_percentage_of_lost_sales, \
    get_percentage_of_delay_by_company, get_prediction_with_all_cost_id_df

TESTED_MODULE = 'src.app_visualization.data_viz_components.metrics_and_KPIs'


def test_get_airline_turnover__add_columns_compagnie_aerienne_to_preds_df(get_regression_pred_df, get_df_companies):
    # Given
    expected = pd.DataFrame({'COMPAGNIE AERIENNE': ['NA', 'THA', 'COA'],
                             'RETARD MINUTES': [187.0, 40.0, 120],
                             'VOL': [4661, 5026, 2021],
                             'AEROPORT ARRIVEE': ['AAL', 'LTK', 'JNB'],
                             'NOMBRE DE PASSAGERS': [10, 40, 30],
                             'CHIFFRE D AFFAIRE': [7651000000, 2300000, 40500000]
                             })
    # When
    actual = get_airline_turnover(get_regression_pred_df, get_df_companies)
    # Then
    assert_frame_equal(actual, expected)


def test_add_cost_20min_delay__add_columns_with_the_cost_of_20min_delay_wrt_arrival_airport(get_regression_pred_df,
                                                                                            get_df_airport):
    # Given
    expected = pd.DataFrame({'COMPAGNIE AERIENNE': ['NA', 'THA', 'COA'],
                             'RETARD MINUTES': [187.0, 40.0, 120],
                             'VOL': [4661, 5026, 2021],
                             'AEROPORT ARRIVEE': ['AAL', 'LTK', 'JNB'],
                             'NOMBRE DE PASSAGERS': [10, 40, 30],
                             'PRIX RETARD PREMIERE 20 MINUTES': [24, 33, 53]
                             })
    # When
    actual = add_cost_20min_delay(get_df_airport, get_regression_pred_df)
    # Then
    assert_frame_equal(actual, expected)


def test_add_cost_10min_delay__add_columns_with_the_cost_of_10min_delay_wrt_arrival_airport(get_regression_pred_df,
                                                                                            get_df_airport):
    # Given
    expected = pd.DataFrame({'COMPAGNIE AERIENNE': ['NA', 'THA', 'COA'],
                             'RETARD MINUTES': [187.0, 40.0, 120],
                             'VOL': [4661, 5026, 2021],
                             'AEROPORT ARRIVEE': ['AAL', 'LTK', 'JNB'],
                             'NOMBRE DE PASSAGERS': [10, 40, 30],
                             'PRIS RETARD POUR CHAQUE MINUTE APRES 10 MINUTES': [5, 9, 3]
                             })
    # When
    actual = add_cost_10min_delay(get_df_airport, get_regression_pred_df)
    # Then
    assert_frame_equal(actual, expected)


def test_cost_of_delay__if_delay_equal_25_and_ten_min_delay_cost_2_twenty_min_delay_cost_20_return_50():
    # Given
    pred_vol = pd.Series({'RETARD MINUTES': 25,
                          'PRIX RETARD PREMIERE 20 MINUTES': 20,
                          'PRIS RETARD POUR CHAQUE MINUTE APRES 10 MINUTES': 2})
    expected = 50
    # When
    actual = cost_of_delay(pred_vol)
    # Then
    assert actual == expected


def test_cost_of_delay__if_delay_equal_15_and_ten_min_delay_cost_2_twenty_min_delay_cost_20_return_10():
    # Given
    pred_vol = pd.Series({'RETARD MINUTES': 15,
                          'PRIX RETARD PREMIERE 20 MINUTES': 20,
                          'PRIS RETARD POUR CHAQUE MINUTE APRES 10 MINUTES': 2})
    expected = 10
    # When
    actual = cost_of_delay(pred_vol)
    # Then
    assert actual == expected


def test_cost_of_delay__if_delay_equal_5_and_ten_min_delay_cost_2_twenty_min_delay_cost_20_return_0():
    # Given
    pred_vol = pd.Series({'RETARD MINUTES': 5,
                          'PRIX RETARD PREMIERE 20 MINUTES': 20,
                          'PRIS RETARD POUR CHAQUE MINUTE APRES 10 MINUTES': 2})
    expected = 0
    # When
    actual = cost_of_delay(pred_vol)
    # Then
    assert actual == expected


def test_get_airport_delay_cost__add_columns_delay_cost(get_regression_pred_df, get_df_airport):
    # Given
    expected = pd.DataFrame({'COMPAGNIE AERIENNE': ['NA', 'THA', 'COA'],
                             'RETARD MINUTES': [187.0, 40.0, 120],
                             'VOL': [4661, 5026, 2021],
                             'AEROPORT ARRIVEE': ['AAL', 'LTK', 'JNB'],
                             'NOMBRE DE PASSAGERS': [10, 40, 30],
                             'COUT DU RETARD': [909.0, 303.0, 383.0]
                             })
    # When
    actual = get_airport_delay_cost(get_regression_pred_df, get_df_airport)
    # Then
    assert_frame_equal(actual, expected)


def test_get_number_of_indemnities_asked__if_delay_equal_to_30_nb_passenger_10_return_2():
    # Given
    pred_vol = pd.Series({'RETARD MINUTES': 30,
                          'NOMBRE DE PASSAGERS': 10})
    expected = 2
    # When
    actual = get_number_of_indemnities_asked(pred_vol)
    # Then
    assert actual == expected


def test_get_number_of_indemnities_asked__if_delay_equal_to_70_nb_passenger_10_return_5():
    # Given
    pred_vol = pd.Series({'RETARD MINUTES': 70,
                          'NOMBRE DE PASSAGERS': 10})
    expected = 5
    # When
    actual = get_number_of_indemnities_asked(pred_vol)
    # Then
    assert actual == expected


def test_get_number_of_indemnities_asked__if_delay_equal_to_190_nb_passenger_10_return_7():
    # Given
    pred_vol = pd.Series({'RETARD MINUTES': 190,
                          'NOMBRE DE PASSAGERS': 10})
    expected = 7
    # When
    actual = get_number_of_indemnities_asked(pred_vol)
    # Then
    assert actual == expected


def test_compensation_due__return_200_if_delay_equal_30_and_indemnities_asked_equal_2():
    # Given
    pred_vol = pd.Series({'RETARD MINUTES': 30,
                          "NOMBRE D'INDEMNITES DEMANDEES": 2})
    expected = 200
    # When
    actual = compensation_due(pred_vol)
    # Then
    assert actual == expected


def test_compensation_due__return_300_if_delay_equal_70_and_indemnities_asked_equal_2():
    # Given
    pred_vol = pd.Series({'RETARD MINUTES': 70,
                          "NOMBRE D'INDEMNITES DEMANDEES": 2})
    expected = 300
    # When
    actual = compensation_due(pred_vol)
    # Then
    assert actual == expected


def test_compensation_due__return_600_if_delay_equal_190_and_indemnities_asked_equal_2():
    # Given
    pred_vol = pd.Series({'RETARD MINUTES': 190,
                          "NOMBRE D'INDEMNITES DEMANDEES": 2})
    expected = 600
    # When
    actual = compensation_due(pred_vol)
    # Then
    assert actual == expected


def test_get_number_of_indemnities_asked_and_compensation_due__add_nb_indemnities_asked_columns_and_indemnities_to_pay(
        get_regression_pred_df):
    # Given
    expected = pd.DataFrame({'COMPAGNIE AERIENNE': ['NA', 'THA', 'COA'],
                             'RETARD MINUTES': [187.0, 40.0, 120],
                             'VOL': [4661, 5026, 2021],
                             'AEROPORT ARRIVEE': ['AAL', 'LTK', 'JNB'],
                             'NOMBRE DE PASSAGERS': [10, 40, 30],
                             "NOMBRE D'INDEMNITES DEMANDEES": [7, 8, 15],
                             "INDEMNITES A PAYER": [2100.0, 800.0, 2250.0]
                             })
    # When
    actual = get_number_of_indemnities_asked_and_compensation_due(get_regression_pred_df)
    # Then
    assert_frame_equal(actual, expected)


def test_get_number_of_lost_customer__return_2700_if_nb_of_lost_customer_equal_3():
    # Given
    nb_of_lost_customers = 3
    expected = 2700
    # When
    actual = get_cost_of_lost_customer(nb_of_lost_customers)
    # Then
    assert actual == expected


def test_cost_of_delay_gb_airlines__return_df_gb_companies():
    # Given
    preds_with_cost_df = pd.DataFrame({'COMPAGNIE AERIENNE': ['NA', 'COA', 'NA'],
                                       'RETARD MINUTES': [187.0, 40.0, 120],
                                       'VOL': [4661, 5026, 2021],
                                       'AEROPORT ARRIVEE': ['AAL', 'LTK', 'JNB'],
                                       'NOMBRE DE PASSAGERS': [10, 40, 30],
                                       "INDEMNITES A PAYER": [2100.0, 800.0, 2250.0],
                                       "CHIFFRE D AFFAIRE": [2, 3, 2],
                                       "COUT DU RETARD": [300, 200, 100],
                                       "NOMBRE DE CLIENTS PERDUS": [34, 74, 93],
                                       "COUT DES CLIENTS PERDUS": [45, 67, 89]})
    expected = pd.DataFrame({'COMPAGNIE AERIENNE': ['COA', 'NA'],
                             'NOMBRE DE RETARD': [1, 2],
                             "CHIFFRE D AFFAIRE": [3, 2],
                             "COUT DU RETARD": [200, 400],
                             "INDEMNITES A PAYER": [800.0, 4350.0],
                             "NOMBRE DE CLIENTS PERDUS": [74, 127],
                             "COUT DES CLIENTS PERDUS": [67, 134]})
    # When
    actual = cost_of_delay_gb_airlines(preds_with_cost_df)
    # Then
    assert_frame_equal(actual, expected)


def test_get_total_to_be_paid__return_the_column_total_a_payer_which_is_the_sum_of_cost_delay_indemnities_to_pay_and_client_lost():
    # Given
    preds_with_cost = pd.DataFrame({"COUT DU RETARD": [200, 400],
                                    "INDEMNITES A PAYER": [800.0, 4350.0],
                                    "COUT DES CLIENTS PERDUS": [67, 134]})
    expected = pd.DataFrame({"COUT DU RETARD": [200, 400],
                             "INDEMNITES A PAYER": [800.0, 4350.0],
                             "COUT DES CLIENTS PERDUS": [67, 134],
                             "TOTAL A PAYER": [1067.0, 4884.0]})
    # When
    actual = get_total_to_be_paid(preds_with_cost)
    # Then
    assert_frame_equal(actual, expected)


def test_get_new_turnover_for_each_airline__add_new_turnover_columns():
    # Given
    pred_with_cost = pd.DataFrame({"COUT DU RETARD": [200, 400],
                                   "CHIFFRE D AFFAIRE": [30000, 200000],
                                   "TOTAL A PAYER": [1067.0, 4884.0]})
    expected = pd.DataFrame({"COUT DU RETARD": [200, 400],
                             "CHIFFRE D AFFAIRE": [30000, 200000],
                             "TOTAL A PAYER": [1067.0, 4884.0],
                             "NV CHIFFRE D'AFFAIRE": [28933.0, 195116.0]})
    # When
    actual = get_new_turnover_for_each_airline(pred_with_cost)
    # Then
    assert_frame_equal(actual, expected)


def test_get_percentage_of_lost_sales__add_percentage_of_CA_lost_column():
    # Given
    pred_with_cost = pd.DataFrame({"COUT DU RETARD": [200, 400],
                                   "CHIFFRE D AFFAIRE": [30000, 200000],
                                   "TOTAL A PAYER": [1067.0, 4884.0]})
    expected = pd.DataFrame({"COUT DU RETARD": [200, 400],
                             "CHIFFRE D AFFAIRE": [30000, 200000],
                             "TOTAL A PAYER": [1067.0, 4884.0],
                             "%CHIFFRE D'AFFAIRE LOST": [3.556666666666667, 2.442]})
    # When
    actual = get_percentage_of_lost_sales(pred_with_cost)
    # Then
    assert_frame_equal(actual, expected)


def test_get_percentage_of_delay_by_company(get_regression_pred_df):
    # Given
    class_preds = pd.DataFrame({'COMPAGNIE AERIENNE': ['NA', 'THA', 'COA'],
                                'RETARD': [1, 0, 1]})
    expected = pd.DataFrame({'COMPAGNIE AERIENNE': ['NA', 'THA', 'COA'],
                             'RETARD MINUTES': [187.0, 40.0, 120],
                             'VOL': [4661, 5026, 2021],
                             'AEROPORT ARRIVEE': ['AAL', 'LTK', 'JNB'],
                             'NOMBRE DE PASSAGERS': [10, 40, 30],
                             'NB DE RETARD': [1, 0, 1],
                             'NB DE VOLS': [1, 1, 1],
                             'POURCENTAGE DE RETARD': [100.0, 0, 100]})
    # When
    actual = get_percentage_of_delay_by_company(get_regression_pred_df, class_preds)
    # Then
    assert_frame_equal(actual, expected)


@patch(f'{TESTED_MODULE}.get_percentage_of_delay_by_company')
@patch(f'{TESTED_MODULE}.get_percentage_of_lost_sales')
@patch(f'{TESTED_MODULE}.get_new_turnover_for_each_airline')
@patch(f'{TESTED_MODULE}.get_total_to_be_paid')
@patch(f'{TESTED_MODULE}.cost_of_delay_gb_airlines')
@patch(f'{TESTED_MODULE}.get_number_of_indemnities_asked_and_compensation_due')
@patch(f'{TESTED_MODULE}.get_airport_delay_cost')
@patch(f'{TESTED_MODULE}.get_airline_turnover')
def test_get_prediction_with_all_cost_id_df__apply_the_function_get_airline_turnover(m_add_turnover,
                                                                                     m_airport_delay,
                                                                                     m_nb_indem,
                                                                                     m_cost_delay,
                                                                                     m_total_paid,
                                                                                     m_new_turnover,
                                                                                     m_percentage_lost,
                                                                                     m_perc,
                                                                                     get_regression_pred_df,
                                                                                     get_df_companies, get_df_airport):
    # Given
    class_preds = pd.DataFrame({'COMPAGNIE AERIENNE': ['NA', 'THA', 'COA'],
                                'RETARD': [1, 0, 1]})
    # When
    get_prediction_with_all_cost_id_df(get_regression_pred_df, get_df_companies, get_df_airport, class_preds)

    # Then
    m_add_turnover.assert_any_call(get_regression_pred_df, get_df_companies)


@patch(f'{TESTED_MODULE}.get_percentage_of_delay_by_company')
@patch(f'{TESTED_MODULE}.get_percentage_of_lost_sales', return_value=pd.DataFrame)
@patch(f'{TESTED_MODULE}.get_new_turnover_for_each_airline', return_value=pd.DataFrame)
@patch(f'{TESTED_MODULE}.get_total_to_be_paid', return_value=pd.DataFrame)
@patch(f'{TESTED_MODULE}.cost_of_delay_gb_airlines', return_value=pd.DataFrame)
@patch(f'{TESTED_MODULE}.get_number_of_indemnities_asked_and_compensation_due',
       return_value=pd.DataFrame({'COMPAGNIE AERIENNE': ['NA', 'COA', 'NA'],
                                  'RETARD MINUTES': [187.0, 40.0, 120],
                                  'NOMBRE DE PASSAGERS': [10, 40, 30]}))
@patch(f'{TESTED_MODULE}.get_airport_delay_cost', return_value=pd.DataFrame())
@patch(f'{TESTED_MODULE}.get_airline_turnover', return_value=pd.DataFrame())
def test_get_prediction_with_all_cost_id_df__apply_all_the_function_in_right_order_using_the_return_value_of_the_past_fct(
        m_add_turnover,
        m_airport_delay,
        m_nb_indem,
        m_cost_delay,
        m_total_paid,
        m_new_turnover,
        m_percentage_lost,
        m_perc,
        get_regression_pred_df,
        get_df_companies, get_df_airport):
    # Given
    class_preds = pd.DataFrame({'COMPAGNIE AERIENNE': ['NA', 'THA', 'COA'],
                                'RETARD': [1, 0, 1]})
    # When
    get_prediction_with_all_cost_id_df(get_regression_pred_df, get_df_companies, get_df_airport, class_preds)

    # Then
    m_add_turnover.assert_any_call(get_regression_pred_df, get_df_companies)
    m_airport_delay.assert_any_call(m_add_turnover.return_value, get_df_airport)
    m_nb_indem.assert_any_call(m_airport_delay.return_value)
    m_cost_delay.assert_any_call(m_nb_indem.return_value)
    m_total_paid.assert_any_call(m_cost_delay.return_value)
    m_new_turnover.assert_any_call(m_total_paid.return_value)
    m_percentage_lost.assert_any_call(m_new_turnover.return_value)
    m_perc.assert_any_call(m_percentage_lost.return_value,class_preds)
