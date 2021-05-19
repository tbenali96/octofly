import datetime
from unittest.mock import patch, Mock

import pandas as pd
from pandas._testing import assert_series_equal, assert_frame_equal

from src.features.feature_engineering import delete_irrelevant_columns, scale_features, scale_feature_in_df, \
    create_is_night_flight_feature, format_hour, convert_time_into_datetime, change_hour_format, \
    add_night_flight_binary_features, add_delay_binary_target, extracting_time_features_from_date

TESTED_MODULE = 'src.features.feature_engineering'


def test_delete_irrelevant_columns_doit_renvoyer_le_dataset_d_entrainement_sans_la_colonne_niveau_de_securite():
    df = pd.DataFrame({"IDENTIFIANT": ["1", "2"], "NIVEAU DE SECURITE": ["10", "10"]})
    new_df = delete_irrelevant_columns(df)
    assert "NIVEAU DE SECURITE" not in new_df.columns


"""
@patch(f'{TESTED_MODULE}.scale_feature_in_df')
def test_scale_features__apply_scaling_to_features(m_scaling_feature):
    # Given
    df = pd.DataFrame({'feature1': [1,10], 'feature2': [2,20]})
    features_to_scale = ['feature1','feature2']
    path = 'fake_path'

    # When
    scale_features(df, features_to_scale, path)

    # Then
    m_scaling_feature.assert_any_call(df,'feature1', 'fake_path')
    m_scaling_feature.assert_any_call(df, 'feature2', 'fake_path')



@patch(f'{TESTED_MODULE}.StandardScaler.transform', return_value='df_transform')
@patch(f'{TESTED_MODULE}.save_scaler')
@patch(f'{TESTED_MODULE}.load')
@patch(f'{TESTED_MODULE}.StandardScaler.fit')
@patch(f'{TESTED_MODULE}.StandardScaler')
def test_scale_feature_in_df___fit_model_if_train(m_scaler, m_fit_scaler, m_load, m_save,
                                                  m_transform):
    # Given
    m_scaler = Mock()
    df = pd.DataFrame({'feature1': [1, 10, 100], 'feature2': [2, 20, 200]})
    feature = 'feature1'
    path = 'fake_path'

    # When
    scale_feature_in_df(df, 'feature1', path)

    # Then
    m_fit_scaler.assert_called_with(np.array(df[feature]).reshape(-1, 1))
"""


def test_create_is_night_flight_feature__fill_with_0_or_1_if_night_flight_between_2300_and_600():
    # Given
    feature = 'DEPART PROGRAMME'
    is_night_flight_feature = "DEPART DE NUIT"
    df = pd.DataFrame({'DEPART PROGRAMME': [2345, 2249, 504, 1000]})
    expected_df_feature = pd.Series([1, 0, 1, 0], name="DEPART DE NUIT")

    # When
    create_is_night_flight_feature(feature, is_night_flight_feature, df)

    # Then
    assert_series_equal(df[is_night_flight_feature], expected_df_feature)


def test_format_hour__change_str_xxxx_into_hour_min_format():
    # Given
    str_xxxx_hour = '2345'
    expected_format = pd.to_timedelta('23:45:00')

    # When
    actual_format = format_hour(str_xxxx_hour)

    # Then
    assert actual_format == expected_format


def test_format_hour__add_0_before_time_if_len_of_time_is_less_than_4():
    # Given
    str_xxxx_hour = '612'
    expected_format = pd.to_timedelta('06:12:00')

    # When
    actual_format = format_hour(str_xxxx_hour)

    # Then
    assert actual_format == expected_format


def test_convert_time_into_datetime():
    # Given
    float_hour = 612.0
    expected_format = datetime.time(6, 12)
    # When
    actual_format = convert_time_into_datetime(float_hour)
    # Then
    assert actual_format == expected_format


def test_change_hour_format__given_a_series_of_float_return_a_series_of_time_format():
    # Given
    df = pd.DataFrame({'ARRIVEE PROGRAMMEE': [612, 2345],
                       'DEPART PROGRAMME': [712, 945]})
    feature1 = 'ARRIVEE PROGRAMMEE'
    feature2 = 'DEPART PROGRAMME'
    expected_feature1 = pd.Series([pd.to_timedelta('06:12:00'), pd.to_timedelta('23:45:00')],
                                  name='ARRIVEE PROGRAMMEE')
    expected_feature2 = pd.Series([pd.to_timedelta('07:12:00'), pd.to_timedelta('09:45:00')],
                                  name='DEPART PROGRAMME')
    # When
    change_hour_format(df)

    # Then
    assert_series_equal(df[feature1], expected_feature1)
    assert_series_equal(df[feature2], expected_feature2)


"""
@patch(f'{TESTED_MODULE}.create_is_night_flight_feature')
def test_add_night_flight_binary_features__apply_twice_create_is_night_flight_feature(m_create_bool_night_flight):
    # Given
    df = pd.DataFrame({'DEPART PROGRAMME': [612, 2345],
                       'ARRIVEE PROGRAMMEE': [490, 1045]})
    feature1 = 'DEPART PROGRAMME'
    is_night_flight_feature1 = "DEPART DE NUIT"
    feature2 = 'ARRIVEE PROGRAMMEE'
    is_night_flight_feature2 = "ARRIVEE DE NUIT"
    m_create_bool_night_flight.return_value = pd.DataFrame({'DEPART PROGRAMME': [612, 2345],
                       'ARRIVEE PROGRAMMEE': [490, 1045],
                        'DEPART DE NUIT': [0,1]})
    # When
    add_night_flight_binary_features(df)

    # Then
    m_create_bool_night_flight.assert_called_with(feature1,is_night_flight_feature1, df)
    m_create_bool_night_flight.assert_called_with(feature2, is_night_flight_feature2, df)
"""


def test_add_delay_binary_target__create_retard_target_with_1_if_retard_a_larrive_else_0():
    # Given
    target = "RETARD"
    df = pd.DataFrame({"RETARD A L'ARRIVEE": [-5.0, 45, -1]})
    expected = pd.Series([0, 1, 0], name='RETARD')

    # When
    add_delay_binary_target(df)

    # Then
    assert_series_equal(df[target], expected)


def test_extracting_time_features_from_date__give_new_date_feature():
    # Given
    df = pd.DataFrame({'year': [2015, 2016],
                       'month': [2, 3],
                       'day': [4, 5]})
    df = pd.to_datetime(df)
    df_new = pd.DataFrame({'DATE': df,
                       'DEPART PROGRAMME': [1809, 510],
                       'ARRIVEE PROGRAMMEE': [1658, 553]})
    expected = pd.DataFrame({'DATE': df,
                             'DEPART PROGRAMME': [1809, 510],
                             'ARRIVEE PROGRAMMEE': [1658, 553],
                             'DAY OF THE WEEK':[3,6],
                             'WEEKEND' :[0,1],
                             'MONTH':[2,3],
                             'DAY OF THE MONTH':[4,5],
                             "HEURE DE DEPART": [18,5],
                             "HEURE D'ARRIVEE":[16,5]})

    # When
    actual = extracting_time_features_from_date(df_new)

    # Then
    assert_frame_equal(actual, expected)
