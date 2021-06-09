import pandas as pd
from pandas._testing import assert_frame_equal

from src.data.aggregate_data import concat_dataframes


def test_concat_dataframes__if_one_more_column_replace_missing_value_by_None_after_concat_df():
    # Given
    df1 = pd.DataFrame({'COL1': [1, 3, 5],
                        'COL2': [6, 9, 7],
                        'COL3': [1, 2, 3]})
    df2 = pd.DataFrame({'COL1': [18, 23, 12],
                        'COL2': [34, 56, 89]})
    expected = pd.DataFrame({'COL1': [1, 3, 5, 18, 23, 12],
                             'COL2': [6, 9, 7, 34, 56, 89],
                             'COL3': [1, 2, 3, None, None, None]})
    # When
    actual = concat_dataframes(df1, df2)
    # Then
    assert_frame_equal(actual, expected)
