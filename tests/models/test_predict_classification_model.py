from unittest.mock import patch

import pandas as pd

from src.models.predict_classification_model import calculate_prediction

TESTED_MODULE = 'src.models.predict_classification_model'


def test_calculate_prediction__return_0_if_thresh_equal_0_point_4_and_x_0_point_3():
    # Given
    x = 0.3
    thresh = 0.4
    expected = 0
    # When
    actual = calculate_prediction(x, thresh)
    # Then
    assert actual == expected


def test_calculate_prediction__return_1_if_thresh_equal_0_point_4_and_x_0_point_6():
    # Given
    x = 0.6
    thresh = 0.4
    expected = 1
    # When
    actual = calculate_prediction(x, thresh)
    # Then
    assert actual == expected

