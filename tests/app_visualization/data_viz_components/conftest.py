import pandas as pd
import pytest


@pytest.fixture
def get_regression_pred_df():
    return pd.DataFrame({'COMPAGNIE AERIENNE': ['NA', 'THA', 'COA'],
                         'RETARD MINUTES': [187.0, 40.0, 120],
                         'VOL': [4661, 5026, 2021],
                         'AEROPORT ARRIVEE': ['AAL', 'LTK', 'JNB'],
                         'NOMBRE DE PASSAGERS':[10,40,30]})


@pytest.fixture
def get_df_companies():
    return pd.DataFrame({'COMPAGNIE':['Try Hard Airlines', 'Corporate Overlord Airways', 'Neverland Airlines'],
                         'CODE': ['THA', 'COA', 'NA'],
                         'CHIFFRE D AFFAIRE': [2300000, 40500000,7651000000]})

@pytest.fixture
def get_df_airport():
    return pd.DataFrame({'CODE IATA': ['JNB', 'AAL', 'LTK'],
                         'PRIX RETARD PREMIERE 20 MINUTES': [53,24,33],
                         'PRIS RETARD POUR CHAQUE MINUTE APRES 10 MINUTES': [3,5,9]})