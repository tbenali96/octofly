import pandas as pd
import plotly.express as px

from config import DATA_PATH
from src.features.feature_engineering import add_delay_binary_target, add_categorical_delay_target


def get_vols_dataframe_with_target_defined():
    df_vols = pd.read_parquet(DATA_PATH + "/parquet_format/train_data/vols.gzip")
    add_delay_binary_target(df_vols)
    df_vols["CATEGORIE RETARD"] = df_vols["RETARD A L'ARRIVEE"].apply(lambda x: add_categorical_delay_target(x))
    return df_vols


def get_scatter_plot_delay_at_arrival_wrt_distance(df_vols):
    df_vols_avec_retard = df_vols[df_vols["RETARD A L'ARRIVEE"] > 0].reset_index(drop=True)
    fig = px.scatter(df_vols_avec_retard, x="DISTANCE", y="RETARD A L'ARRIVEE")
    fig.show()


def get_histogram_nbins100_plot_delay_at_arrival_wrt_distance(df_vols):
    df_vols_avec_retard = df_vols[df_vols["RETARD A L'ARRIVEE"] > 0].reset_index(drop=True)
    fig = px.histogram(df_vols_avec_retard, x="DISTANCE", nbins=100)
    fig.show()


def main():
    df_vols = get_vols_dataframe_with_target_defined()
    get_scatter_plot_delay_at_arrival_wrt_distance(df_vols)
    get_histogram_nbins100_plot_delay_at_arrival_wrt_distance(df_vols)


if __name__ == '__main__':
    main()
