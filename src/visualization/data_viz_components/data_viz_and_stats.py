import pandas as pd
import plotly.express as px
import streamlit as st

from config import DATA_PATH
from src.features.feature_engineering import add_delay_binary_target, add_categorical_delay_target, \
    get_category_delay_target_in_string


def get_vols_dataframe_with_target_defined():
    df_vols = pd.read_parquet(DATA_PATH + "/parquet_format/train_data/vols.gzip")
    add_delay_binary_target(df_vols)
    df_vols["CATEGORIE RETARD"] = df_vols["RETARD A L'ARRIVEE"].apply(lambda x: add_categorical_delay_target(x))
    return df_vols


def get_scatter_plot_delay_at_arrival_wrt_distance(df_vols):
    df_vols_avec_retard = df_vols[df_vols["RETARD A L'ARRIVEE"] > 0].reset_index(drop=True)
    fig = px.scatter(df_vols_avec_retard, x="DISTANCE", y="RETARD A L'ARRIVEE",
                     title="Repartition of the delay in arrival with respect to distance")
    st.plotly_chart(fig)


def get_histogram_nbins100_plot_delay_at_arrival_wrt_distance(df_vols):
    df_vols_avec_retard = df_vols[df_vols["RETARD A L'ARRIVEE"] > 0].reset_index(drop=True)
    df_vols_avec_retard['Category retard str'] = df_vols_avec_retard['CATEGORIE RETARD'].map(
        lambda x: "delay <= 3h" if x == 1 else "delay > 3h")
    fig = px.histogram(df_vols_avec_retard, x="DISTANCE", nbins=100, color="Category retard str",
                       labels={"CATEGORIE RETARD": "CATEGORY DELAY"},
                       title="Repartition of the delay in arrival with respect to distance (hist view)")
    st.plotly_chart(fig)


def get_pie_chart_that_display_the_category_delay_distribution(df_vols):
    df_vols_avec_retard_wth_count = df_vols[['CATEGORIE RETARD']]
    df_vols_avec_retard_wth_count['count'] = 1
    df_vols_avec_retard_wth_count_gb = df_vols_avec_retard_wth_count.groupby(['CATEGORIE RETARD'], as_index=False).sum()
    df_vols_avec_retard_wth_count_gb['Category retard str'] = df_vols_avec_retard_wth_count_gb['CATEGORIE RETARD'].map(
        lambda x: get_category_delay_target_in_string(x))
    fig = px.pie(df_vols_avec_retard_wth_count_gb, values='count', names='Category retard str',
                 color_discrete_map={"on_time": "green",
                                     "delay <= 3h": "orange",
                                     "delay > 3h": "red"},
                 title='Distribution of the category delay')
    st.plotly_chart(fig)


def main():
    df_vols = get_vols_dataframe_with_target_defined()
    get_scatter_plot_delay_at_arrival_wrt_distance(df_vols)
    get_histogram_nbins100_plot_delay_at_arrival_wrt_distance(df_vols)
    get_pie_chart_that_display_the_category_delay_distribution(df_vols)


if __name__ == '__main__':
    main()
