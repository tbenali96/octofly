import streamlit as st

from src.visualization.data_viz_components.data_viz_and_stats import get_vols_dataframe_with_target_defined, \
    get_scatter_plot_delay_at_arrival_wrt_distance, get_histogram_nbins100_plot_delay_at_arrival_wrt_distance, \
    get_pie_chart_that_display_the_category_delay_distribution


def write():

    st.header("DATA VISUALIZATION AND STATISTICS PAGE")

    st.subheader("1. Flight details")
    df_vols = get_vols_dataframe_with_target_defined()
    get_scatter_plot_delay_at_arrival_wrt_distance(df_vols)
    if st.button("Show the graph with histogram view"):
        get_histogram_nbins100_plot_delay_at_arrival_wrt_distance(df_vols)
    get_pie_chart_that_display_the_category_delay_distribution(df_vols)

if __name__ == "__main__":
    write()