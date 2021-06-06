import pandas as pd
import streamlit as st

from src.app_visualization.data_viz_components.metrics_and_KPIs import get_prediction_with_all_cost_id_df, \
    plot_turnover_of_airlines_and_the_total_to_be_paid, plot_former_turnover_of_airlines_and_the_new_one


def write():
    st.header("")
    st.subheader("1. Interprétation métriques et KPIs")
    df_aeroports = pd.read_parquet("../../data/aggregated_data/aeroports.gzip")
    reg_preds = pd.read_parquet("../../data/predictions/predictions_regression.gzip", compression='gzip')
    add_selectbox = st.sidebar.selectbox(
        "How would you like to receive the results and KPIs?",
        ("Email", "Home phone", "Mobile phone")
    )
    prediction_with_cost_gb_airline = get_prediction_with_all_cost_id_df()
    plot_turnover_of_airlines_and_the_total_to_be_paid(prediction_with_cost_gb_airline)
    plot_former_turnover_of_airlines_and_the_new_one(prediction_with_cost_gb_airline)

if __name__ == "__main__":
    write()
