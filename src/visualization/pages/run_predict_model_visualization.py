import streamlit as st

from src.visualization.data_viz_components.data_viz_and_stats import get_vols_dataframe_with_target_defined


def write():
    st.header("")
    st.subheader("1. Informations relatives aux compagnies aériennes")
    df_vols = get_vols_dataframe_with_target_defined()
    add_selectbox = st.sidebar.selectbox(
        "How would you like to be contacted?",
        ("Email", "Home phone", "Mobile phone")
    )


if __name__ == "__main__":
    write()