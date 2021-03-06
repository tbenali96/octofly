import os

from PIL import Image
import streamlit as st
import awesome_streamlit as ast

from config import DATA_PATH
from src.app_visualization.pages import data_vizualisation, run_predict_model_visualization, show_metrics_and_KPIs, \
    train_model

ast.core.services.other.set_logging_format()

if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)

PAGES = {
    "Visualitation des données (historiques) et statistiques": data_vizualisation,
    "Entrainement du modèle : choix minutes de retard" : train_model,
    "Modèle Inférence": run_predict_model_visualization,
    "Interpretation des résultats et KPIs": show_metrics_and_KPIs,
}


def main():
    """Main function of the App"""
    columns = st.sidebar.beta_columns(2)
    if columns[0].button("Exit App"):
        os._exit(1)
    logo = Image.open('src/app_visualization/accento_logo_v2.png')
    logo = logo.resize((400, 300))
    columns[1].image(logo, width=155)
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]
    # with st.spinner(f"Loading {selection} ..."):
    ast.shared.components.write_page(page)


if __name__ == "__main__":
    main()
