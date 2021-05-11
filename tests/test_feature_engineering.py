import pandas as pd
from src.features.build_features import delete_irrelevant_columns

def delete_irrelevant_columns_doit_renvoyer_le_dataset_d_entrainement_sans_la_colonne_niveau_de_securite():

    df = pd.DataFrame({"IDENTIFIANT":["1", "2"], "NIVEAU DE SECURITE":["10", "10"]})
    new_df = delete_irrelevant_columns(df)
    assert "NIVEAU DE SECURITE" not in new_df.columns


if __name__ == '__main__':
    delete_irrelevant_columns_doit_renvoyer_le_dataset_d_entrainement_sans_la_colonne_niveau_de_securite()