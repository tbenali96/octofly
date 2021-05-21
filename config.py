import os

app_directory = os.getcwd()

SCALERS_MODEL_PATH = os.path.join(app_directory, "models/train_features_scalers")
MODEL_PATH = os.path.join(app_directory, "models")
DATA_PATH = os.path.join(app_directory, "data")