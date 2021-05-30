import os

directory = "models/train_features_scalers"

SCALERS_MODEL_PATH = os.path.join(directory)
MODEL_PATH = os.path.join(directory, "models")
DATA_PATH = os.path.join(directory, "data")

if __name__ == '__main__':
    print(SCALERS_MODEL_PATH)
