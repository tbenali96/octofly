import os

project_directory = "/Users/lea.naccache/CODE/Skool AI3/Certification/octofly/"

SCALERS_MODEL_PATH = os.path.join(project_directory + "models/train_features_scalers")
MODEL_PATH = os.path.join(project_directory, "models")
DATA_PATH = os.path.join(project_directory, "data")

if __name__ == '__main__':
    print(SCALERS_MODEL_PATH)
