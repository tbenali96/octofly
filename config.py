import os

#directory = "octofly"
directory = os.getcwd()

MODEL_PATH = os.path.join(directory, "models")
DATA_PATH = os.path.join(directory, "data")

if __name__ == '__main__':
    print(MODEL_PATH, DATA_PATH)
