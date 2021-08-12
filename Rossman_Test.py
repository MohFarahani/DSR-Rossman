import sys
from train_class import Rossman

# PATH to Input data
PATH_STORE_MODIFIED = "data/store_modified.csv"
MODEL_NAME = "XGBoost"

if __name__ == "__main__":
    path_test = sys.argv[1]
    rosmann = Rossman()
    rosmann.testing(path_test)