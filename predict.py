#!/usr/bin/python3

import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from load_csv import load
from model import Model, get_one_hot_value
import pickle

if __name__ == "__main__":

    try:

        test_data: pd.DataFrame = load("test.csv", header=None)

        assert test_data is not None, "data load failure."

        data_test = test_data
        x_test = data_test.iloc[:, 2:].to_numpy()
        y_test = data_test.iloc[:, 1] == "M"
        m = data_test.iloc[:, 1] == "M"
        b = data_test.iloc[:, 1] == "B"
        y_test = pd.DataFrame({"M":m, "B":b})
        y_test = y_test.to_numpy().astype(int)
        print(data_test)

        with open("model.pkl", "rb") as f:
            mlp: Model = pickle.load(f)

        ans = mlp.predict(x_test)

        print(np.all(get_one_hot_value(ans) == y_test, axis=1))

    except Exception as e:
        print(f"{e.__class__.__name__}: {e}", file=sys.stderr)
