import os
import sys
import pickle
import pandas as pd
import numpy as np

from load_csv import load
from smlp.utils import get_one_hot_value
from tensorflow import keras

if __name__ == "__main__":
    try:
        test_data: pd.DataFrame = load("test.csv", header=None)

        assert test_data is not None, "data load failure."

        data_test = test_data
        x_test = data_test.iloc[:, 2:].to_numpy()
        y_test = data_test.iloc[:, 1]

        data_mean, data_std = None, None
        if os.path.exists("scale.pkl"):
            with open("scale.pkl", "rb") as f:
                data_mean, data_std = pickle.load(f)
            if data_mean is not None:
                x_test = (x_test - data_mean) / data_std
                print("standard scaler from smlp applied")

        assert os.path.exists("model_keras.pkl"), "keras model is not found."
        with open("model_keras.pkl", "rb") as f:
            model = pickle.load(f)

        predict = get_one_hot_value(model.predict(x_test)).astype(int)

        predict_list = []
        for pred in predict:
            if np.all(pred == np.array([1, 0])):
                predict_list.append("M")
            else:
                predict_list.append("B")

        result = pd.DataFrame({"real": y_test, "pred": np.array(predict_list)})
        print(result)

        accuracy = np.sum(
            (result["real"] == result["pred"]).to_numpy().astype(int)
        ) / len(result)
        print(f"Accuracy: {accuracy * 100:.4}%")

    except Exception as e:
        print(f"{e.__class__.__name__}: {e}", file=sys.stderr)
