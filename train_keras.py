#!/usr/bin/python3

import os
import sys
import pickle
import pandas as pd
import time
from tensorflow import keras
from matplotlib import pyplot as plt

from load_csv import load


if __name__ == "__main__":
    try:
        train_data: pd.DataFrame = load("train.csv", header=None)
        valid_data: pd.DataFrame = load("valid.csv", header=None)

        assert train_data is not None and valid_data is not None, "data load failure."

        data_train = train_data
        x_train = data_train.iloc[:, 2:].to_numpy()
        y_train = data_train.iloc[:, 1] == "M"
        m = data_train.iloc[:, 1] == "M"
        b = data_train.iloc[:, 1] == "B"
        y_train = pd.DataFrame({"M": m, "B": b})
        y_train = y_train.to_numpy().astype(int)

        valid_data: pd.DataFrame = load("valid.csv", header=None)
        data_valid = valid_data
        x_valid = data_valid.iloc[:, 2:].to_numpy()
        y_valid = data_valid.iloc[:, 1] == "M"
        m = data_valid.iloc[:, 1] == "M"
        b = data_valid.iloc[:, 1] == "B"
        y_valid = pd.DataFrame({"M": m, "B": b})
        y_valid = y_valid.to_numpy().astype(int)

        # apply standard scaler value from smlp.
        data_mean, data_std = None, None
        if os.path.exists("scale.pkl"):
            with open("scale.pkl", "rb") as f:
                data_mean, data_std = pickle.load(f)
            if data_mean is not None:
                x_train = (x_train - data_mean) / data_std
                x_valid = (x_valid - data_mean) / data_std
                print("standard scaler from smlp applied")

        # seed range should be 0 to (2 ** 32 - 1)
        initializer = keras.initializers.HeUniform(
            seed=(time.time_ns() // 1000) & 0xFFFFFFFF
        )
        
        # initializer = keras.initializers.GlorotUniform(
        #     seed=(time.time_ns() // 1000) & 0xFFFFFFFF
        # )

        # keras model setting
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(x_train.shape[1],)),
                keras.layers.Dense(
                    24, activation="sigmoid", kernel_initializer=initializer
                ),
                keras.layers.Dense(
                    24, activation="sigmoid", kernel_initializer=initializer
                ),
                # keras.layers.Dense(
                #     24, activation="sigmoid", kernel_initializer=initializer
                # ),
                # keras.layers.Dense(
                #     24, activation="sigmoid", kernel_initializer=initializer
                # ),
                keras.layers.Dense(
                    24, activation="sigmoid", kernel_initializer=initializer
                ),
                keras.layers.Dense(
                    y_train.shape[1],
                    activation="softmax",
                    kernel_initializer=initializer,
                ),
            ]
        )

        optimizer = keras.optimizers.SGD(
            learning_rate=0.01, momentum=0, nesterov=False
        )
        # optimizer = keras.optimizers.SGD(
        #     learning_rate=0.01, momentum=0.9, nesterov=True
        # )
        # optimizer = keras.optimizers.RMSprop(momentum=0.9)
        # optimizer = keras.optimizers.Adam()

        # optimizer = keras.optimizers.legacy.SGD(
        #     learning_rate=0.01, momentum=0.9, nesterov=True
        # )
        # optimizer = keras.optimizers.legacy.RMSprop(momentum=0.9)
        # optimizer = keras.optimizers.legacy.Adam()

        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy"],
            # metrics=["accuracy", "mse"],
        )

        model.summary()

        es = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=100, start_from_epoch=500
        )

        mlp_model = model.fit(
            x_train,
            y_train,
            validation_data=(x_valid, y_valid),
            epochs=5000,
            batch_size=200,
            callbacks=[es],
        )

        keys = [key for key in mlp_model.history.keys() if "val_" not in key]
        n_graph = len(keys)

        w = 2
        h = n_graph // w + (1 if n_graph % w != 0 else 0)
        fig, axs = plt.subplots(h, w)

        # red for train, blue for valid if exist
        color = ["red", "blue"]
        x_len = mlp_model.epoch

        for i, key in enumerate(keys):

            cur_h, cur_w = divmod(i, w)
            label_name = key[0].upper() + key[1:]

            if h > 1:
                cur_ax = axs[cur_h, cur_w]
            else:
                cur_ax = axs[cur_w]

            cur_ax.set(xlabel="epoch", ylabel=key)
            cur_ax.plot(
                x_len,
                mlp_model.history[key],
                c=color[0],
                label=f"Train-set {label_name}",
            )
            if "val_" + key in mlp_model.history.keys():
                cur_ax.plot(
                    x_len,
                    mlp_model.history["val_" + key],
                    c=color[1],
                    label=f"Valid-set {label_name}",
                )
            cur_ax.legend(loc="best", fontsize="x-small")
            cur_ax.grid()

        plt.tight_layout()
        plt.show()

        with open("model_keras.pkl", "wb") as f:
            pickle.dump(model, f)
        print("model saved as 'model_keras.pkl' to current directory.")

        # test_data: pd.DataFrame = load("test.csv", header=None)
        # data_test = test_data
        # x_test = data_test.iloc[:, 2:].to_numpy()
        # y_test = data_test.iloc[:, 1] == "M"
        # m = data_test.iloc[:, 1] == "M"
        # b = data_test.iloc[:, 1] == "B"
        # y_test = pd.DataFrame({"M": m, "B": b})
        # y_test = y_test.to_numpy().astype(int)

        # eval = model.evaluate(x_test, y_test, batch_size=1, return_dict=True)

    except Exception as e:
        print(f"{e.__class__.__name__}: {e}", file=sys.stderr)
