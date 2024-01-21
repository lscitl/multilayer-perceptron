#!/usr/bin/python3

import sys
import numpy as np
import pandas as pd
import time
from tensorflow import keras
from matplotlib import pyplot as plt

from load_csv import load


def main():
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

        data_valid = valid_data
        x_valid = data_valid.iloc[:, 2:].to_numpy()
        y_valid = data_valid.iloc[:, 1] == "M"
        m = data_valid.iloc[:, 1] == "M"
        b = data_valid.iloc[:, 1] == "B"
        y_valid = pd.DataFrame({"M": m, "B": b})
        y_valid = y_valid.to_numpy().astype(int)

        initializer = keras.initializers.HeUniform(
            seed=int(time.time() * 1000000) & 0xFFFFFFFF
        )

        print(x_train.shape, y_train.shape)

        model = keras.Sequential(
            [
                keras.layers.Input(shape=(x_train.shape[1],)),
                keras.layers.Dense(
                    24, activation="sigmoid", kernel_initializer=initializer
                ),
                keras.layers.Dense(
                    24, activation="sigmoid", kernel_initializer=initializer
                ),
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

        # optimizer = keras.optimizers.Adam()
        # optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        # optimizer = keras.optimizers.RMSprop(momentum=0.9)

        # optimizer = keras.optimizers.legacy.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        # optimizer = keras.optimizers.legacy.RMSprop(momentum=0.9)
        optimizer = keras.optimizers.legacy.Adam()

        model.compile(
            optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
        )

        model.summary()

        es = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=100, start_from_epoch=500
        )

        # mlp_model = model.fit(x_train, y_train, epochs=1000, batch_size=200, verbose=1, shuffle=True)
        mlp_model = model.fit(
            x_train,
            y_train,
            validation_data=(x_valid, y_valid),
            epochs=5000,
            batch_size=200,
            callbacks=[es],
        )
        # mlp_model2 = model.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=8, verbose=1)

        y_loss = mlp_model.history["loss"]
        x_len = mlp_model.epoch
        plt.plot(x_len, y_loss, c="blue", label="Train-set Loss")

        if "val_loss" in mlp_model.history.keys():
            y_val_loss = mlp_model.history["val_loss"]
            plt.plot(x_len, y_val_loss, c="red", label="Valid-set Loss")
        plt.legend(loc="upper right")
        plt.grid()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.show()

        if "accuracy" in mlp_model.history.keys():
            y_acc = mlp_model.history["accuracy"]
            plt.plot(x_len, y_acc, c="black", label="Train-set Acc")

        if "val_accuracy" in mlp_model.history.keys():
            y_val_acc = mlp_model.history["val_accuracy"]
            plt.plot(x_len, y_val_acc, c="green", label="Valid-set Acc")

        if "accuracy" in mlp_model.history.keys():
            plt.legend(loc="lower right")
            plt.grid()
            plt.xlabel("epoch")
            plt.ylabel("accuracy")
            plt.show()

        # eval = model.evaluate(x_train[-100:], y_train[-100:], batch_size=1, return_dict=True)

        return model, mlp_model, eval

    except Exception as e:
        print(f"{e.__class__.__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    model, mlp_model, eval = main()
