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

        data: pd.DataFrame = load("data.csv")

        assert data is not None, "data load failure."

        data_train = data
        # data_train = data[:450]
        # data_valid = data[450:]

        x_train = data_train.iloc[:, 2:].to_numpy()
        y_train = data_train.iloc[:, 1] == "M"
        m = data_train.iloc[:, 1] == "M"
        b = data_train.iloc[:, 1] == "B"
        y_train = pd.DataFrame({"M":m, "B":b})
        y_train = y_train.to_numpy().astype(int)
        # y_train = y_train.reshape((y_train.shape[0], 1))

        print(y_train.shape)

        # x_valid = data_valid.iloc[:, 2:].to_numpy()
        # y_valid = data_valid.iloc[:, 1].to_numpy()
        # y_valid = y_valid.reshape((y_valid.shape[0], 1))
        
        initializer = keras.initializers.HeUniform(seed=None)

        print(x_train.shape, y_train.shape)

        model = keras.Sequential([
            keras.layers.Input(shape=(x_train.shape[1],)),
            keras.layers.Dense(24, activation='sigmoid'),
            keras.layers.Dense(24, activation='sigmoid'),
            keras.layers.Dense(24, activation='sigmoid'),
            keras.layers.Dense(y_train.shape[1], activation='softmax')
        ])
        
        optimizer = keras.optimizers.Adam()
        model.compile(optimizer=optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy', 'mse'])

        model.summary()

        es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=100, start_from_epoch=500)

        # mlp_model = model.fit(x_train, y_train, epochs=1000, batch_size=200, verbose=1, shuffle=True)
        mlp_model = model.fit(x_train, y_train, validation_split=0.1, epochs=5000, batch_size=200, callbacks=[es])
        # mlp_model2 = model.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=8, verbose=1)

        y_loss = mlp_model.history['loss']
        x_len = mlp_model.epoch
        plt.plot(x_len, y_loss, c='blue', label="Train-set Loss")
        
        if "accuracy" in mlp_model.history.keys():
            y_acc = mlp_model.history['accuracy']
            plt.plot(x_len, y_acc, c='black', label="Train-set Acc")
        
        if "val_loss" in mlp_model.history.keys():
            y_val_loss = mlp_model.history['val_loss']
            plt.plot(x_len, y_val_loss, c='red', label="Valid-set Loss")
        
        if "val_accuracy" in mlp_model.history.keys():
            y_val_acc = mlp_model.history['val_accuracy']
            plt.plot(x_len, y_val_acc, c='green', label="Valid-set Acc")

        plt.legend(loc='upper right')
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()

        eval = model.evaluate(x_train[-100:], y_train[-100:], batch_size=1, return_dict=True)

        return model, mlp_model, eval

    except Exception as e:
        print(f"{e.__class__.__name__}: {e}", file=sys.stderr)

if __name__ == "__main__":
    model, mlp_model, eval= main()