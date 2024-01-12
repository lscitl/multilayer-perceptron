#!/usr/bin/python3

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from load_csv import load
from model import Model
from layers import Layers

if __name__ == "__main__":

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

    mlp = Model.sequential([
        Layers.Input(x_train.shape[1]),
        Layers.Dense(24, activation='sigmoid', weights_initializer="heUniform"),
        Layers.Dense(24, activation='sigmoid', weights_initializer="heUniform"),
        Layers.Dense(24, activation='sigmoid', weights_initializer="heUniform"),
        Layers.Dense(y_train.shape[1], activation='softmax', weights_initializer="heUniform")
    ])

    mlp.compile(optimizer="adam", loss="binaryCrossentropy", metrics=['Accuracy'])

    history = mlp.fit(x_train, y_train, validation_split=0.1, batch_size=200, epochs=5000)

    y_loss = history.history['loss']
    y_acc = history.history['accuracy']
    x_len = np.arange(len(y_loss))
    plt.plot(x_len, y_loss, marker='.', c='blue', label="Train-set Loss")
    plt.plot(x_len, y_acc, marker='.', c='black', label="Train-set Acc")
    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()