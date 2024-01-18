#!/usr/bin/python3

import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from load_csv import load
from model import Model
from layers import Layers
from callback import EarlyStopping
from optimizer import SGD, RMSprop, Adam
import pickle

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
        y_train = pd.DataFrame({"M":m, "B":b})
        y_train = y_train.to_numpy().astype(int)

        data_valid = valid_data
        x_valid = data_valid.iloc[:, 2:].to_numpy()
        y_valid = data_valid.iloc[:, 1] == "M"
        m = data_valid.iloc[:, 1] == "M"
        b = data_valid.iloc[:, 1] == "B"
        y_valid = pd.DataFrame({"M":m, "B":b})
        y_valid = y_valid.to_numpy().astype(int)

        mlp = Model.sequential([
            Layers.Input(x_train.shape[1]),
            Layers.Dense(24, activation='sigmoid', weights_initializer="heUniform"),
            Layers.Dense(24, activation='sigmoid', weights_initializer="heUniform"),
            Layers.Dense(24, activation='sigmoid', weights_initializer="heUniform"),
            Layers.Dense(y_train.shape[1], activation='softmax', weights_initializer="heUniform")
        ])

        # optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        # optimizer = RMSprop(momentum=0.9)
        optimizer = Adam()

        mlp.compile(optimizer=optimizer, loss="binaryCrossentropy", metrics=['accuracy', 'mse'])
        # mlp.compile(optimizer=optimizer, loss="binaryCrossentropy", metrics=['accuracy'])

        mlp.summary()

        es = EarlyStopping(monitor="val_accuracy", patience=100, start_from_epoch=500)
        # es = EarlyStopping(monitor="val_loss", patience=100, start_from_epoch=500)
        history = mlp.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=200, epochs=5000, callbacks=[es])

        y_loss = history.history['loss']
        x_len = history.epoch
        plt.plot(x_len, y_loss, c='blue', label="Train-set Loss")
        
        if "val_loss" in history.history.keys():
            y_val_loss = history.history['val_loss']
            plt.plot(x_len, y_val_loss, c='red', label="Valid-set Loss")
        plt.legend(loc='upper right')
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Learning Curve')
        plt.show()
        
        if "accuracy" in history.history.keys():
            y_acc = history.history['accuracy']
            plt.plot(x_len, y_acc, c='black', label="Train-set Acc")
        
        if "val_accuracy" in history.history.keys():
            y_val_acc = history.history['val_accuracy']
            plt.plot(x_len, y_val_acc, c='green', label="Valid-set Acc")

        if "accuracy" in history.history.keys():
            plt.legend(loc='lower right')
            plt.grid()
            plt.xlabel('epoch')
            plt.ylabel('accuracy')
            plt.title('Learning Curve')
            plt.show()

        with open("model.pkl", "wb") as f:
            pickle.dump(mlp, f)
        print("model saved as 'model.pkl' to current directory.")
        # eval = mlp.evaluate(x_train[-100:], y_train[-100:], batch_size=1, return_dict=False)

        # print(mlp.predict(x_train[-100:]))

    except Exception as e:
        print(f"{e.__class__.__name__}: {e}", file=sys.stderr)
