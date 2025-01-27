#!/usr/bin/python3

import sys
import pandas as pd
import pickle
import argparse

from matplotlib import pyplot as plt
from load_csv import load

import smlp


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Train the dataset file.")

        parser.add_argument(
            "--dataset", type=str, required=False, help="Path to the dataset file."
        )
        parser.add_argument(
            "--validset",
            type=str,
            required=False,
            help="Path to the valid dataset file.",
        )

        arg = parser.parse_args()

        if arg.dataset is None:
            train_data: pd.DataFrame = load("train.csv", header=None)
        else:
            train_data: pd.DataFrame = load(arg.dataset, header=None)

        if arg.validset is None:
            valid_data: pd.DataFrame = load("valid.csv", header=None)
        else:
            valid_data: pd.DataFrame = load(arg.validset, header=None)

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

        # initializer = "glorotUniform"
        initializer = "heUniform"

        model_list = []
        for _ in range(4):
            model_list.append(
                smlp.Sequential(
                    [
                        smlp.layers.Input(x_train.shape[1]),
                        smlp.layers.Dense(
                            24, activation="sigmoid", weights_initializer=initializer
                        ),
                        smlp.layers.Dense(
                            24, activation="sigmoid", weights_initializer=initializer
                        ),
                        # smlp.layers.Dense(
                        #     24, activation="sigmoid", weights_initializer=initializer
                        # ),
                        smlp.layers.Dense(
                            24, activation="sigmoid", weights_initializer=initializer
                        ),
                        smlp.layers.Dense(
                            y_train.shape[1],
                            activation="softmax",
                            weights_initializer=initializer,
                        ),
                    ]
                )
            )

        optimizer_list = []

        optimizer_list.append(
            smlp.optimizers.SGD(learning_rate=0.02, momentum=0.0, nesterov=False)
        )

        optimizer_list.append(
            smlp.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        )
        optimizer_list.append(smlp.optimizers.RMSprop(momentum=0.9))
        optimizer_list.append(smlp.optimizers.Adam())

        # model.compile(optimizer=optimizer, loss="binaryCrossentropy", metrics=['accuracy', 'mse'])
        for model, optimizer in zip(model_list, optimizer_list):
            model.compile(
                optimizer=optimizer,
                loss="binaryCrossentropy",
                metrics=["accuracy"],
                # metrics=["accuracy", "mse"],
            )

        # model.summary()

        # es = EarlyStopping(monitor="val_accuracy", patience=100, start_from_epoch=500)
        es = smlp.callbacks.EarlyStopping(
            monitor="val_loss", patience=100, start_from_epoch=500
        )

        fit_history_list = []
        for model in model_list:
            fit_history_list.append(
                model.fit(
                    x_train,
                    y_train,
                    validation_data=(x_valid, y_valid),
                    batch_size=200,
                    epochs=5000,
                    callbacks=[es],
                    standard_scaler=True,
                )
            )

        keys = [key for key in fit_history_list[0].history.keys() if "val_" not in key]
        n_graph = len(keys)

        w = 2
        h = n_graph // w + (1 if n_graph % w != 0 else 0)
        fig, axs = plt.subplots(h, w)

        # red for train, blue for valid if exist
        # color = ["red", "blue"]

        for i, key in enumerate(keys):

            cur_h, cur_w = divmod(i, w)
            label_name = key[0].upper() + key[1:]

            if h > 1:
                cur_ax = axs[cur_h, cur_w]
            else:
                cur_ax = axs[cur_w]

            cur_ax.set(xlabel="epoch", ylabel=key)

            for fit_history in fit_history_list:
                x_len = fit_history.epoch
                cur_ax.plot(
                    x_len,
                    fit_history.history[key],
                    label=f"Train-set {label_name}",
                )
                if "val_" + key in fit_history.history.keys():
                    cur_ax.plot(
                        x_len,
                        fit_history.history["val_" + key],
                        label=f"Valid-set {label_name}",
                    )
                cur_ax.legend(loc="best", fontsize="x-small")
                cur_ax.grid()

        plt.tight_layout()
        plt.show()

        # with open("model.pkl", "wb") as f:
        #     pickle.dump(model, f)
        # print("model saved as 'model.pkl' to current directory.")

        # with open("scale.pkl", "wb") as f:
        #     pickle.dump(model.get_data_mean_std(), f)
        # print("model scale saved as 'scale.pkl' to current directory.")

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
