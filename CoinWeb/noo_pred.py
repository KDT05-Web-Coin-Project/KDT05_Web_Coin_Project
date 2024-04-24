import copy
from pathlib import Path
import warnings
from sqlalchemy import create_engine, Connection, text
import matplotlib.pyplot as plt

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss

warnings.filterwarnings("ignore")

engine = create_engine(
    "mysql+pymysql://root:kdt5@1.251.203.204:33065/Team2?charset=utf8mb4"
)

# ----------------- Data Preparation -----------------
with engine.begin() as conn:
    pass



data = pd.read_csv("./DATA/data_with_btc_scaled.csv")


def make_data(data):
    data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")
    data["time_idx"] = (data["date"].dt.year * 12 + data["date"].dt.month) * 30 + data["date"].dt.day
    data["time_idx"] -= data["time_idx"].min()
    data["time_idx"] = data["time_idx"].astype(int)
    data["month"] = data.date.dt.month.astype(str).astype("category")
    data["group"] = 0

# ----------------- Data Preparation -----------------


def datetimes(data):
    # return datetime as list
    return data["date"].dt.strftime("%Y-%m-%d").tolist()


def actual(data):
    # return actual price as list
    return data["btc"].tolist()


# ----------------- DataSet Preparation -----------------
max_encoder_length = 300
max_prediction_length = 30
# training_cutoff = data["time_idx"].max() - max_prediction_length


def predict(data, max_encoder_length, max_prediction_length):
    best_model_path = "lightning_logs\\version_0\\checkpoints\\epoch=29-step=510.ckpt"
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    # add time index : datetime to int
    data["time_idx"] = (data["date"].dt.year * 12 + data["date"].dt.month) * 30 + data[
        "date"
    ].dt.day
    data["time_idx"] -= data["time_idx"].min()
    # add additional features
    data["month"] = data.date.dt.month.astype(str).astype(
        "category"
    )  # categories have be strings

    # select last 24 months from data (max_encoder_length is 24)
    encoder_data = data[lambda x: x.time_idx > x.time_idx.max() - max_encoder_length]

    last_data = data[lambda x: x.time_idx == x.time_idx.max()]
    decoder_data = pd.concat(
        [
            last_data.assign(date=lambda x: x.date + pd.offsets.DateOffset(day=i))
            for i in range(1, max_prediction_length + 1)
        ],
        ignore_index=True,
    )

    # add time index consistent with "data"
    decoder_data["time_idx"] = (
        decoder_data["date"].dt.year * 12 + decoder_data["date"].dt.month
    ) * 30 + decoder_data["date"].dt.day
    decoder_data["time_idx"] += (
        encoder_data["time_idx"].max() + 1 - decoder_data["time_idx"].min()
    )

    # adjust additional time feature(s)
    decoder_data["month"] = decoder_data.date.dt.month.astype(str).astype(
        "category"
    )  # categories have be strings

    # combine encoder and decoder data
    new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)

    new_raw_predictions = best_tft.predict(
        new_prediction_data, mode="raw", return_x=True
    )

    dates = new_prediction_data["date"][::36].dt.strftime("%Y-%m").to_list()

    pred_list = []
    for i in range(30):
        # print(new_raw_predictions.output.prediction[0][i][3])
        pred_list.append(new_raw_predictions.output.prediction[0][i][3])

    # tensor -> numpy -> list
    pred_list = [pred_list[i].numpy().tolist() for i in range(len(pred_list))]

    return pred_list


if __name__ == "__main__":
    print(datetimes(data))
    print(actual(data))
    print(predict(data, max_encoder_length, max_prediction_length))
