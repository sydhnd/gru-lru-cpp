import os
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch import optim

import export.export_gru as export
import common.data_process_util as dp
from common.windowed_sensor_dataset import SensorWindowDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Expected CSV format: Kaggle oil/gas anomaly detection dataset.
DATA_FILE = PROJECT_ROOT / "data" / "plant_sensor_data.csv"
EXPORT_FILE = PROJECT_ROOT / "native" / "model" /"grutest.bin"


HIDDEN_SIZE = 128
WINDOW_SIZE = 300
BATCH_SIZE = 32
FEATURE_SIZE = 7
TRAIN_SPLIT = 0.8
POS_WEIGHT = 18.675   # from calculation
LEARNING_RATE = 0.001
EPOCHS = 1


# from my other github project
# https://github.com/sydhnd/parallel-lru
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        gru_output, h_n = self.gru(x)
        final_hidden = h_n[-1]

        logits = self.out(final_hidden)
        return logits


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #for benchmarking 
    #device = "cpu"
    print("Using device:", device)
    if (device=="cuda"):
          torch.cuda.init()

    df = pd.read_csv(DATA_FILE)
    features, labels, _ = dp.prep_data(df)

    split_index = int(len(features) * TRAIN_SPLIT)

    x_train = features[:split_index]
    y_train = labels[:split_index]



    train_loader = dp.create_loader(
        x_train,
        y_train,
        shuffle=True,
        window_size=WINDOW_SIZE,
    )

    model = GRUModel(
        input_size=FEATURE_SIZE,
        hidden_size=HIDDEN_SIZE,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([POS_WEIGHT], device=device),
    )

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    start_time = time.time()
    for epoch in range(EPOCHS):
        model.train()

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        for batch_idx, (batch_features, batch_labels) in enumerate(train_loader):
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            output = model(batch_features)
            loss = criterion(
                output.squeeze(-1),
                batch_labels.float().squeeze(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    elapsed_time = time.time() - start_time
    print(f"\nTraining time: {elapsed_time:.2f}s")
    export.export_gru_binary(model,EXPORT_FILE)

if __name__ == "__main__":
    main()