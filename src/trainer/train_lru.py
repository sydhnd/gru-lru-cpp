import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import export.export_lru as export
import common.data_process_util as dp
from models.lru_model import LRU

# from my other github project
# https://github.com/sydhnd/parallel-lru

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Expected CSV format: Kaggle oil/gas anomaly detection dataset.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "plant_sensor_data.csv"
EXPORT_FILE = PROJECT_ROOT / "native" / "model" /"lru_test_model.bin"


WINDOW_SIZE = 300
BATCH_SIZE = 32
HIDDEN_SIZE = 128
EPOCHS = 1
INPUT_FEATURES = 7
OUTPUT_TARGETS = 1
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.8
POS_WEIGHT = 18.675


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"ROCm compiled? {torch.version.hip is not None}")
    print(f"ROCm version: {torch.version.hip}")
    print(f"Device: {device}")

  
    if device == "cuda":
        torch.cuda.init()

    df = pd.read_csv(DATA_FILE)
    features, labels, _ = dp.prep_data(df)

    split_index = int(len(features) * TRAIN_SPLIT)
    x_train, y_train = features[:split_index], labels[:split_index]

    train_loader = dp.create_loader(x_train, y_train, shuffle=True,window_size=WINDOW_SIZE)

    model = LRU(
        in_features=INPUT_FEATURES,
        hidden_size=HIDDEN_SIZE,
        out_features=OUTPUT_TARGETS,
        window_size=WINDOW_SIZE
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([POS_WEIGHT], device=device)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.float().view(-1, 1).to(device)

            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = criterion(predictions, batch_labels)

            loss.backward()
            optimizer.step()

    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed:.2f}s")
    export.export_lru_ultra(model, EXPORT_FILE)
if __name__ == "__main__":
    main()