#!/usr/bin/env python3
#f5419161-0138-4909-8252-ba9794a63e53
#4b50a6fb-a4a6-4b30-9879-0b671f941a72
#964bdfc8-60b0-4398-b837-7c2520532d17

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import npfl138
npfl138.require_version("2526.11")
from npfl138.datasets.modelnet import ModelNet


# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--modelnet", default=20, type=int, help="ModelNet dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--dropout", default=0.3, type=float, help="Dropout rate.")

class Model(npfl138.TrainableModule):
    def __init__(self, resolution:int, args: argparse.Namespace) -> None:
        super().__init__()
        # Trying three blocks of 3D convolutions
        # kernel_size=3 perform usually best (saving even more memory in 3D) (https://www.youtube.com/watch?v=V9ZYDCnItr0)
        self.conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2), 
            
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2), 
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )
        
        # size of the flattened features.
        flat_dim = resolution // 4
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (flat_dim**3), 256),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(256, ModelNet.LABELS) 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Passing input through conv layers and then the fully connected head
        return self.fc(self.conv(x))

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, *args) -> torch.Tensor:
        return F.cross_entropy(y_pred, y_true)

    def compute_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor, *args) -> dict[str, torch.Tensor]:
        self.metrics["accuracy"].update(y_pred, y_true)
        return self.metrics

def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    def train_one(res):
        print(f"\n--- Training Model {res}^3 ---")
        data = ModelNet(res)
        m = Model(res, args)
        m.configure(
            optimizer=torch.optim.AdamW(m.parameters()),
            metrics={"accuracy": npfl138.metrics.CategoricalAccuracy()},
            device=device,
        )

        train_l = DataLoader(TensorDataset(data.train.data["grids"].to(torch.float32), data.train.data["labels"].to(torch.long)), batch_size=args.batch_size, shuffle=True)
        dev_l = DataLoader(TensorDataset(data.dev.data["grids"].to(torch.float32), data.dev.data["labels"].to(torch.long)), batch_size=args.batch_size)
        test_l = DataLoader(TensorDataset(data.test.data["grids"].to(torch.float32), data.test.data["labels"].to(torch.long)), batch_size=args.batch_size)
        
        m.fit(train_l, epochs=args.epochs, dev=dev_l)
        return m, test_l

    model20, test20 = train_one(20)
    model32, test32 = train_one(32)

    os.makedirs(logdir, exist_ok=True)
    model20.eval()
    model32.eval()

    with open(os.path.join(logdir, "3d_recognition.txt"), "w", encoding="utf-8") as predictions_file:
        with torch.no_grad():
            # Zipping test loaders 
            for (x20, _), (x32, _) in zip(test20, test32):

                p20 = F.softmax(model20(x20.to(device)), dim=-1)
                p32 = F.softmax(model32(x32.to(device)), dim=-1)
                
                # Weighted Soft-Voting 
                # p20 model: Epoch 20/20 8.0s loss=0.0521 accuracy=0.9793 dev:loss=0.5004 dev:accuracy=0.9158 
                # p32 model: Epoch 20/20 28.3s loss=0.0332 accuracy=0.9863 dev:loss=0.2677 dev:accuracy=0.9487  
                p_ensemble = 0.3 * p20 + 0.7 * p32
                
                for pred in p_ensemble.argmax(dim=-1):
                    print(pred.item(), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
