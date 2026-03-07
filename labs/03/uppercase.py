#!/usr/bin/env python3
# 964bdfc8-60b0-4398-b837-7c2520532d17
# 4b50a6fb-a4a6-4b30-9879-0b671f941a72
# f5419161-0138-4909-8252-ba9794a63e53
import argparse
import os

import torch
import torchmetrics

import npfl138
npfl138.require_version("2526.3")
from npfl138.datasets.uppercase_data import UppercaseData

# TODO: Set reasonable values for the hyperparameters, especially for
# `alphabet_size`, `batch_size`, `epochs`, and `window`.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", default=35, type=int, help="If given, use this many most frequent chars.")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=15, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=16, type=int, help="Maximum number of threads to use.")
parser.add_argument("--window", default=4, type=int, help="Window size to use.")
parser.add_argument("--hidden_size", default=200, type=int, help="Embedding size to use.")
parser.add_argument("--hidden_layers", default=2, type=int, help="Number of hidden layers.")
parser.add_argument("--dropout", default=0.3, type=float, help="Dropout rate.")
parser.add_argument("--weight_decay", default=0.001, type=float, help="Weight decay rate.")
parser.add_argument("--label_smoothing", default=0.1, type=float, help="Label smoothing rate.")
parser.add_argument("--embedding_size", default=100, type=int, help="Embedding size to use.")

class Dataset(torch.utils.data.Dataset):
    # A dataset must always implement at least `__len__` and `__getitem__`.
    def __init__(self, uppercase_dataset: UppercaseData.Dataset):
        self.windows = uppercase_dataset.windows
        self.labels = uppercase_dataset.labels

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        return self.windows[index], torch.as_tensor(self.labels[index], dtype=torch.float32)

    # When a dataset implements `__getitems__`, this method is used to generate whole batches in a single call.
    # However, `__getitems__` is expected to return a list of items that are later collated together.
    # For maximum speedup, we already return a whole batch from `__getitems__` and implement a trivial `collate`.
    def __getitems__(self, indices):
        indices = torch.as_tensor(indices)
        return self.windows[indices], torch.as_tensor(self.labels[indices], dtype=torch.float32)

    @staticmethod
    def collate(batch):
        return batch


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self._args = args

        # TODO: Implement a suitable model. The inputs are _windows_ of fixed size
        # (`args.window` characters on the left, the character in question, and
        # `args.window` characters on the right), where each character is
        # represented by a `torch.int64` index. To suitably represent the
        # characters, you can:
        # - Convert the character indices into _one-hot encoding_, which you can
        #   achieve by using `torch.nn.functional.one_hot` on the characters,
        #   and then concatenate the one-hot encodings of the window characters.
        # - Alternatively, you can experiment with `torch.nn.Embedding`s (an
        #   efficient implementation of one-hot encoding followed by a Dense layer)
        #   and flattening afterwards, or suitably using `torch.nn.EmbeddingBag`.
        self.E = torch.nn.Embedding(args.alphabet_size,args.embedding_size)
        self.H = torch.nn.ModuleList(
            [
                torch.nn.Linear(args.embedding_size * (args.window*2+1),args.hidden_size),
                torch.nn.Dropout(args.dropout),
                torch.nn.ReLU()
            ]
        )
        for _ in range(args.hidden_layers-1):
            self.H.append(torch.nn.Linear(args.hidden_size,args.hidden_size))
            self.H.append(torch.nn.Dropout(args.dropout))
            self.H.append(torch.nn.ReLU())
        
        self.O = torch.nn.Linear(args.hidden_size,1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass.
        hid = self.E(windows)
        hid = hid.reshape(hid.shape[0],-1)
        for layer in self.H:
            hid = layer(hid)
        return self.O(hid).squeeze(-1)


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    args.logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the data and create windows of integral character indices and integral labels.
    uppercase_data = UppercaseData(args.window, args.alphabet_size)

    train = torch.utils.data.DataLoader(
        Dataset(uppercase_data.train), args.batch_size, collate_fn=Dataset.collate, shuffle=True, pin_memory=True)
    dev = torch.utils.data.DataLoader(
        Dataset(uppercase_data.dev), args.batch_size, collate_fn=Dataset.collate, pin_memory=True)
    test = torch.utils.data.DataLoader(
        Dataset(uppercase_data.test), args.batch_size, collate_fn=Dataset.collate, pin_memory=True)

    # TODO: Implement a suitable model, optionally including regularization, select
    # good hyperparameters, and train the model.
    model = Model(args)
    params =[
        {
            "params": [(name,params)  for name,params in model.named_parameters() if "bias" not in name],
            "weight_decay": args.weight_decay
        },
        {
            "params": [(name,params)  for name,params in model.named_parameters() if "bias" in name],
            "weight_decay": 0
        }
    ] 
    optim = torch.optim.AdamW(
        params
        )
    lr = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=len(train)*args.epochs,eta_min=1e-5)
    loss = torch.nn.BCEWithLogitsLoss()
    metrics = {"accuracy": torchmetrics.Accuracy("binary")}
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.configure(
        optimizer=optim,
        scheduler=lr,
        loss=loss, 
        metrics=metrics,
        logdir=args.logdir,
        device=device
        )
    model.fit(train, dev=dev, epochs=args.epochs, log_graph=True)

    # TODO: Generate correctly capitalized test set and write the result to `predictions_file`,
    # which is by default `uppercase_test.txt` in the `args.logdir` directory).
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "uppercase_test.txt"), "w", encoding="utf-8") as predictions_file:
        # We start by generating the network test set predictions; if you modified the `test` dataloader
        # or your model does not process the dataset windows, you might need to adjust the following line.
        predictions = model.predict(test, data_with_labels=True)

        # Now you need to utilize the network predictions and the unannotated test data (lowercased text)
        # available in `uppercase_data.test.text` to produce capitalized text and print it to the `predictions_file`.
        for p,t in zip(predictions,uppercase_data.test.text):
            if torch.sigmoid(p) > 0.5:
                predictions_file.write(t.upper())
            else:
                predictions_file.write(t)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
