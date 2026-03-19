#!/usr/bin/env python3
# 964bdfc8-60b0-4398-b837-7c2520532d17
# 4b50a6fb-a4a6-4b30-9879-0b671f941a72
# f5419161-0138-4909-8252-ba9794a63e53
import argparse
import os

import numpy as np
import torch
from torchvision.transforms import v2
import torchmetrics
import npfl138
npfl138.require_version("2526.4")
from npfl138.datasets.cifar10 import CIFAR10

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=16, type=int, help="Maximum number of threads to use.")
parser.add_argument("--num_blocks",default=3, type=int)
parser.add_argument("--block-size",default=2, type=int)
parser.add_argument("--filters-first", default=32, type=int)
parser.add_argument("--dropout_last", default=0.3, type=float)
parser.add_argument("--augment", default=True, type=bool)
parser.add_argument("--hidden-size", default=256, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--lr_final", default=0.0001, type=float)
parser.add_argument("--weight-decay", default=0.0001, type=float)


C = 3
H = 32
W = 32
OUT_CLASSES = 10

class Model(npfl138.TrainableModule):

    def _create_block(self, in_channels, channels, block_size):
        modules = torch.nn.ModuleList()
        modules.append(torch.nn.Conv2d(in_channels, channels, kernel_size=1, padding="same"))
        for i in range(block_size):
            modules.append(torch.nn.Conv2d(channels, channels, kernel_size=3, padding="same"))
            modules.append(torch.nn.ReLU())
            modules.append(torch.nn.BatchNorm2d(channels))
        modules.append(torch.nn.MaxPool2d(kernel_size=2))
        return modules
    
    def _create_conv_network(self,args):
        modules = torch.nn.ModuleList()
        resized_modules = torch.nn.ModuleList()
        for i in range(args.num_blocks):
            channels = args.filters_first * (2**i)
            in_channels = int(channels / 2) if i != 0 else C
            modules.append(self._create_block(in_channels, channels, args.block_size))
        return modules

    def __init__(self, args, device:str):
        super().__init__()
        self.conv_blocks = self._create_conv_network(args)
        S = H / (2**args.num_blocks)
        self.hidden = torch.nn.Linear(int(S*S*args.filters_first*(2**(args.num_blocks-1))), args.hidden_size)
        self.dropout = torch.nn.Dropout(args.dropout_last)
        self.output = torch.nn.Linear(args.hidden_size, OUT_CLASSES)

        self.to(device)
    
    def forward(self, x):
        for block in self.conv_blocks:
            x = block[0](x)
            f = x
            for lay in block[1:-1]:
                f = lay(f)
            h = f + x
            x = block[-1](h)
        x = torch.flatten(x, 1)
        x = self.hidden(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        return x

class TransformedDataset(npfl138.TransformedDataset):
    def __init__(self, dataset: CIFAR10.Dataset, augmentation_fn=None) -> None:
        super().__init__(dataset)
        self._augmentation_fn = augmentation_fn

    def transform(self, example: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        img = example["image"].to(dtype=torch.float32) / 255
        upd = self._augmentation_fn(img) if self._augmentation_fn else img
        return upd, example["label"]

def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()
    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the data.
    cifar = CIFAR10()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    

    # TODO: Create the model and train it.
    model = Model(args, device)
    

    if args.augment:
        # Construct a sequence of augmentation transformations from `torchvision.transforms.v2`.
        augmentation_fn = v2.Compose([
            # TODO: Add the following transformations:
            # - first create a `v2.RandomResize` that scales the image to
            v2.RandomResize(28,36),
            #   random size in range [28, 36],
            # - then add `v2.Pad` that pads the image with 4 pixels on each side,
            v2.Pad(4),
            # - then add `v2.RandomCrop` that chooses a random crop of size 32x32,
            v2.RandomCrop(32),
            # - and finally add `v2.RandomHorizontalFlip` that uniformly
            #   randomly flips the image horizontally.
            v2.RandomHorizontalFlip()
            
        ])
    else:
        augmentation_fn = None
    
    train_dataset = TransformedDataset(cifar.train, augmentation_fn)
    train_loader = train_dataset.dataloader(batch_size=args.batch_size, shuffle=True)
    dev_dataset = TransformedDataset(cifar.dev, None)
    dev_loader = dev_dataset.dataloader(batch_size=args.batch_size, shuffle=False)
    
    test_set = TransformedDataset(cifar.test, None)
    test = test_set.dataloader(batch_size=args.batch_size, shuffle=False)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs*len(train_loader), eta_min=args.lr_final)
    model.configure(
        optimizer=optim,
        loss=torch.nn.CrossEntropyLoss(),
        metrics={"accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=OUT_CLASSES)},
        scheduler=scheduler,
    )
    model.fit(train_loader, dev=dev_loader, epochs=args.epochs, log_graph=True)

    # Generate test set annotations, but in `logdir` to allow parallel execution.
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Perform the prediction on the test data. The line below assumes you have
        # a dataloader `test` where the individual examples are `(image, target)` pairs.
        for prediction in model.predict(test, data_with_labels=True):
            print(prediction.argmax().item(), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
