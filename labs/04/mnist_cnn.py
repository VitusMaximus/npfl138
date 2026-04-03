#!/usr/bin/env python3
# 964bdfc8-60b0-4398-b837-7c2520532d17
# 4b50a6fb-a4a6-4b30-9879-0b671f941a72
# f5419161-0138-4909-8252-ba9794a63e53
import argparse

import torch
import torchmetrics

import npfl138
npfl138.require_version("2526.4")
from npfl138.datasets.mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--cnn", default=None, type=str, help="CNN architecture.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Dataset(npfl138.TransformedDataset):
    def transform(self, example):
        image = example["image"]  # a torch.Tensor with torch.uint8 values in [0, 255] range
        image = image.to(torch.float32) / 255  # image converted to float32 and rescaled to [0, 1]
        label = example["label"]  # a torch.Tensor with a single integer representing the label
        return image, label  # return an (input, target) pair¨


def create_model(cnn:str, prev_channels:int = 1) -> torch.nn.ParameterList:
    layers = cnn.split(",")
    modules = torch.nn.ModuleList()
    end = False
    in_res = False
    H = MNIST.H
    for layer in layers:
        if in_res and layer[-1] == "]":
            end = True
            layer = layer[:-1]

        data = layer.split("-")
        if data[0][0] == "R":
            true_params = modules
            modules = torch.nn.ModuleList()
            in_res=True
            data = data[1:]
            data[0] = data[0][1:]
        
        match data[0][0]:
            case "C":
                ch = int(data[1])
                batch_norm = data[0][-1] == "B"
                pad = data[4]
                if pad == "same":
                    H = int(H/int(data[3]))
                else:
                    H = int((H- int(data[2])) / int(data[3])) + 1
                modules.append(
                    torch.nn.Conv2d(prev_channels, ch, int(data[2]), stride=int(data[3]), padding=pad, bias=not batch_norm)
                )
                prev_channels = ch
                if batch_norm:
                    modules.append(torch.nn.BatchNorm2d(ch))
                modules.append(torch.nn.ReLU())
            case "M":
                modules.append(torch.nn.MaxPool2d(int(data[1]), stride=int(data[2])))
                H = int((H - int(data[1])) / int(data[2])) + 1
            case "F":
                modules.append(torch.nn.Flatten())
                prev_channels = H * H * prev_channels
            case "H":
                size = int(data[1])
                modules.append(torch.nn.Linear(prev_channels, size))
                modules.append(torch.nn.ReLU())
                prev_channels = size
            case "D":
                modules.append(torch.nn.Dropout(float(data[1])))
            
        if end:
            true_params.append(Residual(modules))
            modules = true_params
            in_res = False
            end = False    
    return modules, prev_channels

class Residual(torch.nn.Module):
    def __init__(self, layers: torch.nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for layer in self.layers:
            y = layer(y)
        return x + y

class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        # TODO: Add CNN layers specified by `args.cnn`, which contains
        # a comma-separated list of the following layers:
        # - `C-channels-kernel_size-stride-padding`: Add a convolutional layer with ReLU
        #   activation and specified number of channels, kernel size, stride and padding.
        # - `CB-channels-kernel_size-stride-padding`: Same as `C`, but use batch normalization.
        #   In detail, start with a convolutional layer **without bias** and activation,
        #   then add a batch normalization layer, and finally the ReLU activation.
        # - `M-pool_size-stride`: Add max pooling with specified size and stride, using
        #   the default padding of 0 (the "valid" padding).
        # - `R-[layers]`: Add a residual connection. The `layers` contain a specification
        #   of at least one convolutional layer (but not a recursive residual connection `R`).
        #   The input to the `R` layer should be processed sequentially by `layers`, and the
        #   produced output (after the ReLU nonlinearity of the last layer) should be added
        #   to the input (of this `R` layer).
        # - `F`: Flatten inputs. Must appear exactly once in the architecture.
        # - `H-hidden_layer_size`: Add a dense layer with ReLU activation and the specified size.
        # - `D-dropout_rate`: Apply dropout with the given dropout rate.
        # You can assume the resulting network is valid; it is fine to crash if it is not.
        #
        # To implement the residual connections, you can use various approaches, for example:
        # - you can create a specialized `torch.nn.Module` subclass representing a residual
        #   connection that gets the inside layers as an argument, and implement its forward call.
        #   This allows you to have the whole network in a single `torch.nn.Sequential`.
        # - you could represent the model module as a `torch.nn.ModuleList` of `torch.nn.Sequential`s,
        #   each representing one user-specified layer, keep track of the positions of residual
        #   connections, and manually perform them in the forward pass.
        #
        # It might be difficult to compute the number of features after the `F` layer. You can
        # nevertheless use the `torch.nn.LazyLinear`, `torch.nn.LazyConv2d`, and `torch.nn.LazyBatchNorm2d`
        # layers, which do not require the number of input features to be specified in the constructor.
        # During `__init__`, these layers do not allocate their parameters, and only do so when
        # they are first called on a tensor, at which point the number of input features is known.
        # During this first call they also change themselves to the corresponding `torch.nn.Linear` etc.

        # TODO: Finally, add the final Linear output layer with `MNIST.LABELS` units.
        
        layers, prev_channels = create_model(args.cnn)
        self.layers = torch.nn.Sequential(*layers)
        self.layers.append(torch.nn.Linear(prev_channels, MNIST.LABELS))
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)


def main(args: argparse.Namespace) -> dict[str, float]:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads, args.recodex)
    npfl138.global_keras_initializers()

    # Load the data and create dataloaders.
    mnist = MNIST()

    train = torch.utils.data.DataLoader(Dataset(mnist.train), batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(Dataset(mnist.dev), batch_size=args.batch_size)

    # Create the model and train it.
    model = Model(args)

    model.configure(
        optimizer=torch.optim.Adam(model.parameters()),
        loss=torch.nn.CrossEntropyLoss(),
        metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=MNIST.LABELS)},
        logdir=npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args)),
    )

    logs = model.fit(train, dev=dev, epochs=args.epochs)

    # Return development metrics for ReCodEx to validate.
    return {metric: value for metric, value in logs.items() if metric.startswith("dev:")}


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
