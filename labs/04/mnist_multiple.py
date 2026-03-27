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
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


# Create a dataset from consecutive _pairs_ of original examples, assuming
# that the size of the original dataset is even.
class DatasetOfPairs(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset):
        self._dataset = dataset

    def __len__(self):
        # TODO: The new dataset has half the size of the original one.
        return len(self._dataset) //2

    def __getitem__(self, index: int) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        # TODO: Given an `index`, generate an example composed of two input examples.
        # Notably, considering examples `self._dataset[2 * index]` and `self._dataset[2 * index + 1]`,
        # each being a dictionary with keys "image" and "label", return a pair `(input, output)` with
        # - `input` being a pair of images, each converted to `torch.float32` and divided by 255,
        # - `output` being a pair of labels.
        ex1, ex2 = self._dataset[2*index], self._dataset[2*index + 1]
        i1 = ex1["image"].to(dtype=torch.float32) / 255
        i2 = ex2["image"].to(dtype=torch.float32) / 255
        return (i1, i2), (ex1["label"], ex2["label"])


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        # TODO: Create all layers required to implement the forward pass.
        self.c1 = torch.nn.Conv2d(1,10,3,2,"valid")
        self.c2 = torch.nn.Conv2d(10,20,3,2,"valid")
        self.flatten = torch.nn.Flatten()
        self.l1 = torch.nn.Linear(36*20,200)

        self.l_1_2 = torch.nn.Linear(400,200)
        self.o_1 = torch.nn.Linear(200,1)

        self.o_2 = torch.nn.Linear(200,10)


    def forward(
        self, first: torch.Tensor, second: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO: Implement the forward pass of the model using the layers created in the constructor.
        #
        # The model starts by passing each input image through the same
        # module (with shared weights), which should perform

        # - convolution with 10 filters, 3x3 kernel size, stride 2, "valid" padding, ReLU activation,
        i1 = torch.relu(self.c1(first))
        i2 = torch.relu(self.c1(second))
        # - convolution with 20 filters, 3x3 kernel size, stride 2, "valid" padding, ReLU activation,
        i1 = torch.relu(self.c2(i1))
        i2 = torch.relu(self.c2(i2))
        # - flattening layer,
        f1, f2 = self.flatten(i1), self.flatten(i2)
        # - fully connected layer with 200 neurons and ReLU activation,
        l1, l2 = torch.relu(self.l1(f1)), torch.relu(self.l1(f2))
        # obtaining a 200-dimensional feature vector of each image.
        #
        # Using the computed representations, the model should produce four outputs:
        # - first, compute _direct comparison_, a tensor of bools indicating whether
        #   the first digit is greater than the second, by
        #   - concatenating the two 200-dimensional image feature vectors,
        h = torch.concat((l1,l2),dim=1)

        #   - processing them using another 200-neuron ReLU linear layer,
        h = torch.relu(self.l_1_2(h))
        #   - computing one output using a linear layer and the **sigmoid** activation;
        o = torch.sigmoid(self.o_1(h))
        # - then, classify the computed representation FV of the first image using
        #   a linear layer into 10 classes;
        # - then, classify the computed representation FV of the second image using
        #   the same layer (identical, i.e., with shared weights) into 10 classes;
        # - finally, compute _indirect comparison_, a tensor of bools indicating
        #   whether the first digit is greater than the second by comparing the
        #   most probable digits predicted by the above two outputs.
        direct_comparison = o
        digit_1 = self.o_2(l1)
        digit_2 = self.o_2(l2)
        indirect_comparison = torch.argmax(digit_1,1) > torch.argmax(digit_2,1)

        return direct_comparison, digit_1, digit_2, indirect_comparison

    def compute_loss(self, y_pred, y_true, *inputs):
        # The `compute_loss` method can override the loss computation of the model.
        # It is needed when there are multiple model outputs or multiple losses to compute.
        # We start by unpacking the multiple outputs of the model and the multiple targets.
        direct_comparison_pred, digit_1_pred, digit_2_pred, indirect_comparison_pred = y_pred
        digit_1_true, digit_2_true = y_true

        # TODO: Compute the required losses using their implementations from `torch.nn`.
        # Note that the `direct_comparison_pred` is really a probability (sigmoid was applied),
        # while the `digit_1_pred` and `digit_2_pred` are logits of 10-class classification.
        direct_comparison_loss = torch.nn.functional.binary_cross_entropy(direct_comparison_pred.flatten(),(digit_1_true > digit_2_true).to(torch.float32))
        t1 = torch.nn.functional.one_hot(digit_1_true.to(torch.long), 10).to(torch.float32)
        t2 = torch.nn.functional.one_hot(digit_2_true.to(torch.long), 10).to(torch.float32)
        digit_1_loss = torch.nn.functional.cross_entropy(digit_1_pred, t1)
        digit_2_loss = torch.nn.functional.cross_entropy(digit_2_pred, t2)

        return direct_comparison_loss + digit_1_loss + digit_2_loss

    def compute_metrics(self, y_pred, y_true, *inputs):
        # The `compute_metrics` can override metric computation for the model. We start by
        # unpacking the multiple outputs of the model and the multiple targets.
        direct_comparison_pred, digit_1_pred, digit_2_pred, indirect_comparison_pred = y_pred
        digit_1_true, digit_2_true = y_true
        true = digit_1_true > digit_2_true

        # TODO: Update two metrics -- the `direct_comparison` and the `indirect_comparison`.
        self.metrics["direct_comparison"].update(direct_comparison_pred.flatten(), true)
        self.metrics["indirect_comparison"].update(indirect_comparison_pred.flatten(), true)

        # Finally, we return the dictionary of all the metric values.
        return {name: metric for name, metric in self.metrics.items()}


def main(args: argparse.Namespace) -> dict[str, float]:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads, args.recodex)
    npfl138.global_keras_initializers()

    # Load the data and create dataloaders.
    mnist = MNIST()

    train = torch.utils.data.DataLoader(DatasetOfPairs(mnist.train), batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(DatasetOfPairs(mnist.dev), batch_size=args.batch_size)

    # Create the model and train it.
    model = Model(args)

    model.configure(
        optimizer=torch.optim.Adam(model.parameters()),
        metrics={
            # TODO: Create two binary accuracy metrics using `torchmetrics.Accuracy`:
            "direct_comparison": torchmetrics.Accuracy("binary"),
            "indirect_comparison": torchmetrics.Accuracy("binary")
        },
        logdir=npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args)),
    )

    logs = model.fit(train, dev=dev, epochs=args.epochs)

    # Return development metrics for ReCodEx to validate.
    return {metric: value for metric, value in logs.items() if metric.startswith("dev:")}


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
