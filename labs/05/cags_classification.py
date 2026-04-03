#!/usr/bin/env python3
# 964bdfc8-60b0-4398-b837-7c2520532d17
# 4b50a6fb-a4a6-4b30-9879-0b671f941a72
# f5419161-0138-4909-8252-ba9794a63e53

import argparse
import os

import timm
import torch
import torchmetrics
import torchvision.transforms.v2 as v2

import npfl138
npfl138.require_version("2526.5.2")
from npfl138.datasets.cags import CAGS

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--head_epochs", default=10, type=int, help="Number of epochs to train the head only.")
parser.add_argument("--ft_epochs", default=40, type=int, help="Number of epochs to fine-tune the whole model.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")

parser.add_argument("--head_lr", default=1e-4, type=float, help="Learning rate.")
parser.add_argument("--ft_lr", default=1e-5, type=float, help="Learning rate for fine-tuning.")


class Dataset(npfl138.TransformedDataset):
    def __init__(self, dataset: torch.utils.data.Dataset, augment = False) -> None:
        super().__init__(dataset)
        self.augment = augment

    def transform(self, example):
        if self.augment:
            image = self.augment_image(example["image"])
        else:
            image = example["image"]
        return image, example["label"]
    
    def augment_image(self, image):
        image =v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            v2.RandomErasing(p=0.25)
        ])(image)
        return image


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the data. The individual examples are dictionaries with the keys:
    # - "image", a `[3, 224, 224]` tensor of `torch.uint8` values in [0-255] range,
    # - "mask", a `[1, 224, 224]` tensor of `torch.float32` values in [0-1] range,
    # - "label", a scalar of the correct class in `range(CAGS.LABELS)`.
    # The `decode_on_demand` argument can be set to `True` to save memory and decode
    # each image only when accessed, but it will most likely slow down training.
    cags = CAGS(decode_on_demand=False)

    # Load the EfficientNetV2-B0 model without the classification layer. For an
    # input image, the model returns a tensor of shape `[batch_size, 1280]`.
    efficientnetv2_b0 = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, num_classes=0)

    # Create a simple preprocessing performing necessary normalization.
    preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # The `scale=True` also rescales the image to [0, 1].
        v2.Normalize(mean=efficientnetv2_b0.pretrained_cfg["mean"], std=efficientnetv2_b0.pretrained_cfg["std"]),
    ])

    # TODO: Create the model and train it.
    model = torch.nn.Sequential(
        preprocessing,
        efficientnetv2_b0,
        torch.nn.Linear(1280, CAGS.LABELS),
    )

    backbone = model[1]
    head = model[-1]

    for p in model.parameters():
        p.requires_grad = False

    for p in model[-1].parameters():
        p.requires_grad = True

    model = npfl138.TrainableModule(model)

    train = torch.utils.data.DataLoader(Dataset(cags.train, augment=True), batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(Dataset(cags.dev), batch_size=args.batch_size)
    test = torch.utils.data.DataLoader(Dataset(cags.test), batch_size=args.batch_size)

    optimizer = torch.optim.Adam(head.parameters(), lr=args.head_lr)
    model.configure(
        optimizer=optimizer,
        #scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train)),
        loss=torch.nn.CrossEntropyLoss(),
        metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=CAGS.LABELS)},
    )

    model.fit(train, dev=dev, epochs=args.head_epochs)


    for p in backbone.parameters():
        p.requires_grad = True

    optimizer2 = torch.optim.AdamW([
            {"params": backbone.parameters(), "lr": args.ft_lr}, 
            {"params": head.parameters(), "lr": args.head_lr},
            ], weight_decay=2e-4)

    model.configure(
        optimizer=optimizer2,
        #scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=args.ft_epochs * len(train)),
        loss=torch.nn.CrossEntropyLoss(label_smoothing=0.1),
        metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=CAGS.LABELS)},
    )

    model.fit(train, dev=dev, epochs=args.ft_epochs)



    # Generate test set annotations, but in `logdir` to allow parallel execution.
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "cags_classification.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Perform the prediction on the test data. The line below assumes you have
        # a dataloader `test` where the individual examples are `(image, target)` pairs.
        for prediction in model.predict(test, data_with_labels=True):
            print(prediction.argmax().item(), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
