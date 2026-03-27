#!/usr/bin/env python3
# 964bdfc8-60b0-4398-b837-7c2520532d17
# 4b50a6fb-a4a6-4b30-9879-0b671f941a72
# f5419161-0138-4909-8252-ba9794a63e53
import argparse
import os

import numpy as np
import timm
import torch
import torchvision.transforms.v2 as v2
from torchvision import tv_tensors
import npfl138
npfl138.require_version("2526.5.2")
from npfl138.datasets.cags import CAGS

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=22, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=16, type=int, help="Maximum number of threads to use.")
parser.add_argument("--plus", default=True, help="Whether to sum output features with upsampled 'input features' from backbone")
parser.add_argument("--block_size", default=1, type=int, help="Number of convolutional layers in each block")
parser.add_argument("--final_convs", default=3, type=int, help="Number of convolutional layers in the final block")
parser.add_argument("--final_channels", default=64, type=int, help="Number of channels in the final block")
parser.add_argument("--ft", default=17,type=int, help="Finetune the model")
parser.add_argument("--lr", default=0.001,type=float, help="Learning rate")
parser.add_argument("--fin_lr", default=1e-4,type=float, help="Finetune learning rate")
parser.add_argument("--ft_lr", default=1e-4,type=float, help="Finetune learning rate")
parser.add_argument("--ft_fin_lr", default=1e-5,type=float, help="Finetune learning rate")
def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the data. The individual examples are dictionaries with the keys:
    # - "image", a `[3, 224, 224]` tensor of `torch.uint8` values in [0-255] range,
    # - "mask", a `[1, 224, 224]` tensor of `torch.float32` values in [0-1] range,
    # - "label", a scalar of the correct class in `range(CAGS.LABELS)`.
    # The `decode_on_demand` argument can be set to `True` to save memory and decode
    # each image only when accessed, but it will most likely slow down training.
    cags = CAGS(decode_on_demand=False)

    # Load the EfficientNetV2-B0 model without the classification layer.
    # Apart from calling the model as in the classification task, you can call it using
    #   output, features = efficientnetv2_b0.forward_intermediates(batch_of_images)
    # obtaining (assuming the input images have 224x224 resolution):
    # - `output` is a `[N, 1280, 7, 7]` tensor with the final features before global average pooling,
    # - `features` is a list of intermediate features with resolution 112x112, 56x56, 28x28, 14x14, 7x7.
    efficientnetv2_b0 = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, num_classes=0)

    # Create a simple preprocessing performing necessary normalization.
    preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # The `scale=True` also rescales the image to [0, 1].
        v2.Normalize(mean=efficientnetv2_b0.pretrained_cfg["mean"], std=efficientnetv2_b0.pretrained_cfg["std"]),
    ])
    class SegmentBlock(torch.nn.Module):
        def __init__(self, block_size, in_ch, out_ch, plus:bool = True):
            super().__init__()
            self.up = torch.nn.ConvTranspose2d(in_ch,out_ch, 2, 2, padding=0)
            layers = torch.nn.ModuleList()
            self.plus = plus
            for _ in range(block_size):
                layers.append(torch.nn.BatchNorm2d(out_ch))
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Conv2d(out_ch, out_ch, kernel_size=3, padding="same"))
            self.layers = torch.nn.Sequential(*layers)
        
        def forward(self, x):
            up = self.up(x)
            y = self.layers(up)
            if self.plus:
                y = y + up
            return y

    class SegmentationModel(npfl138.TrainableModule):
        def __init__(self, backbone, outputs:list[int],args):
            super().__init__()
            self.backbone = backbone
            self.blocks = torch.nn.ModuleList()
            ch_sorted = outputs[::-1]
            self.blocks.append(SegmentBlock(args.block_size,ch_sorted[0],ch_sorted[1],args.plus))
            for i in range(1,len(outputs)-1):
                self.blocks.append(SegmentBlock(args.block_size, ch_sorted[i]*2, ch_sorted[i+1], args.plus))
            self.blocks.append(SegmentBlock(args.block_size, ch_sorted[-1]*2, 12, args.plus))
            self.final_layers = torch.nn.ModuleList()
            ch = 15
            for i in range(args.final_convs-1):
                self.final_layers.append(torch.nn.BatchNorm2d(ch))
                self.final_layers.append(torch.nn.ReLU())
                self.final_layers.append(torch.nn.Conv2d(ch, args.final_channels, kernel_size=3, padding="same"))
                ch = args.final_channels
            self.final_layers.append(torch.nn.Conv2d(ch, 1, kernel_size=1))

            self.preprocessing = v2.Compose([
                v2.ToDtype(torch.float32, scale=True), 
                v2.Normalize(mean=efficientnetv2_b0.pretrained_cfg["mean"], std=efficientnetv2_b0.pretrained_cfg["std"]),
            ])

        def freeze_backbone(self,freeze:bool = True):
            for p in self.backbone.parameters():
                p.requires_grad = not freeze
                  
        def forward(self, x):
            x = self.preprocessing(x)
            x_in = x
            x, features = self.backbone.forward_intermediates(x)
            

            for i in range(len(self.blocks)):
                if i == 0:
                    x = self.blocks[i](x)
                else:
                    ftrs = features[-i-1]
                    x = self.blocks[i](torch.cat([x, ftrs], dim=1))
            x = torch.cat([x, x_in], dim=1)
            for layer in self.final_layers:
                x = layer(x)
            return torch.sigmoid(x).squeeze(1)

    class Dataset(npfl138.TransformedDataset):
        def __init__(self, dataset: torch.utils.data.Dataset, augment = False) -> None:
            super().__init__(dataset)
            self.augment = augment
            self._transform = v2.Compose([
                v2.RandomHorizontalFlip(),
                v2.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                v2.RandomErasing(p=0.25)
            ])

        def transform(self, example):
            if self.augment:
                image, mask = self.augment_image(example["image"], example["mask"])
            else:
                image, mask = example["image"], example["mask"]
            return image, mask.squeeze(0)
        
        def augment_image(self, image, mask):
            mask = tv_tensors.Mask(mask)
            image, mask = self._transform(image, mask)
            return image, mask

    # TODO: Create the model and train it.
    model = SegmentationModel(efficientnetv2_b0, [16, 32, 48,112,1280], args)
    train_set = Dataset(cags.train, augment=True)
    dev_set = Dataset(cags.dev)
    test_set = Dataset(cags.test)
    train = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(dev_set, batch_size=args.batch_size)
    test = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size)

    def config(model:npfl138.TrainableModule,args, finetune:bool = False):
        model.freeze_backbone(freeze = not finetune)
        if finetune:
            optimizer = torch.optim.AdamW([
                {"params": model.backbone.parameters(), "lr": args.ft_lr},
                {"params": model.blocks.parameters(), "lr": args.ft_lr},
                {"params": model.final_layers.parameters(), "lr": args.ft_lr},
            ], weight_decay=2e-4)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train)*args.ft, eta_min=args.ft_fin_lr)

        else:
            optimizer = torch.optim.AdamW([
                {"params": model.blocks.parameters(), "lr": args.lr},
                {"params": model.final_layers.parameters(), "lr": args.lr},
            ], weight_decay=2e-4)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train)*args.epochs, eta_min=args.fin_lr)
        model.configure(
            optimizer=optimizer,
            loss=torch.nn.BCELoss(),
            metrics={"iou": npfl138.metrics.MaskIoU((224,224),False)},
            scheduler=sch,
            device=device,
        )
            
    config(model,args,False)
    model.fit(train,epochs=args.epochs,dev=dev)
    config(model,args,True)
    model.fit(train,epochs=args.ft,dev=dev)

    # Generate test set annotations, but in `logdir` to allow parallel execution.
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "cags_segmentation.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Perform the prediction on the test data. The line below assumes you have
        # a dataloader `test` where the individual examples are `(image, target)` pairs.
        for mask in model.predict(test, data_with_labels=True, as_numpy=True):
            zeros, ones, runs = 0, 0, []
            for pixel in np.reshape(mask >= 0.5, [-1]):
                if pixel:
                    if zeros or (not zeros and not ones):
                        runs.append(zeros)
                        zeros = 0
                    ones += 1
                else:
                    if ones:
                        runs.append(ones)
                        ones = 0
                    zeros += 1
            runs.append(zeros + ones)
            print(*runs, file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
