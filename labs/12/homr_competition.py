#!/usr/bin/env python3
import argparse
import os

import torch
import torchmetrics
import torchvision.transforms as v2
import numpy as np

import torchaudio.models.decoder

import npfl138
npfl138.require_version("2526.12")
from npfl138.datasets.homr_dataset import HOMRDataset

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")

parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")


class TrainableDataset(torch.utils.data.Dataset):
    def __init__(self, homr: HOMRDataset, target_height: int = 96):
        super().__init__()
        self.homr = homr
        self.target_height = target_height

    def __len__(self) -> int:
        return len(self.homr)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transform(self.homr[index])
    
    def transform(self, example) -> tuple[torch.Tensor, torch.Tensor]:
        image = example["image"]
        image = image.to(torch.float32) / 255
        resize_factor = self.target_height / image.shape[1]
        image = v2.Resize(size=(self.target_height, int(resize_factor * image.shape[2])))(image)
        marks = example["marks"]
        return image, marks
    
    @staticmethod
    def collate(examples):
        images, marks = zip(*examples)
        max_h = max(image.shape[1] for image in images)
        max_w = max(image.shape[2] for image in images)

        padded_images = []
        for image in images:
            pad_h = max_h - image.shape[1]
            pad_w = max_w - image.shape[2]
            padded_image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), value=1.0)
            padded_images.append(padded_image)
        images = torch.stack(padded_images)

        target_lengths = torch.tensor([len(marks) for marks in marks], dtype=torch.long)
        targets = torch.cat(marks).to(torch.long)
        return images, (targets, target_lengths)



class Model(npfl138.TrainableModule):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
        )

        self.rnn = torch.nn.GRU(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.classifier = torch.nn.Linear(64 * 2, HOMRDataset.MARKS)

        self.to(self.device)

        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x:torch.Tensor = self.cnn(images)
        x = x.mean(dim=2)  # Global average pooling over height dimension -> (batch_size, channels, width)
        x = x.permute(0, 2, 1)   # Reshape to (batch_size, width, channels)

        x, _ = self.rnn(x)
        x = self.classifier(x)

        return x
    
    def compute_loss(self, y_pred, y_true, images):
        targets, target_lengths = y_true
        N, T, C = y_pred.shape
        input_lengths = torch.full((N,), T, dtype=torch.long, device=y_pred.device)
        log_probs = torch.nn.functional.log_softmax(y_pred, dim=-1).permute(1, 0, 2)
        return torch.nn.functional.ctc_loss(
            log_probs,
            targets.to(y_pred.device),
            input_lengths.to(y_pred.device),
            target_lengths.to(y_pred.device),
            blank=HOMRDataset.MARKS_VOCAB.PAD,
            zero_infinity=True,
        )

    def ctc_decode(self, y_pred: torch.Tensor) -> list[torch.Tensor]:
        decoder = torchaudio.models.decoder.cuda_ctc_decoder(
            tokens=HOMRDataset.MARK_NAMES,
            nbest=1,
            beam_size=10,
        )
        N, T, C = y_pred.shape
        input_lengths = torch.full((N,), T, dtype=torch.int32, device=y_pred.device)
        log_probs = torch.nn.functional.log_softmax(y_pred, -1)
        decoded = decoder(log_probs, input_lengths)
        return [dec[0].tokens for dec in decoded]
    
    def unpack_targets(self, targets, lengths):
        gold = []
        off = 0
        for l in lengths.tolist():
            gold.append(targets[off:off + l].tolist())
            off += l
        return gold

    def compute_metrics(self, y_pred: torch.Tensor, y_true, images=None) -> dict:
        targets, target_lengths = y_true
        pred_ids = self.ctc_decode(y_pred)               # list[list[int]]
        gold_ids = self.unpack_targets(targets, target_lengths)
        pred_str = [HOMRDataset.MARKS_VOCAB.strings(s) for s in pred_ids]
        gold_str = [HOMRDataset.MARKS_VOCAB.strings(s) for s in gold_ids]
        self.metrics["edit_distance"].update(pred_str, gold_str)
        return self.metrics
        
    


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the data. The individual examples are dictionaries with the keys:
    # - "image", a `[1, HEIGHT, WIDTH]` tensor of `torch.uint8` values in [0-255] range,
    # - "marks", a `[num_marks]` tensor with indices of marks on the image.
    # Using `decode_on_demand=True` loads just the raw dataset (~500MB of undecoded PNG images)
    # and then decodes them on every access. Using `decode_on_demand=False` decodes the images
    # during loading, resulting in much faster access, but requires ~5GB of memory.
    homr = HOMRDataset(decode_on_demand=True)

    train = TrainableDataset(homr.train)
    dev = TrainableDataset(homr.dev)
    test = TrainableDataset(homr.test)

    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=TrainableDataset.collate)
    dev_loader = torch.utils.data.DataLoader(dev, batch_size=args.batch_size, shuffle=False, collate_fn=TrainableDataset.collate)
    test_loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=False, collate_fn=TrainableDataset.collate)

    # TODO: Create the model and train it.
    model = Model()
    
    model.configure(
        optimizer=torch.optim.Adam(model.parameters(), args.lr),
        metrics={"edit_distance": HOMRDataset.EditDistanceMetric(ignore_index=HOMRDataset.MARKS_VOCAB.PAD) },
    )

    model.fit(train_loader, dev=dev_loader, epochs=args.epochs)

    # Generate test set annotations, but in `logdir` to allow parallel execution.
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "homr_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the sequences of recognized marks.
        predictions = model.predict(test_loader)

        for sequence in predictions:
            print(" ".join(HOMRDataset.MARKS_VOCAB.strings(sequence)), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
    #homr = HOMRDataset(decode_on_demand=True)
    #print("count:", len(heights))
    #print("percentiles:", np.percentile(heights, [0,1,5,10,25,50,75,90,95,99,100]))
