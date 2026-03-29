#!/usr/bin/env python3
import argparse
import os

import timm
import torch
import torchvision.transforms.v2 as v2

import bboxes_utils
import npfl138
npfl138.require_version("2526.6")
from npfl138.datasets.svhn import SVHN

from torchvision.ops import batched_nms, sigmoid_focal_loss

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")

parser.add_argument("--lr", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--confidence_threshold", default=0.2, type=float, help="Confidence threshold for predictions.")

class Detector(torch.nn.Module):
    def __init__(self, args, backbone, preprocessing, num_classes = SVHN.LABELS, anchors_per_cell = 1):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args

        self.backbone = backbone
        self.preprocessing = preprocessing
        self.num_classes = num_classes
        self.anchors_per_cell = anchors_per_cell

        # 14 x 14   Last but one feature map
        ch = 112
        self.class_head = torch.nn.Sequential(
            torch.nn.Conv2d(ch, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, (num_classes + 1) * anchors_per_cell, 1)
        )

        self.box_head = torch.nn.Sequential(
            torch.nn.Conv2d(ch, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 4 * anchors_per_cell, 1)
        )

        self.to(self.device)

    def forward(self, x):
        x = self.preprocessing(x)
        _, features = self.backbone.forward_intermediates(x)
        f = features[-2]

        cls = self.class_head(f)
        box = self.box_head(f)

        N, _, H, W = cls.shape
        A = self.anchors_per_cell

        # (N, num_classes * A, H, W) -> (N, H*W*A, num_classes)
        cls = cls.permute(0, 2, 3, 1).reshape(N, H*W*A, self.num_classes + 1)
        box = box.permute(0, 2, 3, 1).reshape(N, H*W*A, 4)

        return cls, box, H, W
    
    def compute_loss(self, predicted_classes, predicted_bboxes, target_classes, target_bboxes):
        #class_loss = torch.nn.functional.cross_entropy(predicted_classes, target_classes, reduction='mean')
        

        targets_onehot = torch.zeros_like(predicted_classes, dtype=torch.float32)
        targets_onehot.scatter_(1, target_classes.unsqueeze(1), 1.0)

        class_loss = sigmoid_focal_loss(predicted_classes, targets_onehot, reduction='mean')

        mask = target_classes > 0
        if mask.any():
            box_loss = torch.nn.functional.smooth_l1_loss(predicted_bboxes[mask], target_bboxes[mask], reduction='mean')
        else:
            box_loss = torch.tensor(0.0, device=predicted_bboxes.device)

        return class_loss + box_loss
    
    def fit(self, train_loader, dev_loader, optimizer, svhn, train_eval_loader):
        self.train()

        for epoch in range(self.args.epochs):
            total_loss = 0.0
            steps = 0
            for images, classes, bboxes, _ in train_loader:
                images = images.to(self.device, non_blocking=True)

                predicted_classes, predicted_boxes, H, W = self(images)

                anchors = generate_anchors(H, W).to(self.device)

                loss = 0.0
                batch_size = images.shape[0]
                for i in range(batch_size):
                    gold_classes = classes[i].to(self.device).long()
                    gold_bboxes = bboxes[i].to(self.device).float()

                    target_classes, target_bboxes = bboxes_utils.bboxes_training(anchors, gold_classes, gold_bboxes, iou_threshold=0.5)
                    target_classes = target_classes.to(self.device)
                    target_bboxes = target_bboxes.to(self.device)

                    loss += self.compute_loss(predicted_classes[i], predicted_boxes[i], target_classes, target_bboxes)

                loss /= batch_size
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                total_loss += loss.item()
                steps += 1
            
            print("Evaluating ...")
            accuracy_dev = self.evaluate(dev_loader, svhn.dev)
            accuracy_train = self.evaluate(train_eval_loader, svhn.train)

            
            print(f"Epoch {epoch+1}/{self.args.epochs}, Loss: {total_loss/steps:.4f}, Dev Accuracy: {accuracy_dev:.4f}, Train Accuracy: {accuracy_train:.4f}")


    def evaluate(self, data_loader, svhn_data, iou_threshold=0.5):
        was_training = self.training
        self.eval()
        predictions = self._get_predictions(data_loader)
        accuracy = SVHN.evaluate(svhn_data, predictions, iou_threshold=iou_threshold)
        if was_training:
            self.train()
        return accuracy

    def _get_predictions(self, data_loader):
        predictions = []
        total_detections = 0
        with torch.no_grad():
            for images, _, _, sizes in data_loader:
                images = images.to(self.device, non_blocking=True)
                predicted_classes, predicted_rcnn_boxes, H, W = self(images)
                anchors = generate_anchors(H, W).to(self.device)

                batch_size = images.shape[0]
                for i in range(batch_size):
                    predicted_boxes_single = bboxes_utils.bboxes_from_rcnn(anchors, predicted_rcnn_boxes[i])
                    probs = torch.softmax(predicted_classes[i], dim=-1)
                    class_ids = torch.argmax(probs, dim=-1)
                    confidences = torch.max(probs, dim=-1)[0]

                    mask = (confidences > self.args.confidence_threshold) & (class_ids > 0)

                    final_classes = (class_ids[mask] -1)
                    final_boxes = predicted_boxes_single[mask].clone()

                    old_h, old_w = sizes[i]
                    scale_y = old_h / 224
                    scale_x = old_w / 224
                    final_boxes[:, 0] *= scale_y
                    final_boxes[:, 1] *= scale_x
                    final_boxes[:, 2] *= scale_y
                    final_boxes[:, 3] *= scale_x

                    total_detections += len(final_classes)

                    if len(final_classes) > 0:
                        nms_idx = batched_nms(final_boxes, confidences[mask], class_ids[mask], iou_threshold=0.5)
                        final_classes = final_classes[nms_idx]
                        final_boxes = final_boxes[nms_idx]

                    predictions.append((final_classes.cpu(), final_boxes.cpu()))

        print(f"Total detections: {total_detections}")
        return predictions

    def predict(self, data_loader):
        self.eval()
        return self._get_predictions(data_loader)


    




def generate_anchors(H, W, img_size=224, aspect_ratios=[0.75], anchor_size=32):
    stride = img_size / H
    anchors = []
    for i in range(H):
        for j in range(W):
            center_y = (i + 0.5) * stride
            center_x = (j + 0.5) * stride
            for aspect_ratio in aspect_ratios:
                w = anchor_size * aspect_ratio
                h = anchor_size / aspect_ratio
                anchors.append([center_y - h/2, center_x - w/2, center_y + h/2, center_x + w/2])

    return torch.tensor(anchors, dtype=torch.float32)


class SVHNDataset(torch.utils.data.Dataset):
    def __init__(self, svhn_dataset):
        self.svhn_dataset = svhn_dataset

    def __len__(self):
        return len(self.svhn_dataset)

    def __getitem__(self, index):
        example = self.svhn_dataset[index]
        image = example["image"]
        classes = example["classes"]
        bboxes = example["bboxes"].to(torch.float32).clone()

        old_h, old_w = image.shape[1], image.shape[2]
        new_h, new_w = 224, 224
        image = v2.Resize((new_h, new_w))(image)

        scale_y = new_h / old_h
        scale_x = new_w / old_w
        bboxes[:, 0] *= scale_y
        bboxes[:, 1] *= scale_x
        bboxes[:, 2] *= scale_y
        bboxes[:, 3] *= scale_x

        return image, classes, bboxes, old_h, old_w

def collate_svhn(batch):
    images = torch.stack([x[0] for x in batch], dim=0)
    classes_list = [x[1] for x in batch]
    bboxes_list = [x[2] for x in batch]
    sizes = [(x[3], x[4]) for x in batch]
    return images, classes_list, bboxes_list, sizes


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    print("Setting up...")
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the data. The individual examples are dictionaries with the keys:
    # - "image", a `[3, SIZE, SIZE]` tensor of `torch.uint8` values in [0-255] range,
    # - "classes", a `[num_digits]` PyTorch vector with classes of image digits,
    # - "bboxes", a `[num_digits, 4]` PyTorch vector with bounding boxes of image digits.
    # The `decode_on_demand` argument can be set to `True` to save memory and decode
    # each image only when accessed, but it will most likely slow down training.
    print("Loading SVHN dataset...")
    svhn = SVHN(decode_on_demand=False)

    # Load the EfficientNetV2-B0 model without the classification layer.
    # Apart from calling the model as in the classification task, you can call it using
    #   output, features = efficientnetv2_b0.forward_intermediates(batch_of_images)
    # obtaining (assuming the input images have 224x224 resolution):
    # - `output` is a `[N, 1280, 7, 7]` tensor with the final features before global average pooling,
    # - `features` is a list of intermediate features with resolution 112x112, 56x56, 28x28, 14x14, 7x7.
    print("Loading EfficientNetV2-B0 model...")
    efficientnetv2_b0 = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, num_classes=0)

    efficientnetv2_b0

    # Create a simple preprocessing performing necessary normalization.
    preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # The `scale=True` also rescales the image to [0, 1].
        v2.Normalize(mean=efficientnetv2_b0.pretrained_cfg["mean"], std=efficientnetv2_b0.pretrained_cfg["std"]),
    ])

    # TODO: Create the model and train it.
    model = Detector(args, efficientnetv2_b0, preprocessing)

    dataset = SVHNDataset(svhn.train)
    train = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_svhn)
    train_eval = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_svhn)
    dev = torch.utils.data.DataLoader(SVHNDataset(svhn.dev), batch_size=args.batch_size, collate_fn=collate_svhn)
    test = torch.utils.data.DataLoader(SVHNDataset(svhn.test), batch_size=args.batch_size, collate_fn=collate_svhn)

    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model.parameters(), T_max=args.epochs)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    print("Starting training...")
    model.fit(train, dev, optimizer, svhn, train_eval)

    # Generate test set annotations, but in `logdir` to allow parallel execution.
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "svhn_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the digits and their bounding boxes on the test set.
        # Assume that for a single test image we get
        # - `predicted_classes`: a 1D array with the predicted digits,
        # - `predicted_bboxes`: a [len(predicted_classes), 4] array with bboxes;
        predictions = model.predict(test)

        for predicted_classes, predicted_bboxes in predictions:
            output = []
            for label, bbox in zip(predicted_classes, predicted_bboxes):
                output += [int(label)] + list(map(float, bbox))
            print(*output, file=predictions_file)

        test_accuracy = SVHN.evaluate(svhn.test, predictions)
        print(f"Test Accuracy: {test_accuracy:.4f}")



if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
