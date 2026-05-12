#!/usr/bin/env python3
# 964bdfc8-60b0-4398-b837-7c2520532d17
# 4b50a6fb-a4a6-4b30-9879-0b671f941a72
# f5419161-0138-4909-8252-ba9794a63e53
import npfl138
import argparse
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # Suppress the LOAD REPORT with weight discrepancies.

import torch
import torchmetrics
import transformers

import npfl138
npfl138.require_version("2526.10")
from npfl138.datasets.text_classification_dataset import TextClassificationDataset

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")

parser.add_argument("--dropout", default=0.3, type=float, help="Dropout rate.")

class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, eleczech: transformers.PreTrainedModel,
                 dataset: TextClassificationDataset.Dataset) -> None:
        super().__init__()

        # TODO: Define the model. Note that
        # - the dimension of the EleCzech output is `eleczech.config.hidden_size`;
        # - the size of the vocabulary of the output labels is `len(dataset.label_vocab)`.
        self.eleczech = eleczech
        self.dropout = torch.nn.Dropout(p=args.dropout)
        self.classifier = torch.nn.Linear(in_features=eleczech.config.hidden_size, out_features=len(dataset.label_vocab))
        

    # TODO: Implement the model computation.
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # result = model(torch.as_tensor(batch.input_ids), attention_mask=torch.as_tensor(batch.attention_mask))from example_transformers.py
        transformer_results = self.eleczech(input_ids, attention_mask=attention_mask)
        
        # embeddings from the final layer
        hidden_states = transformer_results.last_hidden_state

        # extracting special CLS token representation (index 0)
        # this token acts as an aggregate summary of the entire sentence meaning 
        # NOTE: If this wont workout i will try using more information from all tokens
        cls_rep = hidden_states[:, 0, :]

        # Maybe i should try to add dropout layer between the pre-trained transformer and new untrained classification:
        logits = self.classifier(self.dropout(cls_rep))
        return logits


class TrainableDataset(npfl138.TransformedDataset):
    def __init__(self, dataset: TextClassificationDataset.Dataset, tokenizer: transformers.PreTrainedTokenizer) -> None:
        super().__init__(dataset)
        self.tokenizer = tokenizer
    def transform(self, example):
        # TODO: Process single examples containing `example["document"]` and `example["label"]`.
        label_idx = self.dataset.label_vocab.index(example["label"]) 
        return {
            "document": example["document"],
            "label": label_idx if label_idx is not None else 0, # safe guard if test set has unknown labels
        }

    def collate(self, batch):
        # TODO: Construct a single batch using a list of examples from the `transform` function.
        texts = [example["document"] for example in batch]
        labels = [example["label"] for example in batch]

        # tokenizing and padding the entire batch (batch = tokenizer(dataset, padding="longest") from example_transformers.py)
        encoded = self.tokenizer(texts, padding="longest", truncation=True, return_tensors="pt") # trying the truncation if some facebook comment exceeds models capacity

        return (encoded.input_ids, encoded.attention_mask), torch.as_tensor(labels)

def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the Electra Czech small lowercased.
    tokenizer = transformers.AutoTokenizer.from_pretrained("ufal/eleczech-lc-small")
    eleczech = transformers.AutoModel.from_pretrained("ufal/eleczech-lc-small")

    # Load the data.
    facebook = TextClassificationDataset("czech_facebook")

    # TODO: Prepare the data for training.
    train = TrainableDataset(facebook.train, tokenizer).dataloader(batch_size=args.batch_size, shuffle=True)
    dev = TrainableDataset(facebook.dev, tokenizer).dataloader(batch_size=args.batch_size)
    test = TrainableDataset(facebook.test, tokenizer).dataloader(batch_size=args.batch_size)

    # Create the model.
    model = Model(args, eleczech, facebook.train)

    # TODO: Configure and train the model
    # Tried out the scheduler but didnt help...
    model.configure(
        optimizer=torch.optim.AdamW(model.parameters(), lr=2e-5),  # maybe also a scheduler, but not required
        loss=torch.nn.CrossEntropyLoss(),
        metrics={"accuracy": npfl138.metrics.CategoricalAccuracy()},
        logdir=logdir,
    )
    model.fit(train, dev=dev, epochs=args.epochs)

    # Generate test set annotations, but in `logdir` to allow parallel execution.
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "sentiment_analysis.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the tags on the test set.
        predictions = model.predict(test, data_with_labels=True)

        for document_logits in predictions:
            print(facebook.train.label_vocab.string(document_logits.argmax().item()), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
