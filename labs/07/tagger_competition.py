#!/usr/bin/env python3
# 964bdfc8-60b0-4398-b837-7c2520532d17
# 4b50a6fb-a4a6-4b30-9879-0b671f941a72
# f5419161-0138-4909-8252-ba9794a63e53

import argparse
import os

import torch
import torchmetrics

import npfl138
npfl138.require_version("2526.7")
from npfl138.datasets.morpho_dataset import MorphoDataset
from npfl138.datasets.morpho_analyzer import MorphoAnalyzer

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
# Defaults are set to values with which submitted model was trained 
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=100, type=int, help="Batch size.")
parser.add_argument("--epochs", default=25, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")

parser.add_argument("--cle_dim", default=128, type=int, help="CLE embedding dimension.")
parser.add_argument("--rnn", default="GRU", choices=["LSTM", "GRU"], help="RNN layer type.") # GRU gave better results
parser.add_argument("--rnn_dim", default=256, type=int, help="RNN layer dimension.")
parser.add_argument("--we_dim", default=256, type=int, help="Word embedding dimension.")
parser.add_argument("--word_masking", default=0.2, type=float, help="Mask words with the given probability.")

# Compression of 1500+ tags by MorphoAnalyzer before RNN (I am starting with 64...)
parser.add_argument("--analyzer_dim", default=64, type=int, help="MorphoAnalyzer embedding dimension.")

# Adding dropout
parser.add_argument("--dropout", default=0.4, type=float, help="Dropout probability.")

class Model(npfl138.TrainableModule):
    class MaskElements(torch.nn.Module):
        def __init__(self, mask_probability, mask_value):
            super().__init__()
            self._mask_probability = mask_probability
            self._mask_value = mask_value

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            if self.training and self._mask_probability:
                mask_tensor = torch.rand_like(inputs, dtype=torch.float32)
                inputs = torch.where(mask_tensor < self._mask_probability, self._mask_value, inputs)
            return inputs

    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        super().__init__()
        self._word_masking = self.MaskElements(args.word_masking, MorphoDataset.UNK)
        self._char_embedding = torch.nn.Embedding(len(train.words.char_vocab), args.cle_dim)
        self._char_rnn = torch.nn.GRU(input_size=args.cle_dim, hidden_size=args.cle_dim, bidirectional=True)
        self._word_embedding = torch.nn.Embedding(len(train.words.string_vocab), args.we_dim)

        self._dropout = torch.nn.Dropout(args.dropout)

        num_tags = len(train.tags.string_vocab)
        self._analyzer_projection = torch.nn.Linear(num_tags, args.analyzer_dim)

        rnn_input_size = args.we_dim + 2 * args.cle_dim + args.analyzer_dim
        if (args.rnn=="LSTM"):
            self._word_rnn = torch.nn.LSTM(input_size=rnn_input_size, hidden_size=args.rnn_dim, bidirectional=True, num_layers=2, dropout=args.dropout) # Adding dropout and 2 layer RNN
        elif (args.rnn=="GRU"):
            self._word_rnn = torch.nn.GRU(input_size=rnn_input_size, hidden_size=args.rnn_dim, bidirectional=True, num_layers=2, dropout=args.dropout)

        self._output_layer = torch.nn.Linear(args.rnn_dim, num_tags)

    def forward(self, word_ids: torch.Tensor, unique_words: torch.Tensor, word_indices: torch.Tensor, analyzer_features: torch.Tensor) -> torch.Tensor:
        hidden = self._word_masking(word_ids)
        hidden = self._word_embedding(hidden)
        cle = self._char_embedding(unique_words)


        char_lengths = (unique_words != MorphoDataset.PAD).sum(dim=1).cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(cle, char_lengths, batch_first=True, enforce_sorted=False)
        _, final_hidden_state = self._char_rnn(packed)
        cle = torch.cat([final_hidden_state[0], final_hidden_state[1]], dim=1) 
        cle = torch.nn.functional.embedding(word_indices, cle)

        # Compressing multi-hot sparse vector to dense form
        analyzer_emb = torch.nn.functional.relu(self._analyzer_projection(analyzer_features))    

        # Concat all 3 together
        hidden = torch.cat([hidden, cle, analyzer_emb], dim=-1) 

        hidden = self._dropout(hidden)
        
        lengths = (word_ids != MorphoDataset.PAD).sum(dim=1).cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(hidden, lengths , batch_first=True, enforce_sorted=False)
        packed, _ = self._word_rnn(packed)
        hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)
        
        forwards = hidden[:, :, :hidden.size(2)//2]
        backwards = hidden[:, :, hidden.size(2)//2:]
        hidden = forwards + backwards

        hidden = self._dropout(hidden)

        hidden = self._output_layer(hidden).permute(0,2,1)
        return hidden

class TrainableDataset(npfl138.TransformedDataset):
    def __init__(self, dataset, analyses=None):
        super().__init__(dataset)
        self.analyses = analyses

    def transform(self, example):
        words = example["words"]
        word_ids = torch.tensor(self.dataset.words.string_vocab.indices(example["words"]))

        # MorphoAnalyzer multi-hot encoding
        num_tags = len(self.dataset.tags.string_vocab)
        analyzer_features = torch.zeros(len(words), num_tags, dtype=torch.float32)

        if self.analyses:
            for i, word in enumerate(words):
                results = self.analyses.get(word)
                if results:
                    valid_tags = [item.tag for item in results]
                    tag_indices = self.dataset.tags.string_vocab.indices(valid_tags)

                    for t_id in tag_indices:
                        if t_id != self.dataset.tags.string_vocab.UNK:
                            analyzer_features[i, t_id] = 1.0
        
        tag_ids = torch.tensor(self.dataset.tags.string_vocab.indices(example["tags"])) if "tags" in example else torch.tensor([]) # Test sets dont have tags
        return word_ids, words, analyzer_features, tag_ids

    def collate(self, batch):
        word_ids, words, analyzer_features,tag_ids = zip(*batch)
        word_ids = torch.nn.utils.rnn.pad_sequence(word_ids, batch_first=True)
        unique_words, words_indices = self.dataset.cle_batch(words)
        
        analyzer_features = torch.nn.utils.rnn.pad_sequence(analyzer_features, batch_first=True) 
        tag_ids = torch.nn.utils.rnn.pad_sequence(tag_ids, batch_first=True) if len(tag_ids[0]) > 0 else torch.tensor([])

        return (word_ids, unique_words, words_indices, analyzer_features), tag_ids

def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the data. Using analyses is only optional.
    morpho = MorphoDataset("czech_pdt")
    analyses = MorphoAnalyzer("czech_pdt_analyses")

    train_loader = TrainableDataset(morpho.train, analyses).dataloader(batch_size=args.batch_size, shuffle=True)
    dev_loader = TrainableDataset(morpho.dev, analyses).dataloader(batch_size=args.batch_size)
    test = TrainableDataset(morpho.test, analyses).dataloader(batch_size=args.batch_size)
    
    # TODO: Create the model and train it.
    model = Model(args, morpho.train)

    model.configure(
        optimizer=torch.optim.Adam(model.parameters(), weight_decay=1e-5), # Trying out with weight decay=1e-5 
        loss=torch.nn.CrossEntropyLoss(ignore_index=morpho.PAD),
        metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=len(morpho.train.tags.string_vocab), ignore_index=morpho.PAD)},
        logdir=logdir,
    )

    model.fit(train_loader, dev=dev_loader, epochs=args.epochs)

    # Generate test set annotations, but in `logdir` to allow parallel execution.
    os.makedirs(logdir, exist_ok=True)

    with open(os.path.join(logdir, "tagger_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the tags on the test set. The following code assumes you use the same
        # output structure as in `tagger_we`, i.e., that for each sentence, the predictions are
        # a Numpy vector of shape `[num_tags, sentence_len_or_more]`, where `sentence_len_or_more`
        # is the length of the corresponding batch. (FYI, if you instead used the `packed` variant,
        # the prediction for each sentence is a vector of shape `[exactly_sentence_len, num_tags]`.)
        predictions = model.predict(test, data_with_labels=True, as_numpy=True)

        for predicted_tags, words in zip(predictions, morpho.test.words.strings):
            for predicted_tag in predicted_tags[:, :len(words)].argmax(axis=0):
                print(morpho.train.tags.string_vocab.string(predicted_tag), file=predictions_file)
            print(file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
