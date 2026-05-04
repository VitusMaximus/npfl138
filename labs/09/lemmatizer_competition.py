#!/usr/bin/env python3
# 964bdfc8-60b0-4398-b837-7c2520532d17
# 4b50a6fb-a4a6-4b30-9879-0b671f941a72
# f5419161-0138-4909-8252-ba9794a63e53
import argparse
import os

import torch
import torchmetrics
import difflib
import torch.nn as nn
import numpy as np
import npfl138
npfl138.require_version("2526.9")
from npfl138.datasets.morpho_dataset import MorphoDataset
from npfl138.datasets.morpho_analyzer import MorphoAnalyzer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--cle_dim", default=256, type=int, help="CLE embedding dimension.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--rnn_dim", default=256, type=int, help="RNN layer dimension.")
parser.add_argument("--seed", default=41, type=int, help="Random seed.")
parser.add_argument("--show_results_every_batch", default=100, type=int, help="Show results every given batch.")
parser.add_argument("--tie_embeddings", default=True, action="store_true", help="Tie target embeddings.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--dropout", default=0.3, type=float, help="Dropout rate.")
parser.add_argument("--label_smoothing", default=0.1, type=float, help="Label smoothing.")
parser.add_argument("--weight_decay", default=1e-5, type=float, help="AdamW weight decay.")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="Initial learning rate.")
parser.add_argument("--grad_clip", default=1.0, type=float, help="Gradient clip norm (0 to disable).")
parser.add_argument("--load_path",type=str,default="")


class WithAttention(torch.nn.Module):
    """A class adding Bahdanau attention to a given RNN cell."""
    def __init__(self, cell, attention_dim):
        super().__init__()
        self._cell = cell

        # TODO: Define
        # - `self._project_encoder_layer` as a linear layer with `cell.hidden_size` inputs
        #   and `attention_dim` outputs.
        self._project_encoder_layer = nn.Linear(cell.hidden_size,attention_dim)
        # - `self._project_decoder_layer` as a linear layer with `cell.hidden_size` inputs
        #   and `attention_dim` outputs
        self._project_decoder_layer = nn.Linear(cell.hidden_size,attention_dim)
        # - `self._output_layer` as a linear layer with `attention_dim` inputs and 1 output
        self._output_layer = nn.Linear(attention_dim,1)

    def setup_memory(self, encoded:torch.Tensor):
        self._encoded = encoded
        self._encoded_projected = self._project_encoder_layer(encoded)


    def forward(self, inputs, states):
        N = inputs.shape[0]
        # TODO: Compute the attention.
        # - According to the definition, we need to project the encoder states, but we have
        #   already done that in `setup_memory`, so we just take `self._encoded_projected`.
        enc_proj = self._encoded_projected #V@h_j
        # - Compute projected decoder state by passing the given state through the `self._project_decoder_layer`.
        st_proj = self._project_decoder_layer(states) #W@ si-1 ???
        # - Sum the two projections. However, you have to deal with the fact that the first projection has
        #   shape `[batch_size, input_sequence_len, attention_dim]`, while the second projection has
        #   shape `[batch_size, attention_dim]`. The best solution is capable of creating the sum
        #   directly without creating any intermediate tensor.
        summed = enc_proj + st_proj[:,torch.newaxis,:]
        # - Pass the sum through the `torch.tanh` and then through the `self._output_layer`.
        summed = nn.functional.tanh(summed)
        out = self._output_layer(summed)
        # - The logits corresponding to the padding positions in `self._encoded_projected`
        #   should be set to -1e9 so that they do not contribute to the attention.
        mask = self._encoded.sum(-1) == 0
        out[mask] = -1e9

        # - Then, run the softmax activation, generating `weights`.
        weights = nn.functional.softmax(out,1)
        # - Multiply the original (non-projected) encoder states `self._encoded` with `weights` and sum
        #   the result in the axis corresponding to characters, generating `attention`. Therefore,
        #   `attention` is a fixed-size representation for every batch element, independently on
        #   how many characters the corresponding input word had.
        x = (weights * self._encoded).sum(1)
        # - Finally, concatenate `inputs` and `attention` (in this order), and call the `self._cell`
        #   on this concatenated input and the `states`, returning the result.
        conc = torch.concat((inputs,x),-1)
        state = self._cell(conc,states)
        return state


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        super().__init__()
        self._source_vocab = train.words.char_vocab
        self._target_vocab = train.lemmas.char_vocab

        # TODO(lemmatizer_noattn): Define
        # - `self._source_embedding` as an embedding layer of source characters into `args.cle_dim` dimensions
        # - `self._source_rnn` as a bidirectional GRU with `args.rnn_dim` units processing embedded source chars
        self._source_embedding = torch.nn.Embedding(len(self._source_vocab), args.cle_dim,padding_idx=MorphoDataset.PAD)
        self._source_rnn = torch.nn.GRU(args.cle_dim, args.rnn_dim, bidirectional=True,batch_first=True)

        self._embed_dropout = torch.nn.Dropout(args.dropout)
        self._encoder_dropout = torch.nn.Dropout(args.dropout)
        self._decoder_dropout = torch.nn.Dropout(args.dropout)

        # TODO: Define
        # - `self._target_rnn_cell` as a `WithAttention` with `attention_dim=args.rnn_dim`, employing as the
        #   underlying cell the `torch.nn.GRUCell` with `args.rnn_dim`. The cell will process concatenated
        #   target character embeddings and the result of the attention mechanism.
        self._target_rnn_cell = WithAttention(torch.nn.GRUCell(args.rnn_dim+args.cle_dim,args.rnn_dim),args.rnn_dim)

        # TODO(lemmatizer_noattn): Then define
        # - `self._target_output_layer` as a linear layer into as many outputs as there are unique target chars
        self._target_output_layer = torch.nn.Linear(args.rnn_dim,len(self._target_vocab))

        if not args.tie_embeddings:
            # TODO: Define the `self._target_embedding` as an embedding layer of the target
            # characters into `args.cle_dim` dimensions.
            self._target_embedding = torch.nn.Embedding(len(self._target_vocab),args.cle_dim,padding_idx=MorphoDataset.PAD)
        else:
            assert args.cle_dim == args.rnn_dim, "When tying embeddings, cle_dim and rnn_dim must match."
            # TODO: Create a function `self._target_embedding` computing the embedding of given
            # target characters. When called, use `torch.nn.functional.embedding` to suitably
            # index the shared embedding matrix `self._target_output_layer.weight`
            # multiplied by the square root of `args.rnn_dim`.
            def embed_target(indices:torch.Tensor):
                return torch.nn.functional.embedding(indices.to(device), self._target_output_layer.weight,padding_idx=MorphoDataset.PAD)  * (args.rnn_dim**0.5)
            self._target_embedding = embed_target

        self._show_results_every_batch = args.show_results_every_batch
        self._batches = 0

    def forward(self, words: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.encoder(words)
        if targets is not None:
            return self.decoder_training(encoded, targets)
        else:
            return self.decoder_prediction(encoded, max_length=words.shape[1] + 10)

    def encoder(self, words: torch.Tensor) -> torch.Tensor:
        embeded = self._source_embedding(words)
        embeded = self._embed_dropout(embeded)
        lens = (words !=MorphoDataset.PAD).sum(-1).cpu()
        padded = torch.nn.utils.rnn.pack_padded_sequence(embeded,lens,batch_first=True,enforce_sorted=False)
        hidden, _ = self._source_rnn(padded)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(hidden,True,MorphoDataset.PAD)
        h = hidden.shape[-1] // 2
        summed = hidden[:,:,:h] + hidden[:,:,h:]
        summed = self._encoder_dropout(summed)
        return summed

    def decoder_training(self, encoded: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # TODO(lemmatizer_noattn): Generate inputs for the decoder, which are obtained from `targets` by
        # - prepending `MorphoDataset.BOW` as the first element of every batch example,
        # - dropping the last element of `targets`.
        lens = (targets != MorphoDataset.PAD).sum(-1)
        mlens = torch.where(lens!=0,lens-1,-1).cpu()
        targets[torch.arange(0,targets.shape[0]),mlens] = MorphoDataset.PAD
        targets = targets.roll(1,-1)
        targets[:,0] = MorphoDataset.BOW

        # TODO: Pre-compute the projected encoder states in the attention by calling
        # the `setup_memory` of the `self._target_rnn_cell` on the `encoded` input.
        self._target_rnn_cell.setup_memory(encoded)

        embeded = self._target_embedding(targets)
        embeded = self._embed_dropout(embeded)
        states = encoded[:,0]
        results = []
        for i in range(embeded.shape[1]):
            states = self._target_rnn_cell(embeded[:,i,:],states)
            results.append(states)
        results = torch.stack(results,-2)
        results = self._decoder_dropout(results)
        out = self._target_output_layer(results)
        return out.permute(0,2,1)

    def decoder_prediction(self, encoded: torch.Tensor, max_length: int) -> torch.Tensor:
        batch_size = encoded.shape[0]

        # TODO(decoder_training): Pre-compute the projected encoder states in the attention by calling
        # the `setup_memory` of the `self._target_rnn_cell` on the `encoded` input.
        self._target_rnn_cell.setup_memory(encoded)

        # TODO: Define the following variables, that we will use in the cycle:
        # - `index`: the time index, initialized to 0;
        # - `inputs`: a tensor of shape `[batch_size]` containing the `MorphoDataset.BOW` symbols,
        # - `states`: initial RNN state from the encoder, i.e., `encoded[:, 0]`.
        # - `results`: an empty list, where generated outputs will be stored;
        # - `result_lengths`: a tensor of shape `[batch_size]` filled with `max_length`,
        index = 0
        inputs = torch.zeros((batch_size),dtype=torch.int32) + MorphoDataset.BOW
        states = encoded[:,0]
        results = []
        result_lengths = torch.zeros((batch_size),device=device) + max_length

        while index < max_length and torch.any(result_lengths == max_length):
            # TODO(lemmatizer_noattn):
            # - First embed the `inputs` using the `self._target_embedding` layer.
            # - Then call `self._target_rnn_cell` using two arguments, the embedded `inputs`
            #   and the current `states`. The call returns a single tensor, which you should
            #   store as both a new `hidden` and a new `states`.
            # - Pass the outputs through the `self._target_output_layer`.
            # - Generate the most probable prediction for every batch example.
            embeded = self._target_embedding(inputs)
            # - Then call `self._target_rnn_cell` using two arguments, the embedded `inputs`
            #   and the current `states`. The call returns a single tensor, which you should
            #   store as both a new `hidden` and a new `states`.
            hidden = self._target_rnn_cell(embeded,states)
            states = hidden
            # - Pass the outputs through the `self._target_output_layer`.
            # - Generate the most probable prediction for every batch example.
            logits = self._target_output_layer(hidden)
            predictions = torch.argmax(logits,-1)

            # Store the predictions in the `results` and update the `result_lengths`
            # by setting it to current `index` if an EOW was generated for the first time.
            results.append(predictions)
            result_lengths[(results[-1] == MorphoDataset.EOW) & (result_lengths > index)] = index + 1

            # TODO(lemmatizer_noattn): Finally,
            # - set `inputs` to the `predictions`,
            # - increment the `index` by one.
            inputs = predictions
            index += 1 

        results = torch.stack(results, dim=1)
        return results

    def compute_metrics(self, y_pred, y, *xs):
        if self.training:  # In training regime, convert logits to most likely predictions.
            y_pred = y_pred.argmax(dim=1)
        # Compare the lemmas with the predictions using exact match accuracy.
        y_pred = y_pred[:, :y.shape[-1]]
        y_pred = torch.nn.functional.pad(y_pred, (0, y.shape[-1] - y_pred.shape[-1]), value=MorphoDataset.PAD)
        self.metrics["accuracy"].update(torch.all((y_pred == y) | (y == MorphoDataset.PAD), dim=-1))
        return self.metrics  # Report all metrics.

    def train_step(self, xs, y):
        result = super().train_step(xs, y)

        self._batches += 1
        if self._show_results_every_batch and self._batches % self._show_results_every_batch == 0:
            self.log_console("{}: {} -> {}".format(
                self._batches,
                "".join(self._source_vocab.strings(xs[0][0][xs[0][0] != MorphoDataset.PAD].numpy(force=True))),
                "".join(self._target_vocab.strings(list(self.predict_step((xs[0][:1],)))[0].numpy(force=True)))))

        return result

    def test_step(self, xs, y):
        with torch.no_grad():
            y_pred = self(*xs)
            return self.compute_metrics(y_pred, y, *xs)

    def predict_step(self, xs):
        with torch.no_grad():
            for lemma in self(*xs):
                # Trim the predictions at the first EOW
                yield lemma[(lemma == MorphoDataset.EOW).cumsum(-1) == 0]


class TrainableDataset(npfl138.TransformedDataset):
    def __init__(self, dataset: MorphoDataset.Dataset, training: bool) -> None:
        super().__init__(dataset)
        self._training = training

    def transform(self, example):
        # TODO(lemmatizer_noattn): Return `example["words"]` as inputs and `example["lemmas"]` as targets.
        return example["words"], example["lemmas"]

    def collate(self, batch):
        # Construct a single batch, where `batch` is a list of examples generated by `transform`.
        words, lemmas = zip(*batch)
        # TODO(lemmatizer_noattn): The `words` are a list of list of strings. Flatten it into a single list of strings
        # and then map the characters to their indices using the `self.dataset.words.char_vocab` vocabulary.
        # Then create a tensor by padding the words to the length of the longest one in the batch.
        words = [torch.tensor(self.dataset.words.char_vocab.indices(list(word))) for sentence in words for word in sentence]
        lemmas = [torch.tensor(self.dataset.lemmas.char_vocab.indices(list(lemma)) + [MorphoDataset.EOW],dtype=torch.int32) for sentence in lemmas for lemma in sentence]
        words = torch.nn.utils.rnn.pad_sequence(words, batch_first=True,padding_value=MorphoDataset.PAD)
        lemmas = torch.nn.utils.rnn.pad_sequence(lemmas, batch_first=True,padding_value=MorphoDataset.PAD)
        if self._training:
            return (words, lemmas), lemmas.to(torch.long)
        return (words,), lemmas.to(torch.long)


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the data. Using analyses is only optional.
    morpho = MorphoDataset("czech_pdt")
    analyses = MorphoAnalyzer("czech_pdt_analyses")
    train = TrainableDataset(morpho.train, training=True).dataloader(batch_size=args.batch_size, shuffle=True)
    dev = TrainableDataset(morpho.dev, training=False).dataloader(batch_size=args.batch_size)
    test = TrainableDataset(morpho.test, training=False).dataloader(batch_size=args.batch_size)

    # TODO: Create the model and train it.
    model = Model(args,morpho.train)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.grad_clip > 0:
        clip_norm = args.grad_clip
        params = list(model.parameters())
        def _clip_hook(opt, a, kw):
            torch.nn.utils.clip_grad_norm_(params, clip_norm)
        optimizer.register_step_pre_hook(_clip_hook)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train),eta_min=1e-4)
    
    model.configure(
        optimizer=optimizer,
        scheduler=scheduler,
        loss=torch.nn.CrossEntropyLoss(ignore_index=MorphoDataset.PAD, label_smoothing=args.label_smoothing),
        metrics={"accuracy": torchmetrics.MeanMetric()},
        logdir=npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args)),
        device=device,
    )
    if not args.load_path:
        model.fit(train,args.epochs,dev=dev)
        model.save_weights(os.path.join(logdir,"lemmatizer_model.pt"))
    
    else:
        model.load_weights(args.load_path)

    # Generate test set annotations, but in `logdir` to allow parallel execution.
    os.makedirs(logdir, exist_ok=True)

    def postprocess_lemma(lemma:str, word:str):
        opts = [res.lemma for res in analyses.get(word)]
        if opts:
            diffs = [difflib.SequenceMatcher(a=lemma,b=other).ratio() for other in opts]
            max_i = np.argmax(diffs)
            return opts[max_i]
        return lemma
    
    with open(os.path.join(logdir, "lemmatizer_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # Predict the tags on the test set; update the following prediction
        # command if you use a different output structure than in lemmatizer_noattn.
        predictions = iter(model.predict(test, data_with_labels=True))

        for sentence in morpho.test.words.strings:
            for word in sentence:
                lemma = next(predictions)
                str_lemma = "".join(morpho.test.lemmas.char_vocab.strings(lemma))
                str_lemma = postprocess_lemma(str_lemma,word)
                if not str_lemma.strip():
                    str_lemma = word
                print(str_lemma, file=predictions_file)
            print(file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
