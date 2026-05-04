#!/usr/bin/env python3
# 964bdfc8-60b0-4398-b837-7c2520532d17
# 4b50a6fb-a4a6-4b30-9879-0b671f941a72
# f5419161-0138-4909-8252-ba9794a63e53
import argparse
import os

import torch
import torchaudio.models.decoder
import torch.nn.functional as F
import torch.nn as nn
import npfl138
npfl138.require_version("2526.8.1")
from npfl138.datasets.common_voice_cs import CommonVoiceCs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=24, type=int, help="Batch size.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--cnn_stride",default=1,type=int)
parser.add_argument("--batch-norm", default=False,type=bool)
parser.add_argument("--dropout", default=0.1,type=float)
parser.add_argument("--rnn", default="LSTM", choices=["LSTM", "GRU"], help="RNN layer type.")
parser.add_argument("--rnn_dim", default=128, type=int, help="RNN layer dimension.")
parser.add_argument("--layers", default=2, type=int, help="Number of RNN layers.")
parser.add_argument("--hidden", default=True, type=bool, help="Add hidden layer.")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--fin_lr", default=0.0001, type=float, help="Final learning rate.")
parser.add_argument("--model_path", default=None, type=str, help="Path to the model.")

class CNN(nn.Module):
    def __init__(self,stride:int,C:int=13,BN:bool = False):
        super().__init__()
        layers = nn.ModuleList()
        layers.append(nn.Conv1d(C,2*C,9,stride,padding=4))
        for i in range(1,3):
            if BN:
                layers.append(nn.BatchNorm1d((2**i)*C))
            layers.append(nn.ReLU())
            layers.append(nn.Conv1d((2**i)*C,(2**(i+1))*C,9-(2*i),stride,padding=4-i))
        self.layers = layers
        self._model = nn.Sequential(*layers)

    def forward(self,x):
        return self._model(x.permute(0,2,1)).permute(0,2,1)


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, train: CommonVoiceCs.Dataset) -> None:
        super().__init__()
        # TODO: Define the model.
        self.cnn = CNN(args.cnn_stride,BN=args.batch_norm)
        if (args.rnn=="LSTM"):
            self._rnn = torch.nn.LSTM(input_size=13*2**3, hidden_size=args.rnn_dim, bidirectional=True, num_layers=args.layers, dropout=args.dropout) # Adding dropout and 2 layer RNN
        elif (args.rnn=="GRU"):
            self._rnn = torch.nn.GRU(input_size=13*2**3, hidden_size=args.rnn_dim, bidirectional=True, num_layers=args.layers, dropout=args.dropout)
        C = args.rnn_dim
        self.hidden = args.hidden
        if self.hidden:
            self.hidden = nn.Linear(args.rnn_dim,args.rnn_dim*2)
            C = args.rnn_dim*2
        self.out = nn.Linear(C,CommonVoiceCs.LETTERS)
        

    def forward(self, mfccs: torch.Tensor, pred_lens: torch.Tensor) -> torch.Tensor:
        # TODO: Compute the output of the model.
        xs = self.cnn(mfccs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(xs, pred_lens.cpu(), batch_first=True, enforce_sorted=False)
        rnns, _ = self._rnn(packed)
        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(rnns, True)
        h = unpacked.shape[-1] // 2
        hidden = unpacked[:,:,:h] + unpacked[:,:,h:]
        if self.hidden:
            hidden = self.hidden(hidden)
            hidden = F.relu(hidden)
        out = self.out(hidden)
        return out
        

    def compute_loss(self, y_pred: torch.Tensor, y_true: tuple, mfccs: torch.Tensor, pred_lens: torch.Tensor) -> torch.Tensor:
        # TODO: Compute the loss, most likely using the `torch.nn.CTCLoss` class.
        char_ids, true_lens = y_true
        log_probs = F.log_softmax(y_pred, -1)
        return torch.nn.functional.ctc_loss(log_probs.permute(1,0,2), char_ids, pred_lens, true_lens, blank=CommonVoiceCs.PAD)

    def ctc_decoding(self, y_pred: torch.Tensor) -> list[torch.Tensor]:
        # TODO: Compute predictions, either using manual CTC decoding, or you can use:
        # - `torchaudio.models.decoder.ctc_decoder`, which is CPU-based decoding with
        #   rich functionality;
        #   - note that you need to provide `blank_token` and `sil_token` arguments
        #     and they must be valid tokens. For `blank_token`, you need to specify
        #     the token whose index corresponds to the blank token index;
        #     for `sil_token`, you can use also the blank token index (by default,
        #     `sil_token` has ho effect on the decoding apart from being added as the
        #     first and the last token of the predictions unless it is a blank token).
        # - `torchaudio.models.decoder.cuda_ctc_decoder`, which is faster GPU-based
        #   decoder with limited functionality.
        probs = F.softmax(y_pred,dim=-1)
        decoder = torchaudio.models.decoder.ctc_decoder(blank_token="[PAD]",sil_token="[PAD]",lexicon=None,tokens=CommonVoiceCs.LETTER_NAMES)
        decoded = decoder(probs.cpu())
        return [dec[0].tokens for dec in decoded]
        

    def compute_metrics(
        self, y_pred: torch.Tensor, y_true: tuple, mfccs: torch.Tensor, pred_lens: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # TODO: Compute predictions using the `ctc_decoding`. Consider computing it
        # only when `self.training==False` to speed up training.
        char_ids, true_lens = y_true
        predictions = self.ctc_decoding(y_pred)
        self.metrics["edit_distance"].update(predictions, char_ids)
        return self.metrics

    def predict_step(self, xs):
        with torch.no_grad():
            # Perform constrained decoding.
            for (mfccs, pred_lens), _ in xs:
                mfccs = mfccs.to(next(self.parameters()).device)
                pred_lens = pred_lens.to(next(self.parameters()).device)
                y_pred = self.forward(mfccs, pred_lens)
                yield from self.ctc_decoding(y_pred)


class TrainableDataset(npfl138.TransformedDataset):
    def transform(self, example):
        # TODO: Prepare a single example. The structure of the inputs then has to be reflected
        # in the `forward`, `compute_loss`, and `compute_metrics` methods; right now, there are
        # just `...` instead of the input arguments in the definition of the mentioned methods.
        #
        # You can use `CommonVoiceCs.LETTER_NAMES : list[str]` or `CommonVoiceCs.LETTERS_VOCAB : npfl138.Vocabulary`
        # to convert between letters and their indices. While the letters do not explicitly contain
        # a blank token, the [PAD] token can be employed as one.
        char_ids = CommonVoiceCs.LETTERS_VOCAB.indices(list(example["sentence"])) 
        return example["mfccs"],torch.tensor(char_ids,dtype=torch.long)

    def collate(self, batch):
        # TODO: Construct a single batch from a list of individual examples.
        mfccs, char_ids = zip(*batch)
        pred_lens = torch.tensor([mfcc.shape[0] for mfcc in mfccs], dtype=torch.long).cpu()
        mfccs = torch.nn.utils.rnn.pad_sequence(mfccs, batch_first=True, padding_value=CommonVoiceCs.PAD)
        char_ids = torch.nn.utils.rnn.pad_sequence(char_ids, batch_first=True, padding_value=CommonVoiceCs.PAD)
        true_lens = (char_ids != CommonVoiceCs.PAD).sum(-1).cpu()
        return (mfccs, pred_lens), (char_ids, true_lens)


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the data.
    common_voice = CommonVoiceCs()

    train = TrainableDataset(common_voice.train).dataloader(args.batch_size, shuffle=True)
    dev = TrainableDataset(common_voice.dev).dataloader(args.batch_size)
    test = TrainableDataset(common_voice.test).dataloader(args.batch_size)

    # TODO: Create the model and train it. The `Model.compute_metrics` method assumes you
    # passed the following metric to the `configure` method under the name "edit_distance":
    #   CommonVoiceCs.EditDistanceMetric(ignore_index=CommonVoiceCs.PAD)
    model = Model(args, train)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr)
    schedulrer = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs*len(train),eta_min=args.fin_lr)
    model.configure(
        optimizer=optimizer,
        scheduler=schedulrer,
        metrics={
            "edit_distance": CommonVoiceCs.EditDistanceMetric(ignore_index=CommonVoiceCs.PAD),
        },
        device=device
    )
    if args.model_path:
        model.load_weights(args.model_path)
    else:
        model.fit(train, epochs=args.epochs,dev=dev)
        model.save_weights("sp_rec.pt")

    # Generate test set annotations, but in `model.logdir` to allow parallel execution.
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "speech_recognition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the CommonVoice sentences.
        predictions = model.predict_step(test)

        for sentence in predictions:
            print("".join(CommonVoiceCs.LETTERS_VOCAB.strings(sentence)), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
