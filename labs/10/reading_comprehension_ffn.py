#!/usr/bin/env python3
# 964bdfc8-60b0-4398-b837-7c2520532d17
# 4b50a6fb-a4a6-4b30-9879-0b671f941a72
# f5419161-0138-4909-8252-ba9794a63e53
import argparse
import os

os.environ["TRANSFORMERS_VERBOSITY"] = (
    "error"  # Suppress the LOAD REPORT with weight discrepancies.
)

import torch
import torchmetrics
import transformers

import torch.nn as nn
import npfl138

npfl138.require_version("2526.10")
from npfl138.datasets.reading_comprehension_dataset import ReadingComprehensionDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--epochs", default=15, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument(
    "--threads", default=0, type=int, help="Maximum number of threads to use."
)
parser.add_argument("--finetune_lr", type=float, default=2e-5)
parser.add_argument("--lr",type=float,default=1e-3)
parser.add_argument("--lr_fin1",type=float,default=1e-4)
parser.add_argument("--lr_final",type=float,default=1e-6)
parser.add_argument("--finetuning_epochs",type=int,default=5)
parser.add_argument("--decay",type=float,default=0.01)
parser.add_argument("--expansion",type=int,default=2)
parser.add_argument("--num_layers",type=int,default=2)
parser.add_argument("--dropout",type=float,default=0.2)
parser.add_argument("--label_smoothing",type=float,default=0.1)

class Head(torch.nn.Module):
    def __init__(self, dim: int, expansion: int, num_layers:int, dropout:float) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(self.FFN(dim,expansion,dropout))
        
        self.dropout = nn.Dropout(p=dropout)
        self.start = nn.Linear(dim,1)
        self.end = nn.Linear(dim,1)
        
    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            inputs = inputs + layer(inputs)
        inputs = self.dropout(inputs)
        return self.start(inputs), self.end(inputs)
            

    class FFN(torch.nn.Module):
        def __init__(self, dim: int, expansion: int, dropout:float) -> None:
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.LayerNorm(dim),
                torch.nn.Linear(dim, dim * expansion),
                torch.nn.ReLU(),
                torch.nn.Linear(dim * expansion, dim),
                torch.nn.Dropout(p=dropout)
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.layers(inputs)

class Model(npfl138.TrainableModule):
    NEGINF = -1e9

    def __init__(
        self,
        args: argparse.Namespace,
        train: ReadingComprehensionDataset,
        encoder: transformers.AutoModelForMaskedLM,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = Head(self.encoder.config.hidden_size, args.expansion, args.num_layers, args.dropout)

    def set_encoder_state(self, train: bool = False):
        if train:
            self.encoder.train()
        else:
            self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = train

    def forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor, context_lens:torch.Tensor):
        encoded = self.encoder(inputs, attention_mask=attention_mask).last_hidden_state
        starts, ends = self.head(encoded)
        pos = torch.arange(inputs.shape[1], device=device)
        context_masks = pos.unsqueeze(0) >= context_lens.unsqueeze(1).to(device)
        
        starts[context_masks] = self.NEGINF
        ends[context_masks] = self.NEGINF

        if self.training:
            result = torch.stack([starts, ends], dim=2)
            return result
        else:
            starts = starts.squeeze(-1)  # [B, seq]
            ends = ends.squeeze(-1)      # [B, seq]
            best_start = torch.argmax(starts, dim=1)  # [B]
            pos = torch.arange(ends.shape[1], device=device)
            ends = ends.masked_fill(pos.unsqueeze(0) < best_start.unsqueeze(1), self.NEGINF)
            return torch.stack([starts, ends], dim=2)


class TransformedDataset(npfl138.TransformedDataset):
    def __init__(
        self,
        dataset: ReadingComprehensionDataset,
        tokenizer: transformers.AutoTokenizer,
    ) -> None:
        super().__init__(dataset.paragraphs)

        self.tokenizer = tokenizer

    def transform(self, example):
        qas = example.get("qas", [])
        patterns = []
        golds = []
        context = example.get("context", "")

        for qa in qas:
            enc = self.tokenizer(context, qa["question"], return_offsets_mapping=True,
                                 truncation="only_first", max_length=512)
            token_ids = enc["input_ids"]
            attn_mask = enc["attention_mask"]
            offsets = enc["offset_mapping"]
            seq_ids = enc.sequence_ids()
            context_len = 0
            for i, s in enumerate(seq_ids):
                if s == 0:
                    context_len = i + 1

            patterns.append((torch.tensor(token_ids), torch.tensor(attn_mask), torch.tensor(context_len)))
            token_spans = []
            for a in qa["answers"]:
                start_char = a["start"]
                end_char = a["start"] + len(a["text"])
                start_tok, end_tok = 0, 0
                for i, s in enumerate(seq_ids):
                    if s != 0:
                        continue
                    tok_start, tok_end = offsets[i]
                    if tok_start <= start_char < tok_end:
                        start_tok = i
                    if tok_start < end_char <= tok_end:
                        end_tok = i
                        break
                token_spans.append((start_tok, end_tok))
            golds.append(token_spans)
        return patterns, torch.tensor(golds, dtype=torch.int32)

    def collate(self, batch):
        patterns, golds = zip(*batch)
        flatten_patterns = [x for patterns in patterns for x in patterns]
        flatten_golds = [y for golds in golds for y in golds]
        token_ids, masks, context_lens = zip(*flatten_patterns)

        token_ids = nn.utils.rnn.pad_sequence(token_ids,True,self.tokenizer.pad_token_id)
        masks = nn.utils.rnn.pad_sequence(masks,True,padding_value=0)
        context_lens = torch.tensor(context_lens)
        
        return (token_ids, masks, context_lens), torch.stack(flatten_golds)

class OneCorrectMetric(nn.Module,npfl138.Metric):
    def __init__(self):
        super().__init__()
        self.register_buffer("_correct", torch.tensor(0.0, dtype=torch.float32, device=device), persistent=False)
        self.register_buffer("_total", torch.tensor(0.0, dtype=torch.float32, device=device), persistent=False)

    def update(self, y_pred: torch.Tensor, true_answers: torch.Tensor):
        if y_pred.dim() == 4:
            y_pred = y_pred.squeeze(-1)
        pred_start = y_pred[:, :, 0].argmax(dim=1)
        pred_end = y_pred[:, :, 1].argmax(dim=1)
        if true_answers.dim() == 2:
            true_answers = true_answers.unsqueeze(1)
        gold_starts = true_answers[:, :, 0].to(pred_start.device).long()
        gold_ends = true_answers[:, :, 1].to(pred_end.device).long()
        match = (gold_starts == pred_start.unsqueeze(1)) & (gold_ends == pred_end.unsqueeze(1))
        self._correct += match.any(dim=1).float().sum()
        self._total += pred_start.shape[0]

    def compute(self):
        return self._correct / self._total

    def reset(self):
        self._correct.zero_()
        self._total.zero_()

class OneCorrectLoss(npfl138.Loss):
    def __init__(self, ignore_idx, label_smoothing):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_idx, label_smoothing=label_smoothing, reduction="none")

    def __call__(self, y_pred: torch.Tensor, true_answers: torch.Tensor):
        if y_pred.dim() == 4:
            y_pred = y_pred.squeeze(-1)
        starts = y_pred[:, :, 0]
        ends = y_pred[:, :, 1]
        if true_answers.dim() == 2:
            true_answers = true_answers.unsqueeze(1)
        gold_starts = true_answers[:, :, 0].to(starts.device).long()
        gold_ends = true_answers[:, :, 1].to(ends.device).long()
        B, K = gold_starts.shape
        L = starts.shape[1]
        starts_exp = starts.unsqueeze(1).expand(B, K, L).reshape(B * K, L)
        ends_exp = ends.unsqueeze(1).expand(B, K, L).reshape(B * K, L)
        ls = self.ce(starts_exp, gold_starts.reshape(B * K)).reshape(B, K)
        le = self.ce(ends_exp, gold_ends.reshape(B * K)).reshape(B, K)
        per_answer = ls + le
        min_loss, _ = per_answer.min(dim=1)
        return min_loss.mean()
        


def configure_model(model:Model,lr_start,lr_end,train:TransformedDataset,epochs,pad_token, weight_decay=0.01, label_smoothing=0.0):
    optim =torch.optim.AdamW(model.parameters(), lr=lr_start,weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs * len(train),eta_min=lr_end)
    model.configure(
        optimizer=optim,
        scheduler=scheduler,
        loss=OneCorrectLoss(pad_token,label_smoothing),
        metrics={
            "accuracy": OneCorrectMetric(),
        },
        device=device
    )


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the pre-trained RobeCzech model.
    tokenizer = transformers.AutoTokenizer.from_pretrained("ufal/robeczech-base")
    robeczech = transformers.AutoModel.from_pretrained("ufal/robeczech-base")

    # Load the data.
    dataset = ReadingComprehensionDataset()
    train = TransformedDataset(dataset.train,tokenizer).dataloader(args.batch_size, shuffle=True)
    dev = TransformedDataset(dataset.dev,tokenizer).dataloader(args.batch_size)
    test = TransformedDataset(dataset.test,tokenizer).dataloader(args.batch_size)

    # TODO: Create the model and train it.
    model = Model(args, dataset.train, encoder=robeczech)
    
    model.set_encoder_state(False)
    configure_model(model,args.lr,args.lr_fin1,train,args.epochs,tokenizer.pad_token_id, weight_decay=args.decay, label_smoothing=args.label_smoothing)
    model.fit(train, dev=dev, epochs=args.epochs)

    model.set_encoder_state(True)
    configure_model(model,args.finetune_lr,args.lr_final,train,args.finetuning_epochs,tokenizer.pad_token_id, weight_decay=args.decay, label_smoothing=args.label_smoothing)
    model.fit(train, dev=dev, epochs=args.finetuning_epochs)

    model.save_weights(os.path.join(logdir,"model.pt"))

    # Generate test set annotations, but in `logdir` to allow parallel execution.
    os.makedirs(logdir, exist_ok=True)
    with open(
        os.path.join(logdir, "reading_comprehension.txt"), "w", encoding="utf-8"
    ) as predictions_file:
        # TODO: Predict the answers as strings, one per line.
        model.eval()
        with torch.no_grad():
            for (token_ids, masks, context_lens), _ in test:
                predictions = []
                token_ids = token_ids.to(device)
                masks = masks.to(device)
                context_lens = context_lens.to(device)
                y_pred = model(token_ids, masks, context_lens)
                if y_pred.dim() == 4:
                    y_pred = y_pred.squeeze(-1)
                starts = y_pred[:, :, 0].argmax(dim=1)
                ends = y_pred[:, :, 1].argmax(dim=1)
                for ids, s, e in zip(token_ids, starts, ends):
                    print(tokenizer.decode(ids[s:e + 1], skip_special_tokens=True).strip(),file=predictions_file)



if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
