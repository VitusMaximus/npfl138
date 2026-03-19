#!/usr/bin/env python3
# 964bdfc8-60b0-4398-b837-7c2520532d17
# 4b50a6fb-a4a6-4b30-9879-0b671f941a72
# f5419161-0138-4909-8252-ba9794a63e53
import argparse

import numpy as np
import torch
import torchmetrics

import npfl138
npfl138.require_version("2526.2")
from npfl138.datasets.gym_cartpole_dataset import GymCartpoleDataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--evaluate", default=False, action="store_true", help="Evaluate the given model.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--render", default=False, action="store_true", help="Render during evaluation.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=16, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--epochs", default=80, type=int, help="Number of epochs.")
parser.add_argument("--model", default="gym_cartpole_model.pt", type=str, help="Output model path.")
parser.add_argument("--h_size", default=64, type=int, help="size of hidden layer")
parser.add_argument("--lr_start", default=0.005,type=float)
parser.add_argument("--lr_fin", default=0.0001,type=float)
parser.add_argument("--l2", default=0.0,type=float)


def evaluate_model(
    model: torch.nn.Module, seed: int = 42, episodes: int = 100, render: bool = False, report_per_episode: bool = False
) -> float:
    """Evaluate the given model on CartPole-v1 environment.

    Returns the average score achieved on the given number of episodes.
    """
    import gymnasium as gym

    # Create the environment.
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    env.reset(seed=seed)

    # Evaluate the episodes.
    total_score = 0
    for episode in range(episodes):
        observation, score, done = env.reset()[0], 0, False
        while not done:
            obs_tensor = torch.from_numpy(observation).float().unsqueeze(0)
            prediction = model.predict_batch(obs_tensor).squeeze(0).numpy(force=True)
            assert len(prediction) == 2, "The model must output two values."
            action = np.argmax(prediction)

            observation, reward, terminated, truncated, info = env.step(action)
            score += reward
            done = terminated or truncated

        total_score += score
        if report_per_episode:
            print(f"The episode {episode + 1} finished with score {score}.")
    return total_score / episodes


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()

        # TODO: Create the model layers, with the last layer having 2 outputs.
        # To store a list of layers, you can use either `torch.nn.Sequential`
        # or `torch.nn.ModuleList`; you should *not* use a Python list.
        self.h1 = torch.nn.Linear(4,args.h_size)
        self.o = torch.nn.Linear(args.h_size,2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # TODO: Run your model and return its output.
        H = self.h1(inputs)
        H = torch.relu(H)
        O = self.o(H)
        return O


def main(args: argparse.Namespace) -> torch.nn.Module | None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads, args.recodex)
    npfl138.global_keras_initializers()

    if not args.evaluate:
        if args.batch_size is ...:
            raise ValueError("You must specify the batch size, either in the defaults or on the command line.")
        if args.epochs is ...:
            raise ValueError("You must specify the number of epochs, either in the defaults or on the command line.")

        # Load the provided dataset. The `dataset.train` is a collection of 100 examples,
        # each being a pair of (inputs, label), where:
        # - `inputs` is a vector with `GymCartpoleDataset.FEATURES` floating point values,
        # - `label` is a gold 0/1 class index.
        dataset = GymCartpoleDataset()

        train = torch.utils.data.DataLoader(dataset.train, args.batch_size, shuffle=True,num_workers=0)

        model = Model(args)

        # TODO: Configure the model for training.
        # To properly use a GPU, the model parameters should be moved to the device
        # BEFORE initializing the optimizer, otherwise the optimizer tracked states remain on CPU!
        print(torch.cuda.is_available())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        optim = torch.optim.Adam(model.parameters(),lr=args.lr_start, weight_decay=args.l2)
        s = len(train) * args.epochs
        model.configure(
            optimizer=optim,
            loss=torch.nn.CrossEntropyLoss(),
            metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=2)},
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optim,s,args.lr_fin),
            device=device
        )


        # TODO: Train the model.
        #
        # Note that the `fit` method accepts a `callbacks` argument, which is a list
        # of callables that are called at the end of each epoch, each being called
        # with the model, epoch, and logs (a dictionary with logged losses and metrics).
        def callback(model: Model, epoch: int, logs: dict[str, float]) -> None | npfl138.StopTraining:
            # When you add items to the `logs` dictionary, they will be logged
            # both to the console and to TensorBoard.
            score = evaluate_model(model)
            logs["eval_score"] = score
            if score >= 490:
                return npfl138.STOP_TRAINING

        model.fit(train, epochs=args.epochs, callbacks=[callback])

        # Save the model, both the hyperparameters and the parameters. If you
        # added additional arguments to the `Model` constructor beyond `args`,
        # you would have to add them to the `save_options` call below.
        model.save_options(f"{args.model}.json", args=args)
        model.save_weights(args.model)

    else:
        # Evaluating, either manually or in ReCodEx.
        model = Model(**Model.load_options(f"{args.model}.json"))
        model.load_weights(args.model)

        if args.recodex:
            return model
        else:
            score = evaluate_model(model, seed=args.seed, render=args.render, report_per_episode=True)
            print(f"The average score was {score}.")


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
