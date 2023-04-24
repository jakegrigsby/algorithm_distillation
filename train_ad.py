from argparse import ArgumentParser
import random
import os

from experiment import Experiment
from dark_key_to_door import RoomKeyDoor


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument(
        "--experiment",
        type=str,
        default="ad_dark",
        choices=["ad_dark", "ad_light", "bc_dark", "bc_light", "rl2_dark"],
    )
    parser.add_argument("--gpus", nargs="*", type=int, default=[0])
    parser.add_argument("--no_log", action="store_true")
    parser.add_argument(
        "--model",
        type=str,
        default="transformer",
        choices=["transformer", "feedforward"],
    )
    parser.add_argument("--context_len", type=int, default=300)
    parser.add_argument("--buffer_dir", type=str, default="buffers/")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    return args


def gather_dset_files_rl2(buffer_dir: str):
    path = os.path.join(buffer_dir, "merged")
    files = [os.path.join(path, f) for f in os.listdir(path)]
    random.shuffle(files)
    split = int(0.8 * len(files))
    train_files, val_files = files[:split], files[-split:]
    return train_files, val_files


def gather_dset_files_sac(
    kind: str = "Nx1", dark: bool = True, buffer_dir: str = "buffers/"
):
    assert kind in ["Nx1", "1xN"]
    path = os.path.join(buffer_dir, kind)

    # collect buffers
    filenames = os.listdir(path)
    use_filenames = []
    for filename in filenames:
        if kind == "Nx1":
            # Nx1 files can be used in dark or light mode
            use_filenames.append(filename)
        elif kind == "1xN":
            if dark and "dark" in filename:
                use_filenames.append(filename)
            elif not dark and "light" in filename:
                use_filenames.append(filename)

    # train/test split
    random.shuffle(use_filenames)
    total_files = len(use_filenames)
    if kind == "Nx1":
        assert total_files > 700
        split = int(0.8 * total_files)
        train_files = use_filenames[:split]
        val_files = use_filenames[-split:]
    else:
        assert total_files == 1
        train_files = val_files = use_filenames

    train_files = [os.path.join(buffer_dir, kind, p) for p in train_files]
    val_files = [os.path.join(buffer_dir, kind, p) for p in val_files]
    return train_files, val_files


def train(args):
    # Create Env
    if args.experiment in ["ad_dark", "bc_dark", "rl2_dark"]:
        envs = [RoomKeyDoor(dark=True, size=8, max_episode_steps=50) for _ in range(20)]
    else:
        envs = [
            RoomKeyDoor(dark=False, size=8, max_episode_steps=50) for _ in range(20)
        ]

    # Load Correct Source RL Files
    if args.experiment in ["ad_dark", "ad_light"]:
        train_files, val_files = gather_dset_files_sac(
            kind="Nx1", dark=False, buffer_dir=args.buffer_dir
        )
    elif args.experiment in ["bc_dark", "bc_light"]:
        train_files, val_files = gather_dset_files_sac(
            kind="1xN", dark=args.experiment == "bc_dark", buffer_dir=args.buffer_dir
        )
    elif args.experiment == "rl2_dark":
        train_files, val_files = gather_dset_files_rl2(args.buffer_dir)

    # Train
    experiment = Experiment(
        envs=envs,
        run_name=args.run_name,
        gpus=args.gpus,
        train_dset_files=train_files,
        val_dset_files=val_files,
        epochs=args.epochs,
        log_to_wandb=not args.no_log,
        architecture=args.model,
        context_len=args.context_len,
        force_dark=args.experiment == "ad_dark",
        rl2_mode=args.experiment == "rl2_dark",
        log_dir="logs",
        half_precision=False,
    )
    experiment.start()
    experiment.learn()


if __name__ == "__main__":
    args = parse_args()
    train(args)
