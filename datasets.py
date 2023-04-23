from typing import List
import random
import pickle

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np


class ADDataset(Dataset):
    def __init__(
        self,
        buffer_filenames: List[str],
        epoch_length: int,
        force_dark: bool,
        context_length: int,
        action_dim: int = 5,
    ):
        self.filenames = buffer_filenames
        self.force_dark = force_dark
        self.context_length = context_length
        self.action_dim = action_dim
        self.epoch_length = epoch_length

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, i):
        # pick a random file from disk so that we can only use 1 file (Nx1 mode) and have long epochs
        filename = random.choice(self.filenames)

        with open(filename, "rb") as f:
            o, a, r, o1, d = pickle.load(f)
        # select random slice equal to context length
        L = len(a)
        random_start = random.randint(0, L - self.context_length - 1)
        time = slice(random_start, random_start + self.context_length)
        s = o["obs"][time].astype(np.float32)
        if self.force_dark:
            # manually remove key and door information, which does not change
            # the data because the single-task agents never saw these values change
            # during their individual runs.
            s = s[:, :4]

        # offset these sequences by one
        dummy_r = np.zeros((1, 1)).astype(np.float32)
        r = np.concatenate((dummy_r, r), axis=0)[time]
        d = np.concatenate((dummy_r, d), axis=0)[time]
        dummy_a = np.zeros_like(a[0])[:, np.newaxis]
        prev_a = np.concatenate((dummy_a, a), axis=0).astype(np.int_)[time]
        # one-hot embed the previous actions that are policy inputs
        prev_a_one_hot = F.one_hot(
            torch.from_numpy(prev_a).squeeze(1), num_classes=self.action_dim
        ).float()

        # non-shifted actions are the BC labels
        a = a[time].astype(np.int_)

        # create meta-rl input format; (s, a, r, d) order needs to
        # be consistent in environment loop
        seq = np.concatenate((s, prev_a_one_hot, r, d), axis=-1)
        return seq, a


if __name__ == "__main__":

    dset = ADDataset(
        ["buffers/Nx1/sac_light_key_[3, 0]_door_[7, 1]_3877_steps_50_size_8.buffer"],
        20,
        True,
        100,
        5,
    )

    batch = next(iter(dset))
    breakpoint()
