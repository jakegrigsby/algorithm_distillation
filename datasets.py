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
        force_dark: bool,
        context_length: int,
        action_dim: int = 5,
    ):
        self.filenames = buffer_filenames
        self.force_dark = force_dark
        self.context_length = context_length
        self.action_dim = action_dim

    def __len__(self):
        return len(self.filenames)

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
        r = r[time].astype(np.float32)
        d = d[time].astype(np.float32)

        # offset action sequence by one
        dummy_a = np.zeros_like(a[0])[:, np.newaxis]
        prev_a = np.concatenate((dummy_a, a), axis=0).astype(np.int_)
        # one-hot embed the previous actions that are policy inputs
        prev_a_one_hot = F.one_hot(
            torch.from_numpy(prev_a).squeeze(1), num_classes=self.action_dim
        )
        prev_a_one_hot = prev_a_one_hot[time].numpy().astype(np.float32)

        # non-shifted actions are the BC labels
        a = a[time].astype(np.int_)

        # create meta-rl input format
        seq = np.concatenate((s, prev_a_one_hot, r, d), axis=-1)
        return seq, a


if __name__ == "__main__":

    dset = ADDataset(
        ["buffers/sac_light_key_[3, 0]_door_[7, 1]_3877_steps_50_size_8.buffer"],
        True,
        100,
        5,
    )

    batch = next(iter(dset))
    breakpoint()
