from typing import List
import time
import copy
import math
from itertools import chain

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd
from einops import repeat, rearrange
import numpy as np
import gin
import gym

from networks import Actor


class Agent(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gpus: List[int],
        traj_encoder: nn.Module,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.emb_dim = traj_encoder.emb_dim

        self.gpus = gpus
        self.dp = len(gpus) > 0
        self.traj_encoder = (
            nn.DataParallel(traj_encoder, device_ids=gpus) if self.dp else traj_encoder
        )
        if self.dp:
            self.traj_encoder.emb_dim = traj_encoder.emb_dim
        self.needs_mask = traj_encoder.needs_mask
        self.actor = Actor(
            state_dim=traj_encoder.emb_dim,
            action_dim=action_dim,
        )

    def make_attn_mask(self, x):
        if not self.needs_mask:
            return None
        # mask future tokens
        # speedup: make diagonal, move, then expand batch
        batch, length, dim = x.shape
        mask = torch.triu(
            torch.ones((length, length), dtype=torch.bool), diagonal=1
        ).to(x.device)
        mask = mask.repeat(batch, 1, 1)
        return mask

    @property
    def trainable_params(self):
        return chain(
            self.actor.parameters(),
            self.traj_encoder.parameters(),
        )

    def get_actions(self, states, sample=True):
        attn_mask = self.make_attn_mask(states)
        traj_emb = self.traj_encoder(states, mask=attn_mask)
        traj_emb_t = traj_emb[:, -1, :]
        action_dists = self.actor(traj_emb_t)
        if sample:
            actions = action_dists.sample()
        else:
            actions = torch.argmax(action_dists.probs, dim=-1, keepdim=True)
        return actions

    def forward(self, states, actions, log_step: bool):
        self.update_info = {}

        B, L, D_state = states.shape
        assert D_state == self.state_dim
        D_emb = self.traj_encoder.emb_dim

        attn_mask = self.make_attn_mask(states)
        s_rep = self.traj_encoder(states=states, mask=attn_mask)
        assert s_rep.shape == (B, L, D_emb)
        a_dist = self.actor(s_rep)
        loss = (-a_dist.log_prob(actions.squeeze(-1))).mean()
        return loss
