import os
import contextlib
import random
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Tuple, List

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb
import numpy as np
from tqdm import tqdm
import gym
import gin
import matplotlib.pyplot as plt

from networks import TransformerEncoder, FeedForwardEncoder
from agent import Agent
from datasets import ADDataset


@dataclass
class Experiment:
    envs: List[gym.Env]

    # General
    run_name: str
    log_dir: str
    gpus: List[int]
    train_dset_files: List[str]
    val_dset_files: List[str]
    log_to_wandb: bool = False

    # Method
    architecture: str = "transformer"
    context_len: int = 800
    force_dark: bool = True

    # Learning Schedule
    epochs: int = 100
    train_grad_updates_per_epoch: int = 1000
    eval_timesteps: int = 100
    val_interval: int = 2
    val_checks_per_epoch: int = 50
    log_interval: int = 250

    # Optimization
    batch_size: int = 24
    dloader_workers: int = 12
    init_learning_rate: float = 5e-4
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    l2_coeff: float = 1e-4
    half_precision: bool = True

    def start(self):
        self.DEVICE = torch.device(f"cuda:{self.gpus[0]}")
        plt.switch_backend("agg")
        plt.style.use("seaborn-whitegrid")
        self.init_dsets()
        self.init_dloaders()
        self.init_model()
        self.init_optimizer()
        self.init_logger()

    def init_dsets(self):
        self.train_dset = ADDataset(
            buffer_filenames=self.train_dset_files,
            force_dark=self.force_dark,
            context_length=self.context_len,
        )
        self.val_dset = ADDataset(
            buffer_filenames=self.val_dset_files,
            force_dark=self.force_dark,
            context_length=self.context_len,
        )

    def init_dloaders(self):
        self.train_dloader = DataLoader(
            self.train_dset,
            batch_size=self.batch_size,
            num_workers=self.dloader_workers,
            pin_memory=True,
        )
        self.val_dloader = DataLoader(
            self.val_dset,
            batch_size=self.batch_size,
            num_workers=self.dloader_workers,
            pin_memory=True,
        )

    def init_logger(self):
        gin_config = gin.operative_config_str()
        config_path = os.path.join(self.log_dir, self.run_name)
        if not os.path.exists(config_path):
            os.makedirs(config_path)
        with open(os.path.join(config_path, "config.txt"), "w") as f:
            f.write(gin_config)
        if self.log_to_wandb:
            wandb.init(
                project="ad",
                entity="jakegrigsby",
                dir=self.log_dir,
                name=self.run_name,
            )
            wandb.save(os.path.join(config_path, "config.txt"))

    def init_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.policy.trainable_params,
            lr=self.init_learning_rate,
            weight_decay=self.l2_coeff,
        )
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=self.train_grad_updates_per_epoch * self.warmup_epochs,
        )
        self.grad_scaler = torch.cuda.amp.GradScaler(
            enabled=self.half_precision,
            init_scale=10000.0,
            growth_factor=1.5,
            growth_interval=self.train_grad_updates_per_epoch,
        )

    def init_model(self):
        action_size = self.envs[0].action_space.n
        state_size = self.envs[0].observation_space.shape[0] + action_size + 1 + 1

        if self.architecture == "transformer":
            traj_encoder = TransformerEncoder(
                state_dim=state_size,
                max_seq_len=self.context_len,
            )
        elif self.architecture == "feedforward":
            traj_encoder = FeedForwardEncoder(state_dim=state_size)
        else:
            raise NotImplementedError()

        self.policy = Agent(
            state_dim=state_size,
            action_dim=action_size,
            traj_encoder=traj_encoder,
            gpus=self.gpus,
        )
        self.policy.to(self.DEVICE)

        total_params = 0
        for name, parameter in self.policy.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            total_params += params
        print(f"Total Trainable Params: {total_params}")

    def interact(
        self,
        envs: List[gym.Env],
        timesteps: int,
        verbose: bool = False,
        render: bool = False,
    ):
        # evaluation is done in fake parallel mode
        num_envs = len(envs)
        context = []
        current_returns = [0.0 for _ in range(num_envs)]
        return_history = [[] for _ in range(num_envs)]

        # fill each context with the starting observation
        init_context = []
        init_action = torch.zeros(
            (1, envs[0].action_space.n), device=self.DEVICE
        ).float()
        init_done = torch.zeros((1, 1), device=self.DEVICE).float()
        init_rew = torch.zeros_like(init_done)
        for i, env in enumerate(envs):
            init_state = (
                torch.from_numpy(env.reset(new_task=True))
                .to(self.DEVICE)
                .float()
                .unsqueeze(0)
            )
            init_seq = torch.cat((init_state, init_action, init_rew, init_done), dim=-1)
            init_context.append(init_seq)
        context.append(torch.stack(init_context, dim=0))

        if verbose:
            iter_ = tqdm(
                range(timesteps),
                desc="Env Interaction",
                total=timesteps,
                leave=False,
                colour="yellow",
            )
        else:
            iter_ = range(timesteps)

        for step in iter_:

            # gather up to the last `context_len` timesteps
            current_seq = torch.cat(context[-self.context_len :], dim=1)
            with torch.no_grad():
                with self.caster():
                    # sample actions corresponding to the current timestep
                    action = self.policy.get_actions(states=current_seq)

            # execute those actions in each env
            next_context = []
            for i, env in enumerate(envs):
                raw_action = int(action[i].cpu().item())
                next_state, rew, done, _ = env.step(raw_action)
                current_returns[i] += rew
                if done:
                    return_history[i].append((step, current_returns[i]))
                    current_returns[i] = 0.0
                    # keep the task the same!
                    next_state = env.reset(new_task=False)

                # build meta-info for this timestep in this env
                ctxt_action = torch.zeros(
                    (1, env.action_space.n), device=self.DEVICE
                ).float()
                ctxt_action[:, raw_action] = 1.0
                ctxt_state = (
                    torch.from_numpy(next_state).to(self.DEVICE).float().unsqueeze(0)
                )
                ctxt_rew = torch.Tensor([rew]).to(self.DEVICE).float().unsqueeze(0)
                ctxt_done = torch.Tensor([done]).to(self.DEVICE).float().unsqueeze(0)
                next_ctxt_i = torch.cat(
                    (ctxt_state, ctxt_action, ctxt_rew, ctxt_done), dim=-1
                )
                next_context.append(next_ctxt_i)
            # extend the context sequence by 1
            context.append(torch.stack(next_context, dim=0))

            if render:
                # this only renders the last of the "parallel" envs
                env.render()

        return return_history

    def evaluate_val(self):
        all_returns = self.interact(
            self.envs, self.eval_timesteps, verbose=True, render=True
        )
        figures = self.make_evaluation_figures(all_returns)
        self.log(figures, "val")

    def make_evaluation_figures(self, return_history):
        fig = plt.figure()
        ax = plt.axes()
        for env_history in return_history:
            steps = [d[0] for d in env_history]
            returns = [d[1] for d in env_history]
            ax.plot(steps, returns)
        ax.set_xlabel("Meta-Learning Timesteps")
        ax.set_ylabel("Episode Return")
        plt.tight_layout()
        img = wandb.Image(fig)
        plt.close()
        return {"evaluation": img}

    def log(self, metrics_dict, key):
        log_dict = {}
        for k, v in metrics_dict.items():
            if isinstance(v, torch.Tensor):
                if v.ndim == 0:
                    log_dict[k] = v.detach().cpu().float().item()
            else:
                log_dict[k] = v

        if self.log_to_wandb:
            wandb.log({f"{key}/{subkey}": val for subkey, val in log_dict.items()})

    def compute_loss(self, seq, actions, log_step: bool):
        with self.caster():
            loss = self.policy(seq, actions)
        update_info = self.policy.update_info
        return {"loss": loss} | update_info

    def _get_grad_norms(self):
        ggn = utils.get_grad_norm
        grads = dict(
            actor_grad_norm=ggn(self.policy.actor),
            traj_encoder_grad_norm=ggn(self.policy.traj_encoder),
        )
        print(grads)
        return grads

    def train_step(self, batch: Tuple[torch.Tensor], log_step: bool):
        l = self.compute_loss(*batch)
        self.optimizer.zero_grad(set_to_none=True)
        self.grad_scaler.scale(l["loss"]).backward()
        self.grad_scaler.unscale_(self.optimizer)
        if log_step:
            l.update(self._get_grad_norms())
        nn.utils.clip_grad_norm_(self.policy.trainable_params, max_norm=self.grad_clip)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        if self.half_precision and log_step:
            l["grad_scaler_scale"] = self.grad_scaler.get_scale()

        return l

    def caster(self):
        if self.half_precision:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            return contextlib.suppress()

    def val_step(self, batch, epoch: int):
        with torch.no_grad():
            return self.compute_loss(*batch, log_step=True)

    def learn(self):
        def make_pbar(loader, training, epoch):
            if training:
                desc = f"{self.run_name} Epoch {epoch} Train"
                steps = self.train_grad_updates_per_epoch
                c = "green"
            else:
                desc = f"{self.run_name} Epoch {epoch} Val"
                steps = self.val_checks_per_epoch
                c = "red"
            return tqdm(enumerate(loader), desc=desc, total=steps, colour=c)

        for epoch in range(self.epochs):
            if epoch % self.val_interval == 0:
                self.policy.eval()
                self.evaluate_val()

            self.policy.train()
            for train_step, batch in make_pbar(self.train_dloader, True, epoch):
                total_step = (epoch * self.train_grad_updates_per_epoch) + train_step
                log_step = total_step % self.log_interval == 0
                loss_dict = self.train_step(batch, epoch=epoch, log_step=log_step)
                if log_step:
                    self.log(loss_dict, key="train")
                # lr scheduling done here so we can see epoch/step
                self.warmup_scheduler.step(total_step)

            if epoch % self.val_interval == 0:
                self.policy.eval()
                for val_step, batch in make_pbar(self.val_dloader, False, epoch):
                    loss_dict = self.val_step(batch, epoch=epoch)
                    self.log(loss_dict, key="val")
                self.log(figures, key="val")
