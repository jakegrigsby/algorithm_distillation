from argparse import ArgumentParser
from functools import partial

import gin
import gym
import numpy as np
import torch

import gcrl2
from gcrl2.envs import ParallelEnvs, SequenceWrapper, KGoalEnv
from gcrl2.hindsight import GoalSeq

from dark_key_to_door import RoomKeyDoor


def parse_cli():
    parser = ArgumentParser()
    parser.add_argument("--gpus", nargs="*", type=int, default=[0])
    parser.add_argument("--no_log", action="store_true")
    parser.add_argument("--ckpt", type=int, default=None)
    parser.add_argument(
        "--model_config", type=str, default="configs/tiny_rl2_transformer.gin"
    )
    parser.add_argument("--method_config", type=str, default="configs/base_rl2.gin")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--room_size", type=int, default=8)
    parser.add_argument("--max_episode_steps", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--buffer_dir", type=str, default="buffers")
    parser.add_argument("--hide_meta_info", action="store_true")
    args = parser.parse_args()
    return args


def parse_gin(args):
    gin.parse_config_file(args.model_config)
    gin.parse_config_file(args.method_config)
    if args.hide_meta_info:
        gin.bind_parameter("gcrl2.hindsight.Trajectory.show_meta_info", False)
    gin.bind_parameter("gcrl2.learning.Experiment.goal_emb_method", "ff")
    gin.bind_parameter("gcrl2.nets.goal_embedders.FFGoalEmb.goal_emb_dim", 2)
    gin.finalize()


class RL2DarkRoomDoor(KGoalEnv):
    def __init__(self, episodes, room_size, max_steps):
        env = RoomKeyDoor(
            dark=True,
            size=room_size,
            max_episode_steps=max_steps,
            key_location="random",
            goal_location="random",
        )
        self.episodes = episodes
        super().__init__(env, max_timesteps=episodes * max_steps)

    """
    All of the gcrl2 codebase's goal-conditioning is ignored
    """

    @property
    def achieved_goal(self) -> np.ndarray:
        return [np.array([0.0], dtype=np.float32)]

    @property
    def kgoal_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
        )

    @property
    def goal_sequence(self):
        goal_seq = [np.array(0.0)]
        return GoalSeq(seq=goal_seq, active_idx=0)

    def inner_reset(self):
        self.step_count = 0
        self.episode_step_count = 0
        self.current_episode = 0
        self.current_episode_return = 0
        # reset the environment (to a new task)
        obs = self.env.reset(new_task=True)
        return obs

    def step(self, action):
        # when episodes end, reset the environment to the same task, until the global step limit is hit
        return super().step(
            action,
            normal_rl_reward=True,
            normal_rl_reset=False,
            reset_kwargs={"new_task": False},
        )

    def inner_step(self, action):
        assert isinstance(action, np.ndarray)
        assert action.dtype == np.uint8
        goal_seq = self.goal_sequence
        obs, reward, done, info = self.env.step(action)
        self.episode_step_count += 1
        soft_done = done
        self.current_episode_return += reward
        if soft_done:
            self.current_episode += 1
            self.current_episode_return = 0
            self.episode_step_count = 0
        return obs, reward, soft_done, info


def train_rl2(args):

    NAME = f"RL2_size_{args.room_size}_ep_{args.episodes}_steps_{args.max_episode_steps}_meta_info_{not args.hide_meta_info}"

    def make_env(args, split):
        env = RL2DarkRoomDoor(
            episodes=args.episodes,
            room_size=args.room_size,
            max_steps=args.max_episode_steps,
        )
        env = SequenceWrapper(
            env,
            env_name=NAME,
            max_goal_seq_length=1,
            make_dset=True,
            dset_root=args.buffer_dir,
            dset_name=NAME,
            dset_split=split,
        )
        return env

    train_envs = ParallelEnvs([partial(make_env, args, "train") for _ in range(16)])
    val_envs = ParallelEnvs([partial(make_env, args, "val") for _ in range(8)])
    test_envs = val_envs

    experiment = gcrl2.Experiment(
        envs=(train_envs, val_envs, test_envs),
        gpus=args.gpus,
        run_name=NAME,
        dset_root=args.buffer_dir,
        dset_name=NAME,
        log_to_wandb=not args.no_log,
        dloader_workers=10,
        epochs=args.epochs,
        train_timesteps_per_epoch=args.episodes * args.max_episode_steps + 1,
        val_timesteps_per_epoch=2 * args.episodes * args.max_episode_steps + 1,
        train_grad_updates_per_epoch=1000,
        val_interval=15,
        relabel="none",
        noise_start_start=1.0,
        noise_start_final=0.1,
        noise_end_start=0.7,
        noise_end_final=0.02,
        noise_anneal_epochs=args.epochs,
        noise_anneal_steps=args.episodes * args.max_episode_steps,
        max_seq_len=None,
    )
    experiment.start()
    if args.ckpt is not None:
        experiment.load_checkpoint(args.ckpt)
    experiment.learn()
    experiment.load_checkpoint(loading_best=True)
    experiment.evaluate_test(
        timesteps=20 * args.episodes * args.max_episode_steps, render=False
    )


if __name__ == "__main__":
    args = parse_cli()
    parse_gin(args)
    train_rl2(args)
    """
    env = RL2DarkRoomDoor(episodes=3, room_size=8, max_steps=10)
    for _ in range(10):
        env.reset()
        done = False
        while not done:
            env.render()
            action = input()
            raw_action = np.array({"a": 0, "w": 1, "d": 2, "s": 3, "n": 4}[action], dtype=np.uint8)
            next_state, reward, done, info = env.step(raw_action)
            print(next_state)
            print(reward)
            print(done)
    """
