import argparse
import pickle
import os
import random

import gym
import gin
import super_sac
from super_sac.wrappers import (
    SimpleGymWrapper,
    NormActionSpace,
    ParallelActors,
    DiscreteActionWrapper,
)

from dark_key_to_door import RoomKeyDoor


class IdentityEncoder(super_sac.nets.Encoder):
    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def embedding_dim(self):
        return self._dim

    def forward(self, obs_dict):
        return obs_dict["obs"]


def train(args):
    # set RL defaults (discrete SAC w/ aggressive hparams)
    gin.parse_config_file(args.config)

    # 1 env or N envs?
    if args.fixed_env:
        key = random.choices(range(args.room_size), k=2)
        door = random.choices(range(args.room_size), k=2)
    else:
        key = "random"
        door = "random"

    def make_env():
        env = RoomKeyDoor(
            dark=args.dark,
            key_location=key,
            goal_location=door,
            max_episode_steps=args.max_episode_steps,
            size=args.room_size,
        )
        env = DiscreteActionWrapper(env)
        return env

    train_env = SimpleGymWrapper(ParallelActors(make_env, args.parallel_envs))
    test_env = SimpleGymWrapper(make_env())

    # create agent
    agent = super_sac.Agent(
        act_space_size=train_env.action_space.n,
        encoder=IdentityEncoder(train_env.observation_space.shape[0]),
    )
    buffer = super_sac.replay.ReplayBuffer(size=args.steps + 512 + 1)

    # run training
    run_name = f"sac_{'dark' if args.dark else 'light'}_key_{key}_door_{door}_{random.randint(1, 1_0000)}_steps_{args.max_episode_steps}_size_{args.room_size}"
    super_sac.super_sac(
        agent=agent,
        train_env=train_env,
        test_env=test_env,
        buffer=buffer,
        name=run_name,
        logging_method="wandb",
        wandb_entity="jakegrigsby",
        wandb_project="ad",
        max_episode_steps=args.max_episode_steps,
        infinite_bootstrap=False,
        num_steps_online=args.steps,
        random_warmup_steps=512,
        base_save_path=args.log_dir,
    )

    # save data
    data = buffer.get_all_transitions()
    if not os.path.exists(args.buffer_dir):
        os.makedirs(args.buffer_dir)
    with open(f"{os.path.join(args.buffer_dir, run_name)}.buffer", "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--room_size", type=int, default=8)
    parser.add_argument("--max_episode_steps", type=int, default=50)
    parser.add_argument("--fixed_env", action="store_true")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--dark", action="store_true")
    parser.add_argument("--parallel_envs", type=int, default=1)
    parser.add_argument("--config", type=str, default="configs/sac_discrete.gin")
    parser.add_argument("--buffer_dir", type=str, default="buffers")
    parser.add_argument("--log_dir", type=str, default="logs")
    args = parser.parse_args()

    for _ in range(args.trials):
        train(args)
