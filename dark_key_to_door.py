import random
from typing import Tuple

import numpy as np
import gym


class RoomKeyDoor(gym.Env):
    def __init__(
        self,
        dark: bool,
        size: int,
        max_episode_steps: int,
        key_location: Tuple[int, int] = "random",
        goal_location: Tuple[int, int] = "random",
    ):
        self.dark = dark
        self.size = size
        self.H = max_episode_steps
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(4 if self.dark else 8,)
        )
        self.action_space = gym.spaces.Discrete(5)
        self.goal_location = goal_location
        self.key_location = key_location

    def reset(self, new_task=True):
        if new_task:
            key, goal = self.generate_task()
            self.goal = np.array(goal)
            self.key = np.array(key)
        self.has_key = False
        self.t = 0
        self.pos = np.array([0, 0])
        return self.obs()

    def generate_task(self):
        key = (
            random.choices(range(self.size), k=2)
            if self.key_location == "random"
            else self.key_location
        )
        goal = (
            random.choices(range(self.size), k=2)
            if self.goal_location == "random"
            else self.goal_location
        )
        return key, goal

    def step(self, action: int):
        dirs = [[0, -1], [-1, 0], [0, 1], [1, 0], [0, 0]]
        self.pos = np.clip(self.pos + np.array(dirs[action]), 0, self.size - 1)
        reward = 0.0
        done = False
        if self.has_key and (self.pos == self.goal).all():
            reward = 1.0
            done = True
        elif not self.has_key and (self.pos == self.key).all():
            reward = 1.0
            self.has_key = True
        if self.t >= self.H:
            done = True
        self.t += 1
        return self.obs(), reward, done, {}

    def obs(self):
        x, y = self.pos
        norm = lambda j: float(j) / self.size
        base = [norm(x), norm(y), self.has_key, float(self.t) / self.H]
        if not self.dark:
            goal_x, goal_y = self.goal
            key_x, key_y = self.key
            base += [norm(goal_x), norm(goal_y), norm(key_x), norm(key_y)]
        return np.array(base, dtype=np.float32)

    def render(self, *args, **kwargs):
        img = [["." for _ in range(self.size)] for _ in range(self.size)]
        player_x, player_y = self.pos
        goal_x, goal_y = self.goal
        key_x, key_y = self.key
        img[player_x][player_y] = "P"
        img[goal_x][goal_y] = "G"
        img[key_x][key_y] = "K"
        print(
            f"{'Dark' if self.dark else 'Light'} Room Key-Door: Key = {self.key}, Door = {self.goal}, Player = {self.pos}"
        )
        for row in img:
            print(" ".join(row))


if __name__ == "__main__":
    env = RoomKeyDoor(
        dark=False,
        size=9,
        max_episode_steps=50,
        key_location="random",
        goal_location="random",
    )

    for _ in range(10):
        env.reset()
        done = False
        while not done:
            env.render()
            action = input()
            raw_action = {"a": 0, "w": 1, "d": 2, "s": 3, "n": 4}[action]
            next_state, reward, done, info = env.step(raw_action)
            print(next_state)
            print(reward)
            print(done)
