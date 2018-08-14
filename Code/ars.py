#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Python Augmented Random Search Implementation

Dependecies:
    numpy
    gym
    pybullet
"""

# AI 2018 - Augmented Random Search

# Importing the libraries
import os
import numpy as np
import gym
import pybullet_envs
from gym import wrappers


# Setting the Hyper Parameters


class Hp():

    def __init__(self):
        self._nb_steps = 1000
        self._episode_length = 1000
        self._learning_rate = 0.02
        self._nb_directions = 16
        self._nb_best_directions = 16
        assert self._nb_best_directions <= self._nb_directions
        self._noise = 0.03
        self._seed = 1
        self._env_name = 'HalfCheetahBulletEnv-v0'

# Normalizing the states


class Normalizer():

    def __init__(self, nb_inputs):
        self._n = np.zeros(nb_inputs)
        self._mean = np.zeros(nb_inputs)
        self._mean_diff = np.zeros(nb_inputs)
        self._var = np.zeros(nb_inputs)

    def observe(self, x):
        self._n += 1.
        last_mean = self._mean.copy()
        self._mean += (x - self._mean) / self._n
        self._mean_diff += (x - last_mean) * (x - self._mean)
        self._var = (self._mean_diff / self._n).clip(min=1e-2)

    def normalize(self, inputs):
        obs_mean = self._mean  # observed mean
        obs_std = np.sqrt(self._var)  # observed standar deviation
        return (inputs - obs_mean) / obs_std

# Building the AI


class Policy():

    def __init__(self, input_size, output_size):
        self._theta = np.zeros((output_size, input_size))  # matrix of weights

    def evaluate(self, input, delta=None, direction=None):
        if direction is None:
            return self._theta.dot(input)
        elif direction == 'positive':
            return (self._theta + hp._noise*delta).dot(input)
        else:
            return (self._theta - hp._noise*delta).dot(input)

    def sample_deltas(self):
        return [np.random.randn(*self._theta.shape) for _ in range(
                hp._nb_directions)]

    def update(self, rollouts, sigma_r):
        step = np.zeros(self._theta.shape)
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg) * d
        self._theta += hp._learning_rate / (hp._nb_best_directions * sigma_r) \
            * step

# Exploring the policy on one specific direction and over one episode


def explore(env, normalizer, policy, direction=None, delta=None):
    state = env.reset()
    done = False
    num_plays = 0.
    sum_rewards = 0
    while not done and num_plays < hp._episode_length:
        normalizer.observe(state)
        state = normalizer.normalize(state)
        action = policy.evaluate(state, delta, direction)
        state, reward, done, _ = env.step(action)
        reward = max(min(reward, 1), -1)
        sum_rewards += reward
        num_plays += 1
    return sum_rewards

# Training the AI


def train(env, policy, normalizer, hp):

    for step in range(hp._nb_steps):

        # Initializing the perturbations deltas and the positive/negative
        # rewards
        deltas = policy.sample_deltas()
        positive_rewards = [0] * hp._nb_directions
        negative_rewards = [0] * hp._nb_directions

        # Getting the positive rewards in the positive directions
        for k in range(hp._nb_directions):
            positive_rewards[k] = explore(env, normalizer, policy,
                                          direction='positive',
                                          delta=deltas[k])

        # Getting the negative rewards in the negative/opposite directions
        for k in range(hp._nb_directions):
            negative_rewards[k] = explore(env, normalizer, policy,
                                          direction='negative',
                                          delta=deltas[k])

        # Gathering all the positive/negative rewards to compute the standard
        # deviation of these rewards
        all_rewards = np.array(positive_rewards + negative_rewards)
        sigma_r = all_rewards.std()

        # Sorting the rollouts by the max(r_pos, r_neg) and selecting the best
        # directions
        scores = {k: max(r_pos, r_neg) for k, (r_pos, r_neg)
                  in enumerate(zip(positive_rewards, negative_rewards))}
        order = sorted(scores.keys(), key=lambda x: scores[x])[
            :hp._nb_best_directions]
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k])
                    for k in order]

        # Updating our policy
        policy.update(rollouts, sigma_r)

        # Printing the final reward of the policy after the update
        reward_evaluation = explore(env, normalizer, policy)
        print('Step: ', step, 'Reward: ', reward_evaluation)

# Running the main code


def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def main():
    work_dir = mkdir('exp', 'brs')
    monitor_dir = mkdir(work_dir, 'monitor')

    np.random.seed(hp._seed)
    env = gym.make(hp._env_name)
    env = wrappers.Monitor(env, monitor_dir, force=True)
    nb_inputs = env.observation_space.shape[0]
    nb_outputs = env.action_space.shape[0]
    policy = Policy(nb_inputs, nb_outputs)
    normalizer = Normalizer(nb_inputs)
    train(env, policy, normalizer, hp)
    
if __name__ == "__main__":
    hp = Hp()
    main()
