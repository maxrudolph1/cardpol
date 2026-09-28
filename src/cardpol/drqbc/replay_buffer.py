# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""V-D4RL's numpy replay buffer (frames stored once, stacked on sampling) and offline-data loading."""
from collections import deque

import h5py
import numpy as np
from dm_env import StepType

from cardpol.envs.dmc import ExtendedTimeStep


class EfficientReplayBuffer:
    def __init__(self, buffer_size, batch_size, nstep, discount, frame_stack):
        self.buffer_size = buffer_size
        self.index = -1
        self.traj_index = 0
        self.frame_stack = frame_stack
        self.batch_size = batch_size
        self.nstep = nstep
        self.full = False
        self.discount_vec = np.power(discount, np.arange(nstep))
        self.next_dis = discount ** nstep

    def _initial_setup(self, time_step):
        self.index = 0
        self.obs_shape = list(time_step.observation.shape)
        self.ims_channels = self.obs_shape[0] // self.frame_stack
        self.obs = np.zeros([self.buffer_size, self.ims_channels, *self.obs_shape[1:]], dtype=np.uint8)
        self.act = np.zeros([self.buffer_size, *time_step.action.shape], dtype=np.float32)
        self.rew = np.zeros([self.buffer_size], dtype=np.float32)
        self.dis = np.zeros([self.buffer_size], dtype=np.float32)
        self.valid = np.zeros([self.buffer_size], dtype=np.bool_)

    def add(self, time_step):
        if self.index == -1:
            self._initial_setup(time_step)
        latest_obs = time_step.observation[-self.ims_channels:]
        if time_step.first():
            end_index = self.index + self.frame_stack
            end_invalid = end_index + self.frame_stack + 1
            if end_invalid > self.buffer_size:
                if end_index > self.buffer_size:
                    end_index = end_index % self.buffer_size
                    self.obs[self.index:self.buffer_size] = latest_obs
                    self.obs[0:end_index] = latest_obs
                    self.full = True
                else:
                    self.obs[self.index:end_index] = latest_obs
                end_invalid = end_invalid % self.buffer_size
                self.valid[self.index:self.buffer_size] = False
                self.valid[0:end_invalid] = False
            else:
                self.obs[self.index:end_index] = latest_obs
                self.valid[self.index:end_invalid] = False
            self.index = end_index
            self.traj_index = 1
        else:
            np.copyto(self.obs[self.index], latest_obs)
            np.copyto(self.act[self.index], time_step.action)
            self.rew[self.index] = time_step.reward
            self.dis[self.index] = time_step.discount
            self.valid[(self.index + self.frame_stack) % self.buffer_size] = False
            if self.traj_index >= self.nstep:
                self.valid[(self.index - self.nstep + 1) % self.buffer_size] = True
            self.index += 1
            self.traj_index += 1
            if self.index == self.buffer_size:
                self.index = 0
                self.full = True

    def __next__(self):
        indices = np.random.choice(self.valid.nonzero()[0], size=self.batch_size)
        ranges = (indices[:, None] + np.arange(-self.frame_stack, self.nstep)[None]) % self.buffer_size
        reward_ranges = ranges[:, self.frame_stack:]
        obs_ranges = ranges[:, :self.frame_stack]
        next_obs_ranges = ranges[:, -self.frame_stack:]
        rew = np.sum(self.rew[reward_ranges] * self.discount_vec, axis=1, keepdims=True)
        obs = np.reshape(self.obs[obs_ranges], [len(indices), *self.obs_shape])
        next_obs = np.reshape(self.obs[next_obs_ranges], [len(indices), *self.obs_shape])
        dis = np.expand_dims(self.next_dis * self.dis[next_obs_ranges[:, -1]], axis=-1)
        return obs, self.act[indices], rew, dis, next_obs

    def __len__(self):
        return self.buffer_size if self.full else self.index


def load_rollouts(path, replay_buffer, frame_stack):
    """Add a cardpol.render rollout file to the buffer.

    Rollout files store (frame_t, action_t taken there, reward_t received after it);
    the buffer expects each time step to carry the action and reward that *led*
    to it, so both are shifted forward by one step.
    """
    with h5py.File(path, 'r') as f:
        data = {k: f[k][:] for k in ('frames', 'action', 'reward', 'step_type')}
    step_type = data['step_type']
    action = np.zeros_like(data['action'])
    reward = np.zeros_like(data['reward'])
    acted = np.nonzero(step_type != StepType.LAST)[0]
    action[acted + 1] = data['action'][acted]
    reward[acted + 1] = data['reward'][acted]
    frames = data['frames'].transpose(0, 3, 1, 2)
    for t in range(len(step_type)):
        time_step = ExtendedTimeStep(step_type=StepType(int(step_type[t])), reward=reward[t], discount=1.0,
                                     observation=frames[t], action=action[t])
        if time_step.first():
            stack = deque([frames[t]] * frame_stack, maxlen=frame_stack)
        else:
            stack.append(frames[t])
        replay_buffer.add(time_step._replace(observation=np.concatenate(stack, axis=0)))
    print(f'loaded {len(step_type)} offline steps from {path}')
