# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Pixel DMC environments with distractions, adapted from V-D4RL's dmc.py."""
import os
from collections import OrderedDict, deque
from typing import Any, NamedTuple

os.environ.setdefault('MUJOCO_GL', 'egl')

import dm_env
import numpy as np
from dm_control import suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs

from cardpol.common import ACTION_REPEAT, DAVIS_DIR, FRAME_STACK, IMG_SIZE
from cardpol.envs.dcs.suite import distracting_wrapper

DISTRACTING_KWARGS = {
    'clean': {'difficulty': None, 'dynamic': False},
    'easy': {'difficulty': 'easy', 'dynamic': True},
    'fixed_easy': {'difficulty': 'easy', 'dynamic': False},
    'hard': {'difficulty': 'hard', 'dynamic': True},
    'fixed_hard': {'difficulty': 'hard', 'dynamic': False},
}
# The camera never moves, for every distraction.
STATIC_CAMERA = dict(horizontal_delta=0, vertical_delta=0, max_vel=0, vel_std=0, roll_delta=0,
                     max_roll_vel=0, roll_std=0, max_zoom_in_percent=0, max_zoom_out_percent=0)


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST


class _Wrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        return self._env.reset()

    def step(self, action):
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionRepeatWrapper(_Wrapper):
    def __init__(self, env, num_repeats):
        super().__init__(env)
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for _ in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break
        return time_step._replace(reward=reward, discount=discount)


class ActionDTypeWrapper(_Wrapper):
    def __init__(self, env, dtype):
        super().__init__(env)
        spec = env.action_spec()
        self._action_spec = specs.BoundedArray(spec.shape, dtype, spec.minimum, spec.maximum, 'action')

    def step(self, action):
        return self._env.step(action.astype(self._env.action_spec().dtype))

    def action_spec(self):
        return self._action_spec


class FrameStackWrapper(_Wrapper):
    """Stacks the last `num_frames` renders channel-first into obs['pixels'] (or obs if pixels_only)."""

    def __init__(self, env, num_frames, pixels_only=False, pixels_key='pixels'):
        super().__init__(env)
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key
        self._pixels_only = pixels_only
        h, w, c = env.observation_spec()[pixels_key].shape
        self._stacked_spec = specs.BoundedArray(shape=(c * num_frames, h, w), dtype=np.uint8,
                                                minimum=0, maximum=255, name='observation')

    def _transform_observation(self, time_step):
        stacked = np.concatenate(list(self._frames), axis=0)
        if self._pixels_only:
            return time_step._replace(observation=stacked)
        obs = OrderedDict(time_step.observation)
        obs[self._pixels_key] = stacked
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        return time_step.observation[self._pixels_key].transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        self._frames.append(self._extract_pixels(time_step))
        return self._transform_observation(time_step)

    def observation_spec(self):
        if self._pixels_only:
            return self._stacked_spec
        out = OrderedDict(self._env.observation_spec())
        out[self._pixels_key] = self._stacked_spec
        return out


class ExtendedTimeStepWrapper(_Wrapper):
    def reset(self):
        return self._augment(self._env.reset())

    def step(self, action):
        return self._augment(self._env.step(action), action)

    def _augment(self, time_step, action=None):
        if action is None:
            spec = self.action_spec()
            action = np.zeros(spec.shape, dtype=spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation, step_type=time_step.step_type,
                                action=action, reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)


def make(domain, task, seed, distracting_mode='clean', pixels_only=False,
         frame_stack=FRAME_STACK, action_repeat=ACTION_REPEAT, davis_dir=DAVIS_DIR):
    """DMC task rendered at 84x84 with a static camera and the requested distraction.

    distracting_mode: clean | fixed_easy | easy | hard | fixed_hard. Backgrounds
    always come from the 60 DAVIS 2017 *training* videos.
    """
    assert distracting_mode in DISTRACTING_KWARGS, distracting_mode
    env = suite.load(domain, task, task_kwargs={'random': seed}, visualize_reward=False)
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)

    kwargs = DISTRACTING_KWARGS[distracting_mode]
    if kwargs['difficulty'] is not None and not os.path.isdir(davis_dir):
        raise FileNotFoundError(
            f'DAVIS backgrounds not found at {davis_dir}. Run scripts/download_davis.sh '
            f'or set DAVIS_DIR to .../DAVIS/JPEGImages/480p')
    env = distracting_wrapper(
        env, domain,
        difficulty=kwargs['difficulty'],
        dynamic=kwargs['dynamic'],
        background_dataset_path=str(davis_dir),
        background_dataset_videos='train',
        camera_kwargs=STATIC_CAMERA,
        render_kwargs=dict(height=IMG_SIZE, width=IMG_SIZE, camera_id=0),
        pixels_only=False,
    )
    env = FrameStackWrapper(env, frame_stack, pixels_only=pixels_only)
    return ExtendedTimeStepWrapper(env)
