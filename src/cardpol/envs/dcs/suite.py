# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Distraction wrappers from the Distracting Control Suite (Stone et al., 2021).

Trimmed to `distracting_wrapper`; background.py, camera.py, color.py and
suite_utils.py are unmodified apart from the upstream tf -> PIL/cv2 port.
"""
from dm_control.suite.wrappers import pixels

from cardpol.envs.dcs import background, camera, color, suite_utils


def distracting_wrapper(
        env,
        domain_name,
        difficulty=None,
        dynamic=False,
        background_dataset_path=None,
        background_dataset_videos="train",
        background_kwargs=None,
        camera_kwargs=None,
        color_kwargs=None,
        render_kwargs=None,
        pixels_only=False,
        pixels_observation_key="pixels",
        env_state_wrappers=None,
        #
        fixed_distraction=False,
        background_seed=None,
        color_seed=None,
        camera_seed=None):
    # Apply background distractions.
    if difficulty or background_kwargs:
        background_dataset_path = (
                background_dataset_path or suite_utils.DEFAULT_BACKGROUND_PATH)
        final_background_kwargs = dict(fixed=fixed_distraction, seed=background_seed)

        if difficulty:
            # Get kwargs for the given difficulty.
            num_videos = suite_utils.DIFFICULTY_NUM_VIDEOS[difficulty]
            final_background_kwargs.update(
                suite_utils.get_background_kwargs(domain_name, num_videos, dynamic,
                                                  background_dataset_path,
                                                  background_dataset_videos))
        else:
            # Set the dataset path and the videos.
            final_background_kwargs.update(
                dict(dataset_path=background_dataset_path,
                     dataset_videos=background_dataset_videos))
        if background_kwargs:
            # Overwrite kwargs with those passed here.
            final_background_kwargs.update(background_kwargs)
        env = background.DistractingBackgroundEnv(env, **final_background_kwargs)

    # Apply camera distractions.
    if difficulty or camera_kwargs:
        final_camera_kwargs = dict(camera_id=render_kwargs["camera_id"], fixed=fixed_distraction, seed=camera_seed)
        # final_camera_kwargs = dict(camera_id=render_kwargs["camera_id"], dynamic=True, seed=camera_seed)
        if difficulty:
            # Get kwargs for the given difficulty.
            scale = suite_utils.DIFFICULTY_SCALE[difficulty]
            final_camera_kwargs.update(
                suite_utils.get_camera_kwargs(domain_name, scale, dynamic))
        if camera_kwargs:
            # Overwrite kwargs with those passed here.
            final_camera_kwargs.update(camera_kwargs)
        env = camera.DistractingCameraEnv(env, **final_camera_kwargs)

    # Apply color distractions.
    if difficulty or color_kwargs:
        final_color_kwargs = dict(fixed=fixed_distraction, seed=color_seed)
        # final_color_kwargs = dict(dynamic=True, seed=color_seed)
        if difficulty:
            # Get kwargs for the given difficulty.
            scale = suite_utils.DIFFICULTY_SCALE[difficulty]
            final_color_kwargs.update(suite_utils.get_color_kwargs(scale, dynamic))
        if color_kwargs:
            # Overwrite kwargs with those passed here.
            final_color_kwargs.update(color_kwargs)
        env = color.DistractingColorEnv(env, **final_color_kwargs)

    if env_state_wrappers is not None:
        for wrapper in env_state_wrappers:
            env = wrapper(env)
    # Apply Pixel wrapper after distractions. This is needed to ensure the
    # changes from the distraction wrapper are applied to the MuJoCo environment
    # before the rendering occurs.

    env = pixels.Wrapper(
        env,
        pixels_only=pixels_only,
        render_kwargs=render_kwargs,
        observation_key=pixels_observation_key)

    return env
