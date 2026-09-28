# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""DrQ+BC (V-D4RL, Lu et al. 2022): DrQ-v2 with the TD3+BC actor loss, on a (frozen) pretrained encoder."""
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal

from cardpol.models import weight_init


class RandomShiftsAug(nn.Module):
    def __init__(self, pad=4):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        x = F.pad(x, tuple([self.pad] * 4), 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)
        return F.grid_sample(x, base_grid + shift, padding_mode='zeros', align_corners=False)


class RandomNoiseAug(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        return x + torch.randn_like(x) * self.std


AUGMENTATIONS = {
    'shift': lambda: RandomShiftsAug(pad=4),                                     # V-D4RL default; used in the paper
    'shift+noise': lambda: nn.Sequential(RandomShiftsAug(pad=4), RandomNoiseAug(0.1)),
    'none': lambda: nn.Identity(),
}


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low, self.high, self.eps = low, high, eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        return x - x.detach() + clamped_x.detach()

    def sample(self, clip=None, sample_shape=torch.Size()):
        eps = _standard_normal(self._extended_shape(sample_shape), dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        return self._clamp(self.loc + eps)


def schedule(schdl, step):
    """'linear(init,final,duration)' or a constant."""
    try:
        return float(schdl)
    except ValueError:
        init, final, duration = [float(g) for g in re.match(r'linear\((.+),(.+),(.+)\)', schdl).groups()]
        mix = np.clip(step / duration, 0.0, 1.0)
        return (1.0 - mix) * init + mix * final


class Actor(nn.Module):
    def __init__(self, repr_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())
        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_dim))
        self.apply(weight_init)

    def forward(self, obs, std):
        mu = torch.tanh(self.policy(self.trunk(obs)))
        return TruncatedNormal(mu, torch.ones_like(mu) * std)


class Critic(nn.Module):
    def __init__(self, repr_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())

        def q_net():
            return nn.Sequential(nn.Linear(feature_dim + action_dim, hidden_dim), nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim, 1))
        self.Q1, self.Q2 = q_net(), q_net()
        self.apply(weight_init)

    def forward(self, obs, action):
        h_action = torch.cat([self.trunk(obs), action], dim=-1)
        return self.Q1(h_action), self.Q2(h_action)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class DrQBCAgent:
    def __init__(self, encoder, action_dim, device, lr=3e-4, feature_dim=256, hidden_dim=1024,
                 critic_target_tau=0.01, update_every_steps=2, stddev_schedule='linear(1.0,0.1,250000)',
                 stddev_clip=0.3, bc_weight=2.5, augmentation='shift', freeze_encoder=True):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.bc_weight = bc_weight
        self.freeze_encoder = freeze_encoder

        self.encoder = encoder.to(device)
        repr_dim = encoder.repr_dim
        self.actor = Actor(repr_dim, action_dim, feature_dim, hidden_dim).to(device)
        self.critic = Critic(repr_dim, action_dim, feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(repr_dim, action_dim, feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        if not freeze_encoder:
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.aug = AUGMENTATIONS[augmentation]()
        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        for m in (self.encoder, self.actor, self.critic):
            m.train(training)

    @staticmethod
    def normalize_obs(obs):
        return obs / 255.0 * 2.0 - 1.0

    @torch.no_grad()
    def act(self, obs, step):
        """Deterministic (mean) action for evaluation."""
        obs = self.normalize_obs(torch.as_tensor(obs, device=self.device))
        dist = self.actor(self.encoder(obs.unsqueeze(0)), schedule(self.stddev_schedule, step))
        return dist.mean.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        with torch.no_grad():
            dist = self.actor(next_obs, schedule(self.stddev_schedule, step))
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_Q = reward.float() + discount * torch.min(target_Q1, target_Q2)
        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if not self.freeze_encoder:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if not self.freeze_encoder:
            self.encoder_opt.step()
        return {'critic_loss': critic_loss.item(), 'critic_q1': Q1.mean().item(),
                'critic_target_q': target_Q.mean().item()}

    def update_actor(self, obs, behavioural_action, step):
        dist = self.actor(obs, schedule(self.stddev_schedule, step))
        action = dist.sample(clip=self.stddev_clip)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)
        bc_loss = F.mse_loss(action, behavioural_action)
        lam = self.bc_weight / Q.detach().abs().mean()  # TD3+BC, Eq. 5 of arXiv:2106.06860
        actor_loss = -Q.mean() * lam + bc_loss
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        return {'actor_q_loss': -Q.mean().item(), 'actor_bc_loss': bc_loss.item()}

    def update(self, replay_buffer, step):
        if step % self.update_every_steps != 0:
            return {}
        obs, action, reward, discount, next_obs = (torch.as_tensor(x, device=self.device)
                                                   for x in next(replay_buffer))
        obs = self.aug(self.normalize_obs(obs).float())
        next_obs = self.aug(self.normalize_obs(next_obs).float())
        if self.freeze_encoder:
            with torch.no_grad():
                obs = self.encoder(obs)
        else:
            obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        metrics = {'batch_reward': reward.mean().item()}
        metrics.update(self.update_critic(obs, action, reward, discount, next_obs, step))
        metrics.update(self.update_actor(obs.detach(), action.detach(), step))
        soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
        return metrics
