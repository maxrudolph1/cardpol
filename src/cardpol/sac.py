"""Stage 1: train the low-level SAC policies from proprioceptive state.

CleanRL's sac_continuous_action (https://docs.cleanrl.dev/rl-algorithms/sac/)
on dm_control with action repeat 2. The actor is saved every `--save-every`
env steps; `cardpol.select_policies` then picks checkpoints of different
quality from these snapshots.

    python -m cardpol.sac --domain cheetah --seed 1
"""
import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_control import suite

from cardpol.common import ACTION_REPEAT, RUNS_DIR, STATE_KEYS, TASK

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class StateEnv:
    """dm_control task with action repeat; observations are the flattened state."""

    def __init__(self, domain, seed, action_repeat=ACTION_REPEAT):
        self.env = suite.load(domain, TASK[domain], task_kwargs={'random': seed})
        self.keys = STATE_KEYS[domain]
        self.action_repeat = action_repeat
        spec = self.env.action_spec()
        self.action_low, self.action_high = spec.minimum, spec.maximum
        self.obs_dim = len(self.reset())
        self.action_dim = spec.shape[0]

    def _obs(self, time_step):
        return np.concatenate([np.atleast_1d(time_step.observation[k]) for k in self.keys]).astype(np.float32)

    def reset(self):
        return self._obs(self.env.reset())

    def step(self, action):
        reward = 0.0
        for _ in range(self.action_repeat):
            time_step = self.env.step(action)
            reward += time_step.reward or 0.0
            if time_step.last():
                break
        return self._obs(time_step), reward, time_step.last()


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = F.relu(self.fc1(torch.cat([x, a], 1)))
        return self.fc3(F.relu(self.fc2(x)))


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, action_low=-1.0, action_high=1.0):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        low = torch.tensor(np.array(action_low), dtype=torch.float32) * torch.ones(action_dim)
        high = torch.tensor(np.array(action_high), dtype=torch.float32) * torch.ones(action_dim)
        self.register_buffer('action_scale', (high - low) / 2.0)
        self.register_buffer('action_bias', (high + low) / 2.0)

    @classmethod
    def load(cls, path):
        state_dict = torch.load(path, map_location='cpu', weights_only=True)
        actor = cls(state_dict['fc1.weight'].shape[1], state_dict['fc_mean.weight'].shape[0])
        actor.load_state_dict(state_dict)
        return actor

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_logstd(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        """Returns (sampled action, its log-prob, deterministic action)."""
        mean, log_std = self(x)
        normal = torch.distributions.Normal(mean, log_std.exp())
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t) - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob.sum(1, keepdim=True), mean


class ReplayBuffer:
    def __init__(self, size, obs_dim, action_dim):
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.next_obs = np.zeros((size, obs_dim), np.float32)
        self.actions = np.zeros((size, action_dim), np.float32)
        self.rewards = np.zeros(size, np.float32)
        self.size, self.idx, self.full = size, 0, False

    def add(self, obs, next_obs, action, reward):
        self.obs[self.idx], self.next_obs[self.idx] = obs, next_obs
        self.actions[self.idx], self.rewards[self.idx] = action, reward
        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size if self.full else self.idx, size=batch_size)
        return [torch.as_tensor(x[idx], device=device)
                for x in (self.obs, self.actions, self.next_obs, self.rewards)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--domain', choices=list(TASK), required=True)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--total-timesteps', type=int, default=1_000_000, help='env steps (after action repeat)')
    p.add_argument('--buffer-size', type=int, default=1_000_000)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--tau', type=float, default=0.005)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--learning-starts', type=int, default=5000)
    p.add_argument('--policy-lr', type=float, default=3e-4)
    p.add_argument('--q-lr', type=float, default=1e-3)
    p.add_argument('--policy-frequency', type=int, default=2)
    p.add_argument('--save-every', type=int, default=10_000)
    p.add_argument('--out', type=Path, default=None, help='default: runs/sac/<domain>/seed<seed>')
    args = p.parse_args()

    out = args.out or RUNS_DIR / 'sac' / args.domain / f'seed{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = StateEnv(args.domain, args.seed)
    obs_dim, action_dim = env.obs_dim, env.action_dim
    actor = Actor(obs_dim, action_dim, env.action_low, env.action_high).to(device)
    qf1, qf2 = SoftQNetwork(obs_dim, action_dim).to(device), SoftQNetwork(obs_dim, action_dim).to(device)
    qf1_target, qf2_target = SoftQNetwork(obs_dim, action_dim).to(device), SoftQNetwork(obs_dim, action_dim).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.policy_lr)
    # automatic entropy tuning
    target_entropy = -float(action_dim)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.exp().item()
    a_optimizer = torch.optim.Adam([log_alpha], lr=args.q_lr)

    rb = ReplayBuffer(args.buffer_size, obs_dim, action_dim)
    log = open(out / 'returns.csv', 'w', newline='')
    writer = csv.writer(log)
    writer.writerow(['step', 'episode_return'])
    obs, episode_return, last_return = env.reset(), 0.0, 0.0
    start = time.time()
    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            action = np.random.uniform(env.action_low, env.action_high).astype(np.float32)
        else:
            with torch.no_grad():
                action = actor.get_action(torch.as_tensor(obs, device=device)[None])[0][0].cpu().numpy()
        next_obs, reward, last = env.step(action)
        # dm_control episodes only end at the time limit, so never cut the bootstrap
        rb.add(obs, next_obs, action, reward)
        episode_return += reward
        obs = next_obs
        if last:
            last_return = episode_return
            writer.writerow([global_step, round(last_return, 2)])
            log.flush()
            print(f'step={global_step} episode_return={last_return:.1f} sps={global_step / (time.time() - start):.0f}')
            obs, episode_return = env.reset(), 0.0

        if global_step > args.learning_starts:
            b_obs, b_act, b_next, b_rew = rb.sample(args.batch_size, device)
            with torch.no_grad():
                next_act, next_log_pi, _ = actor.get_action(b_next)
                min_q_next = torch.min(qf1_target(b_next, next_act), qf2_target(b_next, next_act)) - alpha * next_log_pi
                next_q = b_rew + args.gamma * min_q_next.view(-1)
            qf_loss = F.mse_loss(qf1(b_obs, b_act).view(-1), next_q) + F.mse_loss(qf2(b_obs, b_act).view(-1), next_q)
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # delayed actor updates, TD3-style
                for _ in range(args.policy_frequency):
                    pi, log_pi, _ = actor.get_action(b_obs)
                    actor_loss = (alpha * log_pi - torch.min(qf1(b_obs, pi), qf2(b_obs, pi))).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(b_obs)
                    alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            for net, target in ((qf1, qf1_target), (qf2, qf2_target)):
                for param, target_param in zip(net.parameters(), target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        if global_step % args.save_every == 0:
            torch.save(actor.state_dict(), out / f'step_{global_step:07d}.pt')
    log.close()


if __name__ == '__main__':
    main()
