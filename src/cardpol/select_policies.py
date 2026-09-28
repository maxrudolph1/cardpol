"""Stage 1b: pick SAC snapshots of graded quality as the data-collecting policies.

Every snapshot saved by `cardpol.sac` is rolled out (stochastic actions, as in
data rendering) and, for each target return, the snapshot whose mean return is
closest is copied to <out>/<domain>/ckpt_<i>.pt. The default targets are the
returns of the paper's policies (common.PAPER_POLICY_RETURNS).

    python -m cardpol.select_policies --domain cheetah --sac-dir runs/sac/cheetah/seed1
"""
import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import torch

from cardpol.common import PAPER_POLICY_RETURNS, RUNS_DIR, TASK
from cardpol.sac import Actor, StateEnv


def evaluate(actor, env, episodes):
    returns = []
    for _ in range(episodes):
        obs, done, ret = env.reset(), False, 0.0
        while not done:
            with torch.no_grad():
                action = actor.get_action(torch.as_tensor(obs)[None])[0][0].numpy()
            obs, reward, done = env.step(action)
            ret += reward
        returns.append(ret)
    return float(np.mean(returns))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--domain', choices=list(TASK), required=True)
    p.add_argument('--sac-dir', type=Path, required=True, help='directory of step_*.pt snapshots')
    p.add_argument('--targets', type=float, nargs='+', default=None, help='default: the paper policies\' returns')
    p.add_argument('--episodes', type=int, default=5, help='evaluation episodes per snapshot')
    p.add_argument('--out', type=Path, default=RUNS_DIR / 'policies')
    p.add_argument('--seed', type=int, default=100)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    targets = args.targets or PAPER_POLICY_RETURNS[args.domain]
    env = StateEnv(args.domain, args.seed)
    snapshots = sorted(args.sac_dir.glob('step_*.pt'))
    assert snapshots, f'no step_*.pt in {args.sac_dir}'
    scores = []
    for path in snapshots:
        scores.append(evaluate(Actor.load(path), env, args.episodes))
        print(f'{path.name}: {scores[-1]:.1f}')

    out = args.out / args.domain
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'selection.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ckpt', 'target_return', 'snapshot', 'eval_return'])
        for i, target in enumerate(targets):
            j = int(np.argmin([abs(s - target) for s in scores]))
            shutil.copy(snapshots[j], out / f'ckpt_{i}.pt')
            writer.writerow([i, target, snapshots[j].name, round(scores[j], 1)])
            print(f'ckpt_{i}: target {target:.0f} -> {snapshots[j].name} ({scores[j]:.1f})')


if __name__ == '__main__':
    main()
