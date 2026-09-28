"""Stage 4: downstream offline RL (DrQ+BC) on top of a frozen pretrained encoder.

The offline dataset is the rendered rollouts (with rewards) of a single policy
(cheetah ckpt 6, walker ckpt 7) under the same distraction the encoder was
trained on; evaluation runs online in that distracting environment.

    python -m cardpol.drqbc --encoder runs/pretrain/cardpol/cheetah_easy_video/seed0 --seed 0

Writes <encoder>/drqbc/seed<seed>/{eval.csv, train.csv, result.json}. The
reported score is final5: the mean of the last 5 evaluations (10 episodes each).
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from cardpol.common import DATA_DIR, FRAME_STACK, OFFLINE_CKPT, RUNS_DIR, SETTINGS, TASK, rollout_path
from cardpol.drqbc.agent import AUGMENTATIONS, DrQBCAgent
from cardpol.drqbc.replay_buffer import EfficientReplayBuffer, load_rollouts
from cardpol.envs import dmc
from cardpol.models import Encoder


def evaluate(agent, env, episodes, step):
    returns = []
    agent.train(False)
    for _ in range(episodes):
        time_step, ret = env.reset(), 0.0
        while not time_step.last():
            time_step = env.step(agent.act(time_step.observation, step))
            ret += time_step.reward
        returns.append(ret)
    agent.train(True)
    return float(np.mean(returns)), float(np.std(returns))


def load_encoder(run_dir):
    config = json.loads((run_dir / 'config.json').read_text())
    encoder = Encoder(3 * FRAME_STACK, config['feature_dim'], config['output_norm'])
    encoder.load_state_dict(torch.load(run_dir / 'encoder.pt', map_location='cpu', weights_only=True))
    return encoder, config


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--encoder', type=Path, default=None, help='a cardpol.pretrain run directory')
    p.add_argument('--from-scratch', action='store_true',
                   help='no pretraining: randomly initialized encoder trained end-to-end (needs --domain/--setting)')
    p.add_argument('--domain', choices=list(TASK), default=None)
    p.add_argument('--setting', choices=list(SETTINGS), default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--steps', type=int, default=500_000, help='gradient steps (the agent updates every 2nd step)')
    p.add_argument('--eval-every', type=int, default=10_000)
    p.add_argument('--eval-episodes', type=int, default=10)
    p.add_argument('--augmentation', choices=list(AUGMENTATIONS), default='shift')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--nstep', type=int, default=3)
    p.add_argument('--offline-ckpt', type=int, default=None, help='policy whose rollouts are the offline data')
    p.add_argument('--data-dir', type=Path, default=DATA_DIR)
    p.add_argument('--out', type=Path, default=None, help='default: <encoder>/drqbc/seed<seed>')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--wandb', action='store_true')
    args = p.parse_args()

    if args.from_scratch:
        assert args.domain and args.setting, '--from-scratch needs --domain and --setting'
        encoder, config = Encoder(3 * FRAME_STACK, 256, 'layernorm'), {'method': 'scratch', 'domain': args.domain,
                                                                   'setting': args.setting}
        default_out = RUNS_DIR / 'pretrain' / 'scratch' / f'{args.domain}_{args.setting}' / 'drqbc' / f'seed{args.seed}'
    else:
        assert args.encoder, 'pass --encoder <pretrain run dir> (or --from-scratch)'
        encoder, config = load_encoder(args.encoder)
        default_out = args.encoder / 'drqbc' / f'seed{args.seed}'
    domain, setting = config['domain'], config['setting']
    out = args.out or default_out
    if (out / 'result.json').exists() and not args.overwrite:
        print(f'{out}/result.json exists, skipping (pass --overwrite to rerun)')
        return
    out.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = dmc.make(domain, TASK[domain], seed=args.seed, distracting_mode=SETTINGS[setting], pixels_only=True)
    agent = DrQBCAgent(encoder, env.action_spec().shape[0], device, augmentation=args.augmentation,
                       freeze_encoder=not args.from_scratch)
    buffer = EfficientReplayBuffer(1_000_000, args.batch_size, args.nstep, discount=0.99, frame_stack=FRAME_STACK)
    offline_ckpt = OFFLINE_CKPT[domain] if args.offline_ckpt is None else args.offline_ckpt
    load_rollouts(rollout_path(domain, setting, 'train', offline_ckpt, args.data_dir), buffer, FRAME_STACK)

    run_config = {**vars(args), 'domain': domain, 'setting': setting, 'offline_ckpt': offline_ckpt,
                  'pretrain': config}
    (out / 'config.json').write_text(json.dumps(run_config, indent=2, default=str))
    if args.wandb:
        import wandb
        wandb.init(project='cardpol', group=f'drqbc_{config["method"]}', name=f'{domain}_{setting}_s{args.seed}',
                   config=run_config)
    eval_file, train_file = open(out / 'eval.csv', 'w', newline=''), open(out / 'train.csv', 'w', newline='')
    eval_log, train_log = csv.writer(eval_file), None
    eval_log.writerow(['step', 'episode_reward', 'episode_return_std'])

    start = time.time()
    for step in range(args.steps):
        if step % args.eval_every == 0:
            mean, std = evaluate(agent, env, args.eval_episodes, step)
            eval_log.writerow([step, round(mean, 2), round(std, 2)])
            eval_file.flush()
            print(f'step {step}: eval return {mean:.1f} +- {std:.1f} ({time.time() - start:.0f}s)', flush=True)
            if args.wandb:
                wandb.log({'eval/episode_reward': mean}, step=step)
        metrics = agent.update(buffer, step)
        if step % 2000 == 0 and metrics:
            if train_log is None:
                train_log = csv.DictWriter(train_file, fieldnames=['step'] + list(metrics))
                train_log.writeheader()
            train_log.writerow({'step': step, **metrics})
            train_file.flush()
    eval_file.close()
    train_file.close()

    with open(out / 'eval.csv') as f:
        rewards = [float(r['episode_reward']) for r in csv.DictReader(f)]
    result = {'method': config['method'], 'domain': domain, 'setting': setting, 'seed': args.seed,
              'final5': float(np.mean(rewards[-5:])), 'last_eval_step': step - step % args.eval_every}
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == '__main__':
    main()
