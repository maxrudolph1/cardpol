"""Stage 2: roll out the SAC policies and render 84x84 frames with distractions.

Each policy acts from proprioceptive state (sampled, not mean, actions), while
the pixels are rendered under the chosen distraction. For every policy
checkpoint this writes one merged file

    data/rollouts/<domain>/<setting>/<split>/ckpt_<i>.hdf5
        frames        uint8   (N, 84, 84, 3)
        state         float32 (N, state_dim)
        action        float32 (N, 6)    action taken at this step (0 at the last step)
        reward        float32 (N,)      reward received after `action` (0 at the last step)
        step_type     int8    (N,)      0 first / 1 mid / 2 last (501 steps per episode)
        episode_index int32   (N,)

    python -m cardpol.render --domain cheetah --setting easy_video --workers 16
"""
import argparse
import multiprocessing as mp
import shutil
from pathlib import Path

import h5py
import numpy as np

from cardpol.common import (DATA_DIR, OFFLINE_CKPT, POLICY_DIR, PRETRAIN_CKPTS, SETTINGS,
                            STATE_KEYS, TASK, policy_path, rollout_path)

KEYS = {'frames': np.uint8, 'state': np.float32, 'action': np.float32, 'reward': np.float32,
        'step_type': np.int8}
SPLIT_SEED = {'train': 0, 'val': 10_000}


def rollout_episode(env, actor, domain):
    import torch
    keys = STATE_KEYS[domain]
    ep = {k: [] for k in KEYS}
    time_step = env.reset()
    while True:
        obs = time_step.observation
        state = np.concatenate([np.atleast_1d(obs[k]) for k in keys]).astype(np.float32)
        ep['frames'].append(obs['pixels'][-3:].transpose(1, 2, 0))  # newest frame of the stack
        ep['state'].append(state)
        ep['step_type'].append(int(time_step.step_type))
        if time_step.last():
            ep['action'].append(np.zeros_like(ep['action'][-1]))
            ep['reward'].append(0.0)
            break
        with torch.no_grad():
            action = actor.get_action(torch.as_tensor(state)[None])[0][0].numpy()
        time_step = env.step(action)
        ep['action'].append(action)
        ep['reward'].append(time_step.reward)
    return {k: np.asarray(v, dtype=KEYS[k]) for k, v in ep.items()}


def worker(worker_id, episodes, policy, domain, distracting_mode, seed, tmp_dir):
    import torch
    from cardpol.envs import dmc
    from cardpol.sac import Actor
    torch.manual_seed(seed)
    np.random.seed(seed)
    actor = Actor.load(policy)
    env = dmc.make(domain, TASK[domain], seed=seed, distracting_mode=distracting_mode)
    for i in episodes:
        ep = rollout_episode(env, actor, domain)
        np.savez(tmp_dir / f'episode_{i:05d}.npz', **ep)
        print(f'[worker {worker_id}] episode {i}: return {ep["reward"].sum():.1f}', flush=True)


def merge(tmp_dir, num_episodes, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_suffix('.hdf5.partial')
    with h5py.File(partial, 'w') as f:
        dsets, n, returns = {}, 0, []
        for i in range(num_episodes):
            ep = dict(np.load(tmp_dir / f'episode_{i:05d}.npz'))
            ep['episode_index'] = np.full(len(ep['reward']), i, dtype=np.int32)
            returns.append(ep['reward'].sum())
            if not dsets:
                for k, v in ep.items():
                    dsets[k] = f.create_dataset(k, shape=(0, *v.shape[1:]), maxshape=(None, *v.shape[1:]),
                                                dtype=v.dtype, chunks=(512, *v.shape[1:]), compression='lzf')
            for k, v in ep.items():
                dsets[k].resize(n + len(v), axis=0)
                dsets[k][n:] = v
            n += len(ep['reward'])
        f.attrs['num_episodes'] = num_episodes
        f.attrs['mean_return'] = float(np.mean(returns))
        f.attrs['std_return'] = float(np.std(returns))
    partial.rename(out_path)
    return np.mean(returns), np.std(returns)


def save_preview(path, png_path, n=8):
    from PIL import Image
    with h5py.File(path, 'r') as f:
        frames = f['frames'][0:500:500 // n][:n]
    Image.fromarray(np.concatenate(list(frames), axis=1)).save(png_path)


def render(domain, setting, split, ckpt, episodes, workers, data_dir, policy_dir, overwrite=False):
    out_path = rollout_path(domain, setting, split, ckpt, data_dir)
    if out_path.exists() and not overwrite:
        print(f'exists, skipping: {out_path}')
        return
    tmp_dir = out_path.parent / f'.tmp_ckpt_{ckpt}'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    todo = [i for i in range(episodes) if not (tmp_dir / f'episode_{i:05d}.npz').exists()]  # resumable
    chunks = [c.tolist() for c in np.array_split(todo, min(workers, max(len(todo), 1))) if len(c)]
    procs = [mp.Process(target=worker, args=(w, chunk, policy_path(domain, ckpt, policy_dir), domain,
                                              SETTINGS[setting], SPLIT_SEED[split] + 100 * ckpt + w, tmp_dir))
             for w, chunk in enumerate(chunks)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
    if any(proc.exitcode != 0 for proc in procs):
        raise RuntimeError(f'a render worker failed; rerun the same command to resume ({tmp_dir})')
    mean, std = merge(tmp_dir, episodes, out_path)
    shutil.rmtree(tmp_dir)
    save_preview(out_path, out_path.with_suffix('.png'))
    print(f'wrote {out_path}: {episodes} episodes, return {mean:.1f} +- {std:.1f}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--domain', choices=list(TASK), required=True)
    p.add_argument('--setting', choices=list(SETTINGS), required=True)
    p.add_argument('--ckpts', type=int, nargs='+', default=None,
                   help='policy checkpoints (default: the pretraining set, which includes the offline RL policy)')
    p.add_argument('--splits', nargs='+', default=['train', 'val'], choices=list(SPLIT_SEED))
    p.add_argument('--train-episodes', type=int, default=500)
    p.add_argument('--val-episodes', type=int, default=50)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--data-dir', type=Path, default=DATA_DIR)
    p.add_argument('--policy-dir', type=Path, default=POLICY_DIR,
                   help='directory with <domain>/ckpt_<i>.pt (default: the paper policies in policies/)')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    mp.set_start_method('spawn', force=True)  # EGL contexts don't survive fork
    ckpts = args.ckpts or sorted(set(PRETRAIN_CKPTS[args.domain]) | {OFFLINE_CKPT[args.domain]})
    for split in args.splits:
        episodes = args.train_episodes if split == 'train' else args.val_episodes
        for ckpt in ckpts:
            render(args.domain, args.setting, split, ckpt, episodes, args.workers,
                   args.data_dir, args.policy_dir, args.overwrite)


if __name__ == '__main__':
    main()
