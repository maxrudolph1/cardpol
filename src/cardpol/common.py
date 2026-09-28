"""Constants shared by every stage of the pipeline, and the on-disk layout."""
import os
from pathlib import Path

DOMAINS = ['cheetah', 'walker']
TASK = {'cheetah': 'run', 'walker': 'walk'}
STATE_DIM = {'cheetah': 17, 'walker': 24}
ACTION_DIM = {'cheetah': 6, 'walker': 6}
# Proprioceptive state fed to the SAC policies, in the order gymnasium's
# FlattenObservation (sorted keys) used when they were trained.
STATE_KEYS = {'cheetah': ['position', 'velocity'], 'walker': ['height', 'orientations', 'velocity']}

# setting -> distracting_mode of cardpol.envs.dmc.make
#   clean       no distraction
#   static      one fixed DAVIS frame as background + small static color shift
#   easy_video  4 DAVIS training videos, playing, + drifting color
#   hard_video  all 60 DAVIS training videos, playing, + larger drifting color
SETTINGS = {'clean': 'clean', 'static': 'fixed_easy', 'easy_video': 'easy', 'hard_video': 'hard'}

# SAC checkpoints used in the paper, indexed by training progress. The mean
# return of each (stochastic) policy, over 500 rendered episodes:
PAPER_POLICY_RETURNS = {
    'cheetah': [4.5, 75, 195, 393, 542, 577, 634, 699, 804],
    'walker': [30, 49, 409, 733, 810, 838, 885, 968],
}
# Checkpoints whose rollouts form the representation-learning dataset.
PRETRAIN_CKPTS = {'cheetah': [0, 2, 4, 6, 8], 'walker': [0, 2, 4, 6, 7]}
# Checkpoint whose rollouts (with rewards) are the downstream offline RL dataset.
OFFLINE_CKPT = {'cheetah': 6, 'walker': 7}

FRAME_STACK = 3
ACTION_REPEAT = 2
IMG_SIZE = 84

ROOT = Path(os.environ.get('CARDPOL_ROOT', Path(__file__).resolve().parents[2]))
DATA_DIR = Path(os.environ.get('CARDPOL_DATA', ROOT / 'data'))
RUNS_DIR = Path(os.environ.get('CARDPOL_RUNS', ROOT / 'runs'))
DAVIS_DIR = Path(os.environ.get('DAVIS_DIR', DATA_DIR / 'DAVIS' / 'JPEGImages' / '480p'))
POLICY_DIR = ROOT / 'policies'


def policy_path(domain, ckpt, policy_dir=POLICY_DIR):
    return Path(policy_dir) / domain / f'ckpt_{ckpt}.pt'


def rollout_path(domain, setting, split, ckpt, data_dir=DATA_DIR):
    """Merged rollouts of one policy: data/rollouts/<domain>/<setting>/<split>/ckpt_<i>.hdf5"""
    return Path(data_dir) / 'rollouts' / domain / setting / split / f'ckpt_{ckpt}.hdf5'
