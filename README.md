# CardPol: action-free representations from multi-policy video

This repository reproduces the DeepMind Control experiments on **cheetah-run**
and **walker-walk** under four kinds of visual distraction. It covers two
representation-learning objectives:

- **CardPol** (`acroaf`) classifies which policy produced a pair of frame stacks
  from the same trajectory, taken up to K steps apart.
- **Policy contrastive** (`polcon`) is a supervised-contrastive baseline:
  frame stacks from the same policy are positives.

Neither objective uses actions or rewards. Both learn only from videos
recorded by several policies of different skill.

The pipeline has four stages, and each one is a single command:

```
1. cardpol.sac             train SAC policies from state (and pick snapshots of graded skill)
        │  policies/<domain>/ckpt_<i>.pt   (the paper's policies are included)
2. cardpol.render          roll the policies out; render 84x84 frames with distractions
        │  data/rollouts/<domain>/<setting>/{train,val}/ckpt_<i>.hdf5
3. cardpol.pretrain        learn an image encoder with acroaf or polcon
        │  runs/pretrain/<method>/<domain>_<setting>/seed0/encoder.pt
4. cardpol.drqbc           offline RL (DrQ+BC) on the frozen encoder; evaluate in the distracting env
           runs/pretrain/<method>/<domain>_<setting>/seed0/drqbc/seed<s>/result.json
```

## Install

You need Linux, an NVIDIA GPU and Python 3.10 or newer. MuJoCo renders
headlessly through EGL.

```bash
git clone <this repo> cardpol && cd cardpol
uv venv --python 3.11 && source .venv/bin/activate   # or: python -m venv .venv
uv pip install -e .                                  # or: pip install -e .
bash scripts/download_davis.sh                       # DAVIS 2017 backgrounds -> data/DAVIS (~830 MB)
```

Rendering uses `MUJOCO_GL=egl`, which is set automatically. If EGL isn't
available, try `MUJOCO_GL=osmesa`, which is slower.

To check the install, run every stage on a tiny budget. This takes about 10
minutes and writes to `data_smoke/` and `runs_smoke/`:

```bash
bash scripts/smoke_test.sh
```

## Distraction settings

| `--setting` | background | color | used for |
|---|---|---|---|
| `clean` | default DMC skybox | – | – |
| `static` | one fixed frame from a DAVIS video | small static shift | paper "static" |
| `easy_video` | 4 DAVIS videos, playing | drifting | paper "easy video" |
| `hard_video` | all 60 DAVIS training videos, playing | larger drift | paper "hard video" |

The camera is always static. Backgrounds come from the DAVIS 2017 training
videos in every split.

## Stage 1: low-level policies (optional)

`policies/` already holds the exact SAC policies used in the paper (actor
weights, 17 files, 5 MB). Their mean returns are:

- **cheetah** `ckpt_0`–`ckpt_8`: 4, 75, 195, 393, 542, 577, 634, 699, 804
- **walker** `ckpt_0`–`ckpt_7`: 30, 49, 409, 733, 810, 838, 885, 968

To regenerate them:

```bash
python -m cardpol.sac --domain cheetah --seed 1             # 1M env steps, snapshot every 10k (~10 h on GPU)
python -m cardpol.select_policies --domain cheetah --sac-dir runs/sac/cheetah/seed1
# -> runs/policies/cheetah/ckpt_<i>.pt : the snapshots whose return is closest to the paper's
```

To use them, pass `--policy-dir runs/policies` to `cardpol.render`. By
default, the representation dataset uses policies `{0,2,4,6,8}` (cheetah) or
`{0,2,4,6,7}` (walker). The offline RL dataset uses `ckpt_6` (cheetah) or
`ckpt_7` (walker).

## Stage 2: render data

```bash
python -m cardpol.render --domain cheetah --setting easy_video --workers 16
```

This rolls out every policy with sampled actions. It renders 500 training and
50 validation episodes per policy (501 steps each, action repeat 2), with
frames, state, action, reward and step type. Output goes to
`data/rollouts/<domain>/<setting>/<split>/ckpt_<i>.hdf5`, plus a `.png`
preview strip. Workers are separate processes, so use about one per CPU core.
Rerunning the command resumes an interrupted render.

## Stage 3: representation learning

```bash
python -m cardpol.pretrain --method acroaf --domain cheetah --setting easy_video
python -m cardpol.pretrain --method polcon --domain cheetah --setting easy_video
```

These are the paper's settings. Both methods share the same encoder: the
DrQ-v2 conv trunk plus a linear layer to 256 dimensions. Other settings are
batch size 512, Adam at lr 1e-4 with weight decay 1e-4, and random-crop plus
color-jitter augmentation. Training runs 4 epochs, or 3 for walker
clean/static/easy_video. The method-specific settings are:

- **acroaf:** a 2×512 MLP discriminator, and K=10.
- **polcon:** SupCon temperature 0.1.

The whole training set is loaded into RAM, so plan for about 40 GB of RAM.
Each run takes roughly 20–60 minutes on one A40. The logs include a detached
state-regression probe (`state_loss`), which is a cheap check of how much
state the encoder captures. See `python -m cardpol.pretrain -h` for every
option.

## Stage 4: downstream offline RL

```bash
for s in 0 1 2 3; do
  python -m cardpol.drqbc --encoder runs/pretrain/acroaf/cheetah_easy_video/seed0 --seed $s
done
```

This runs DrQ+BC (V-D4RL), keeping the encoder frozen. Settings:

- 500k steps; the agent updates every 2nd step.
- Shift augmentation.
- The offline data is the 500 rendered episodes of the offline policy, in the
  same distraction setting.
- Evaluation runs every 10k steps, over 10 episodes in the distracting
  environment.

The score is **final5**, the mean of the last 5 evaluations. Each run takes
about 2 hours on an A40. `--from-scratch --domain D --setting S` trains a
randomly initialized encoder end to end instead, as a no-pretraining
reference.

## Reproducing the table

```bash
for d in cheetah walker; do for s in clean static easy_video hard_video; do
  bash scripts/reproduce.sh $d $s        # render -> pretrain both methods -> 4 DrQ+BC seeds each
done; done
python scripts/summarize.py              # markdown table, with the paper's numbers alongside
```

The runs are independent, so you can spread them over a cluster. On Slurm,
submit each `reproduce.sh` call, or each individual command, as its own job.
Every command skips outputs that already exist.

These are the paper's DrQ+BC results: final5, mean ± std over 4 seeds.

| Method | Cheetah clean | Cheetah static | Cheetah easy video | Cheetah hard video | Walker clean | Walker static | Walker easy video | Walker hard video |
|---|---|---|---|---|---|---|---|---|
| CardPol (`acroaf`) | 523 ± 10 | 614 ± 8 | 449 ± 32 | 293 ± 21 | 885 ± 34 | 649 ± 67 | 578 ± 17 | 443 ± 10 |
| Policy contrastive (`polcon`) | 402 ± 80 | 425 ± 30 | 336 ± 51 | 196 ± 26 | 726 ± 42 | 384 ± 87 | 378 ± 19 | 354 ± 36 |

## Differences from the original research code

- **K for acroaf.** `acroaf` samples the gap between its two frame stacks
  uniformly from {1..K}, with K=10. The earlier paper encoders were trained
  with gaps in {1..7}; walker hard video used {1..10}. To match those earlier
  encoders, use `--K 7`.
- **Rendering seeds.** Seeds are assigned per worker, split and policy, so
  re-rendered data is statistically equivalent to the paper's data but not
  identical bit for bit.

## Layout

```
src/cardpol/
  common.py            settings, default checkpoints, data paths
  sac.py               stage 1: SAC (CleanRL) on state
  select_policies.py   stage 1b: pick snapshots of graded quality
  render.py            stage 2: rollouts + distraction rendering -> HDF5
  data.py, models.py   dataset/augmentation, encoder/MLP
  pretrain.py          stage 3: acroaf / polcon
  drqbc/               stage 4: DrQ+BC agent, replay buffer, entry point
  envs/dmc.py          pixel DMC wrappers (from V-D4RL)
  envs/dcs/            Distracting Control Suite wrappers (Apache-2.0)
scripts/               download_davis.sh, smoke_test.sh, reproduce.sh, summarize.py
policies/              the paper's SAC policies
AGENTS.md              instructions for coding agents running this repo
```

You can relocate the defaults with environment variables: `CARDPOL_DATA`
(default `data/`), `CARDPOL_RUNS` (default `runs/`), and `DAVIS_DIR` (default
`$CARDPOL_DATA/DAVIS/JPEGImages/480p`).

## Acknowledgements

- DrQ+BC and the pixel DMC wrappers come from
  [V-D4RL](https://github.com/conglu1997/v-d4rl) and DrQ-v2 (MIT).
- The distraction wrappers come from the
  [Distracting Control Suite](https://github.com/google-research/google-research/tree/master/distracting_control)
  (Apache-2.0).
- SAC follows [CleanRL](https://github.com/vwxyzjn/cleanrl).
