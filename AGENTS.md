# Running this repo: guide for coding agents

This is the operational manual. For the science and the full description, see
README.md. Every stage is a `python -m cardpol.<stage>` command. Each command
takes `--help`, writes to a fixed path, and skips work whose output already
exists (`--overwrite` forces a rerun). If you rerun an interrupted command, it
resumes; it never duplicates work.

## 1. Set up and verify

```bash
uv venv --python 3.11 && source .venv/bin/activate && uv pip install -e .
# If `python -c "import torch; print(torch.cuda.is_available())"` prints False, the default torch
# wheel is built for a newer CUDA than the driver supports: install a matching wheel, e.g.
#   uv pip install torch --index-url https://download.pytorch.org/whl/cu124
bash scripts/download_davis.sh          # or: export DAVIS_DIR=/existing/DAVIS/JPEGImages/480p
bash scripts/smoke_test.sh              # ~10 min; must end with "smoke test passed"
```

Don't start the full pipeline until the smoke test passes. It exercises every
stage on tiny budgets, in `data_smoke/` and `runs_smoke/`, which you can
delete afterwards.

## 2. The pipeline

Here `D` is `cheetah` or `walker`, `S` is `clean`, `static`, `easy_video` or
`hard_video`, and `M` is `cardpol` or `polcon`.

| step | command | output | cost (1 A40) |
|---|---|---|---|
| policies | already in `policies/` (skip unless asked to retrain) | `policies/D/ckpt_i.pt` | – |
| retrain policies | `python -m cardpol.sac --domain D` then `python -m cardpol.select_policies --domain D --sac-dir runs/sac/D/seed1` | `runs/policies/D/ckpt_i.pt` | ~10 h, 1 GPU |
| render | `python -m cardpol.render --domain D --setting S --workers <ncpu>` | `data/rollouts/D/S/{train,val}/ckpt_i.hdf5` + `.png` | ~0.5–2 h on 16 CPUs |
| pretrain | `python -m cardpol.pretrain --method M --domain D --setting S` | `runs/pretrain/M/D_S/seed0/encoder.pt` | 20–60 min, ~40 GB RAM |
| downstream | `python -m cardpol.drqbc --encoder runs/pretrain/M/D_S/seed0 --seed {0,1,2,3}` | `.../seed0/drqbc/seed<s>/result.json` | ~2 h per seed |
| table | `python scripts/summarize.py` | markdown table on stdout | seconds |

`bash scripts/reproduce.sh D S` chains render, pretrain for both methods, and
4 DrQ+BC seeds for one cell. The full table has 16 cells: 2 domains × 4
settings × 2 methods. Each cell needs one pretrain run and 4 DrQ+BC runs, and
the 8 renders are shared between the two methods. The cells are independent,
so on a cluster submit one job per command, respecting the order
render → pretrain → drqbc.

## 3. Checking results

- **Render:** open the `.png` next to each `.hdf5` and check the frames show
  the intended distraction. The command prints the mean return per policy. It
  should be close to the returns in `common.PAPER_POLICY_RETURNS`, for example
  cheetah `ckpt_8` ≈ 804.
- **Pretrain:** `val_metrics.csv` has `loss`, `state_loss` and, for CARDPol,
  `accuracy`. The loss should fall. For CARDPol, `accuracy` should end well
  above chance (1/5).
- **DrQ+BC:** `eval.csv` has one row per 10k steps. The score is
  `result.json:final5`. `summarize.py` puts it next to the paper's numbers.
  Expect the ranking CARDPol > polcon; single seeds are noisy (std up to about
  80).

## 4. Gotchas

- **Headless rendering.** `MUJOCO_GL=egl` is set automatically. `render.py`
  uses `spawn` workers because EGL contexts don't survive `fork`. If EGL is
  missing, use `MUJOCO_GL=osmesa`.
- **DAVIS path.** Any non-`clean` setting raises `FileNotFoundError` naming
  the missing DAVIS path. Fix it with `download_davis.sh` or `DAVIS_DIR`.
- **Pretraining memory.** Pretraining loads every training rollout of 5
  policies into RAM (about 30 GB of uint8 frames). Its DataLoader workers
  share that memory. If the process is OOM-killed, lower `--num-workers`; do
  not lower the dataset size, because that changes the experiment.
- **DrQ+BC memory.** DrQ+BC loads one policy's rollouts (about 5 GB) into its
  replay buffer.
- **Locations.** Outputs default to `data/` and `runs/` inside the repo. Point
  `CARDPOL_DATA`, `CARDPOL_RUNS` or `DAVIS_DIR` elsewhere for big disks.
  Every stage honours them.
- **Paper hyperparameters.** Keep the defaults when reproducing the paper.
  They are the paper's settings (see README, "Differences from the original
  research code"). Change them only when asked, and record any change you
  make.
- **Wandb.** It is optional (`--wandb`, after `uv pip install -e .[wandb]`).
  CSV logs are always written.
