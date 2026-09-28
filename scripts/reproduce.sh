#!/usr/bin/env bash
# Run the whole pipeline for one domain x setting, for both methods:
#   render data -> pretrain acroaf + polcon -> DrQ+BC (4 seeds each) -> table.
# Stage 1 (training SAC policies) is skipped: the paper's policies ship in policies/.
#
#   bash scripts/reproduce.sh cheetah easy_video
#   SEEDS="0 1 2 3" WORKERS=16 bash scripts/reproduce.sh walker clean
#
# Each command skips work whose output already exists, so rerunning resumes.
set -euo pipefail
DOMAIN=$1
SETTING=$2
SEEDS=${SEEDS:-"0 1 2 3"}
WORKERS=${WORKERS:-16}
RUNS=${CARDPOL_RUNS:-runs}

python -m cardpol.render --domain "$DOMAIN" --setting "$SETTING" --workers "$WORKERS"
for METHOD in acroaf polcon; do
  python -m cardpol.pretrain --method "$METHOD" --domain "$DOMAIN" --setting "$SETTING"
  for SEED in $SEEDS; do
    python -m cardpol.drqbc --encoder "$RUNS/pretrain/$METHOD/${DOMAIN}_${SETTING}/seed0" --seed "$SEED"
  done
done
python scripts/summarize.py
