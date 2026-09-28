#!/usr/bin/env bash
# Tiny end-to-end run of every stage (~10 min on one GPU) to check an install.
# Writes to data_smoke/ and runs_smoke/, never touching data/ or runs/.
set -euo pipefail
export CARDPOL_DATA=${CARDPOL_DATA:-data_smoke} CARDPOL_RUNS=${CARDPOL_RUNS:-runs_smoke}
export DAVIS_DIR=${DAVIS_DIR:-data/DAVIS/JPEGImages/480p}
DOMAIN=${1:-cheetah}
SETTING=${2:-easy_video}
OFFLINE=$([ "$DOMAIN" = walker ] && echo 7 || echo 6)  # the offline RL policy must be rendered

python -m cardpol.sac --domain "$DOMAIN" --total-timesteps 3000 --learning-starts 1000 --save-every 1000
python -m cardpol.select_policies --domain "$DOMAIN" --sac-dir "$CARDPOL_RUNS/sac/$DOMAIN/seed1" \
  --targets 0 10 --episodes 1
python -m cardpol.render --domain "$DOMAIN" --setting "$SETTING" --ckpts 0 "$OFFLINE" \
  --train-episodes 4 --val-episodes 2 --workers 2
for METHOD in cardpol polcon; do
  python -m cardpol.pretrain --method "$METHOD" --domain "$DOMAIN" --setting "$SETTING" --ckpts 0 "$OFFLINE" \
    --batch-size 64 --max-steps 30 --val-every 10 --val-batches 2 --num-workers 2
  python -m cardpol.drqbc --encoder "$CARDPOL_RUNS/pretrain/$METHOD/${DOMAIN}_${SETTING}/seed0" \
    --steps 2000 --eval-every 1000 --eval-episodes 1
done
python scripts/summarize.py --runs "$CARDPOL_RUNS"
echo "smoke test passed"
