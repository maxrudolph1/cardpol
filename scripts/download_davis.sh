#!/usr/bin/env bash
# Download the DAVIS 2017 (trainval, 480p) videos used as distracting backgrounds.
# Result: $CARDPOL_DATA/DAVIS/JPEGImages/480p/<video>/*.jpg  (~830 MB zip)
set -euo pipefail
DATA=${CARDPOL_DATA:-$(cd "$(dirname "$0")/.." && pwd)/data}
mkdir -p "$DATA"
cd "$DATA"
if [ -d DAVIS/JPEGImages/480p ]; then echo "already present: $DATA/DAVIS"; exit 0; fi
curl -L -o DAVIS-2017-trainval-480p.zip https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
unzip -q DAVIS-2017-trainval-480p.zip
rm DAVIS-2017-trainval-480p.zip
echo "DAVIS ready: $DATA/DAVIS/JPEGImages/480p ($(ls DAVIS/JPEGImages/480p | wc -l) videos)"
