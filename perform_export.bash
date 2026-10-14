#!/usr/bin/env bash
# Export the five SAC DWCs of the paper to HDL, with the fixed-point input encoder.
#
# Usage: ./perform_export.bash MODEL_DIR [INPUT_BITS]
#   MODEL_DIR   directory holding model_sac_<Env>-v4_256_63_<seed>_6.pth
#   INPUT_BITS  signed width of the observation inputs (default: 12)
#
# Needs data/abs_max_<Env>-v4.npy (the per-dimension quantization range).
set -euo pipefail

if [[ $# -lt 1 ]]; then
  sed -n '4,6p' "$0"
  exit 1
fi
model_dir=$1
input_bits=${2:-12}

declare -A seeds=(
  [Hopper]=1000
  [HalfCheetah]=1001
  [Walker2d]=1005
  [Humanoid]=1006
  [Ant]=1002
)

for env in Hopper HalfCheetah Walker2d Humanoid Ant; do
  seed="${seeds[$env]}"
  full_env="${env}-v4"

  python export_hdl.py \
    --path="${model_dir}/model_sac_${full_env}_256_63_${seed}_6.pth" \
    --bits=63 \
    --env="${full_env}" \
    --run_episode \
    --input_quantization_bits="${input_bits}"
done
