#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/home/tuantb/diffusion_collapse"
SMOKE_ROOT="${WORKDIR}/results/r4_smoke"
rm -rf "${SMOKE_ROOT}"
mkdir -p "${SMOKE_ROOT}"

echo "=== R4 Step 1: Dream-7B-Instruct (alg=entropy, steps=64, K=4) - Part 1 (5 items) ==="
python3 -u "${WORKDIR}/scripts/run_benchmark.py" \
  --model Dream-org/Dream-v0-Instruct-7B \
  --arch dream \
  --split smoke10 \
  --k 4 \
  --steps 64 \
  --alg entropy \
  --run-lem \
  --max-items 5 \
  --out-dir "${SMOKE_ROOT}/dream_inst_entropy"

COUNT_PART1=$(wc -l < "${SMOKE_ROOT}/dream_inst_entropy/predictions.jsonl")
echo "Checkpoint count after Part 1: ${COUNT_PART1}"
test "${COUNT_PART1}" -eq 5

echo "=== R4 Step 2: Resume Dream-7B-Instruct (alg=entropy) -> remaining 5 items ==="
python3 -u "${WORKDIR}/scripts/run_benchmark.py" \
  --model Dream-org/Dream-v0-Instruct-7B \
  --arch dream \
  --split smoke10 \
  --k 4 \
  --steps 64 \
  --alg entropy \
  --run-lem \
  --out-dir "${SMOKE_ROOT}/dream_inst_entropy"

COUNT_PART2=$(wc -l < "${SMOKE_ROOT}/dream_inst_entropy/predictions.jsonl")
echo "Checkpoint count after Resume: ${COUNT_PART2}"
test "${COUNT_PART2}" -eq 10

echo "=== R4 Step 3: Dream-7B-Instruct (alg=maskgit_plus, steps=64, K=4) ==="
python3 -u "${WORKDIR}/scripts/run_benchmark.py" \
  --model Dream-org/Dream-v0-Instruct-7B \
  --arch dream \
  --split smoke10 \
  --k 4 \
  --steps 64 \
  --alg maskgit_plus \
  --out-dir "${SMOKE_ROOT}/dream_inst_maskgit"

echo "=== R4 Step 4: Dream-7B-Instruct (alg=entropy, steps=32, PopQA LEM check) ==="
python3 -u "${WORKDIR}/scripts/run_benchmark.py" \
  --model Dream-org/Dream-v0-Instruct-7B \
  --arch dream \
  --datasets popqa_stratified \
  --split smoke10 \
  --k 4 \
  --steps 32 \
  --alg entropy \
  --run-lem \
  --out-dir "${SMOKE_ROOT}/dream_inst_entropy_s32"

echo "=== R4 Step 5: Qwen2.5-7B-Instruct AR baseline (smoke10, K=4) ==="
python3 -u "${WORKDIR}/scripts/run_benchmark.py" \
  --model Qwen/Qwen2.5-7B-Instruct \
  --arch ar \
  --split smoke10 \
  --k 4 \
  --out-dir "${SMOKE_ROOT}/qwen_inst_ar"

echo "=== R4 Step 6: 3-Tier Evaluation (BGE-M3 + Bootstrap + Plots) ==="
python3 -u "${WORKDIR}/scripts/evaluate_run.py" \
  --run-dirs \
    "${SMOKE_ROOT}/dream_inst_entropy" \
    "${SMOKE_ROOT}/dream_inst_maskgit" \
    "${SMOKE_ROOT}/dream_inst_entropy_s32" \
    "${SMOKE_ROOT}/qwen_inst_ar" \
  --out-json "${SMOKE_ROOT}/r4_eval_report.json" \
  --plot-dir "${SMOKE_ROOT}/plots"

echo "=== R4 SMOKE RUN COMPLETED SUCCESSFULLY ==="
