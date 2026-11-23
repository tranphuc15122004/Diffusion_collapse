#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/home/tuantb/diffusion_collapse"
R5_DIR="${WORKDIR}/results/r5_pilot"
mkdir -p "${R5_DIR}"

echo "=== R5 Gate 0: Quick Base-model 2-item sanity check (Dream-Base & Qwen-Base) ==="
python3 -u "${WORKDIR}/scripts/run_benchmark.py" \
  --model Dream-org/Dream-v0-Base-7B \
  --arch dream \
  --datasets coverageqa,infinity_chat100 \
  --split smoke10 \
  --k 2 \
  --steps 0 \
  --alg entropy \
  --prompt-mode unified_fewshot \
  --max-items 2 \
  --out-dir "${R5_DIR}/_sanity_dream_base"

echo "=== R5 Run 1A: Dream-7B-Base (pilot 100q, K=8, steps=auto [64 QA / 128 Open], alg=entropy) ==="
python3 -u "${WORKDIR}/scripts/run_benchmark.py" \
  --model Dream-org/Dream-v0-Base-7B \
  --arch dream \
  --split pilot \
  --k 8 \
  --steps 0 \
  --alg entropy \
  --prompt-mode unified_fewshot \
  --out-dir "${R5_DIR}/dream_base_entropy"

echo "=== R5 Run 1B: Dream-7B-Instruct (pilot 100q, K=8, steps=auto, alg=entropy, +LEM on PopQA) ==="
python3 -u "${WORKDIR}/scripts/run_benchmark.py" \
  --model Dream-org/Dream-v0-Instruct-7B \
  --arch dream \
  --split pilot \
  --k 8 \
  --steps 0 \
  --alg entropy \
  --prompt-mode unified_fewshot \
  --open-prompt-mode chat_template \
  --run-lem \
  --out-dir "${R5_DIR}/dream_inst_entropy"

echo "=== R5 Run 1C: Qwen2.5-7B-Base (pilot 100q, K=8) ==="
python3 -u "${WORKDIR}/scripts/run_benchmark.py" \
  --model Qwen/Qwen2.5-7B \
  --arch ar \
  --split pilot \
  --k 8 \
  --prompt-mode unified_fewshot \
  --out-dir "${R5_DIR}/qwen_base_ar"

echo "=== R5 Run 1D: Qwen2.5-7B-Instruct (pilot 100q, K=8) ==="
python3 -u "${WORKDIR}/scripts/run_benchmark.py" \
  --model Qwen/Qwen2.5-7B-Instruct \
  --arch ar \
  --split pilot \
  --k 8 \
  --prompt-mode unified_fewshot \
  --open-prompt-mode chat_template \
  --out-dir "${R5_DIR}/qwen_inst_ar"

echo "=== R5 Run 2A: FM-3 Step Compression on PopQA (Dream-Instruct, steps=32, K=8, +LEM) ==="
python3 -u "${WORKDIR}/scripts/run_benchmark.py" \
  --model Dream-org/Dream-v0-Instruct-7B \
  --arch dream \
  --datasets popqa_stratified \
  --split pilot \
  --k 8 \
  --steps 32 \
  --alg entropy \
  --prompt-mode unified_fewshot \
  --run-lem \
  --out-dir "${R5_DIR}/dream_inst_popqa_s32"

echo "=== R5 Run 2B: FM-3 Step Compression on PopQA (Dream-Instruct, steps=16, K=8, +LEM) ==="
python3 -u "${WORKDIR}/scripts/run_benchmark.py" \
  --model Dream-org/Dream-v0-Instruct-7B \
  --arch dream \
  --datasets popqa_stratified \
  --split pilot \
  --k 8 \
  --steps 16 \
  --alg entropy \
  --prompt-mode unified_fewshot \
  --run-lem \
  --out-dir "${R5_DIR}/dream_inst_popqa_s16"

echo "=== R5 Run 3: FM-4 Static Remasking Strategies on fm4_30 (Dream-Instruct, K=8, steps=auto) ==="
python3 - << 'PY'
import json
from pathlib import Path
import sys
sys.path.insert(0, "/home/tuantb/diffusion_collapse")
from scripts.run_benchmark import select_items
fm4_ids = {it["id"] for it in select_items(["coverageqa", "popqa_stratified", "infinity_chat100"], "fm4_30")}
src_dir = Path("/home/tuantb/diffusion_collapse/results/r5_pilot/dream_inst_entropy")
dst_dir = Path("/home/tuantb/diffusion_collapse/results/r5_pilot/fm4_dream_inst_entropy")
dst_dir.mkdir(parents=True, exist_ok=True)
(dst_dir / "trajectories").mkdir(exist_ok=True)
lines = [ln for ln in (src_dir / "predictions.jsonl").read_text().splitlines() if ln.strip() and json.loads(ln)["id"] in fm4_ids]
(dst_dir / "predictions.jsonl").write_text("\n".join(lines) + "\n")
for qid in fm4_ids:
    src_npz = src_dir / "trajectories" / f"{qid}.npz"
    dst_npz = dst_dir / "trajectories" / f"{qid}.npz"
    if src_npz.exists() and not dst_npz.exists():
        dst_npz.symlink_to(src_npz)
PY

for ALG in origin maskgit_plus topk_margin; do
  python3 -u "${WORKDIR}/scripts/run_benchmark.py" \
    --model Dream-org/Dream-v0-Instruct-7B \
    --arch dream \
    --split fm4_30 \
    --k 8 \
    --steps 0 \
    --alg "${ALG}" \
    --prompt-mode unified_fewshot \
    --open-prompt-mode chat_template \
    --out-dir "${R5_DIR}/fm4_dream_inst_${ALG}"
done

echo "=== R5 Evaluation (3-Tier Metrics + Claim Clustering + Cross-Model Sim + External PPL + Plots) ==="
python3 -u "${WORKDIR}/scripts/evaluate_run.py" \
  --run-dirs \
    "${R5_DIR}/dream_base_entropy" \
    "${R5_DIR}/dream_inst_entropy" \
    "${R5_DIR}/qwen_base_ar" \
    "${R5_DIR}/qwen_inst_ar" \
    "${R5_DIR}/dream_inst_popqa_s32" \
    "${R5_DIR}/dream_inst_popqa_s16" \
    "${R5_DIR}/fm4_dream_inst_entropy" \
    "${R5_DIR}/fm4_dream_inst_origin" \
    "${R5_DIR}/fm4_dream_inst_maskgit_plus" \
    "${R5_DIR}/fm4_dream_inst_topk_margin" \
  --compute-ppl \
  --out-json "${R5_DIR}/r5_eval_report.json" \
  --plot-dir "${R5_DIR}/plots"

echo "=== R5 Statistical Decision Analysis & Memo ==="
python3 -u "${WORKDIR}/scripts/analyze_r5_pilot.py" \
  --r5-dir "${R5_DIR}" \
  --eval-json "${R5_DIR}/r5_eval_report.json"

echo "=== R5 PILOT STUDY COMPLETED SUCCESSFULLY ==="
