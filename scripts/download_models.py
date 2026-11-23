"""Download all missing models for Pilot R5 and Full R6 into HuggingFace cache."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_utils import load_protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tier",
        choices=["r5", "all"],
        default="all",
        help="'r5' downloads only primary twin Base models; 'all' also downloads R6 cross-check models",
    )
    args = parser.parse_args()

    proto = load_protocol()
    p_models = proto["models"]["primary_pair"]
    c_models = proto["models"]["cross_check_pair"]

    targets = [
        p_models["dllm_base"],
        p_models["ar_base"],
    ]
    if args.tier == "all":
        targets.extend([
            c_models["dllm_base"],
            c_models["dllm_instruct"],
            c_models["ar_base"],
            c_models["ar_instruct"],
        ])

    for repo in targets:
        t0 = time.perf_counter()
        print(f"=== Downloading {repo} ===", flush=True)
        local_dir = snapshot_download(
            repo_id=repo,
            ignore_patterns=["original/*", "*.pth", "*.bin.index.json"]
            if "llama" in repo.lower()
            else None,
            max_workers=4,
        )
        elapsed = time.perf_counter() - t0
        print(f"=== Done {repo} -> {local_dir} ({elapsed:.1f}s) ===", flush=True)


if __name__ == "__main__":
    main()
