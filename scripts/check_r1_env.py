"""R1 Environment, Hardware, Protocol, and Tokenizer Sanity Verification."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_utils import format_prompt, load_frozen_dataset, load_protocol


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    print("=== 1. Hardware & PyTorch Check ===")
    assert torch.cuda.is_available(), "CUDA GPU is required"
    dev_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {dev_name} ({vram_gb:.2f} GB VRAM), PyTorch: {torch.__version__}")

    proto = load_protocol()
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=proto["hardware"]["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=proto["hardware"]["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=proto["hardware"]["quantization"]["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=torch.float16,
    )
    print("BitsAndBytesConfig OK:", bnb_cfg.bnb_4bit_quant_type, bnb_cfg.bnb_4bit_compute_dtype)

    print("\n=== 2. Frozen Dataset Manifest & Checksums ===")
    manifest = json.loads((ROOT / "data" / "frozen" / "manifest.json").read_text())
    for name, info in manifest["datasets"].items():
        fpath = ROOT / info["path"]
        actual_sha = sha256_file(fpath)
        assert actual_sha == info["sha256"], f"Checksum mismatch for {name}"
        print(f"  {name}: sha256 OK | pilot={info['pilot_n']}")

    print("\n=== 3. Model Configs & Tokenizers (No Weights Downloaded) ===")
    for repo in [
        proto["models"]["primary_pair"]["dllm_base"],
        proto["models"]["primary_pair"]["dllm_instruct"],
        proto["models"]["primary_pair"]["ar_base"],
        proto["models"]["primary_pair"]["ar_instruct"],
        proto["models"]["evaluator"]["embedder"],
    ]:
        trust = "Dream-org" in repo
        cfg = AutoConfig.from_pretrained(repo, trust_remote_code=trust)
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=trust)
        tok.padding_side = "left"
        print(f"  {repo}: config={type(cfg).__name__}, tok={type(tok).__name__}, pad_side={tok.padding_side}")

    print("\n=== 4. Prompt Formatting Sanity ===")
    cov_sample = load_frozen_dataset("coverageqa", pilot_only=True)[0]
    inf_sample = load_frozen_dataset("infinity_chat100", pilot_only=True)[0]
    p_cov = format_prompt(cov_sample, proto, mode="unified_fewshot")
    p_inf = format_prompt(inf_sample, proto, mode="unified_fewshot")
    assert "Question:" in p_cov and p_cov.endswith("Answer:")
    assert "Request:" in p_inf and p_inf.endswith("Response:")
    print("Prompt formatting OK. R1 environment checks PASSED.")


if __name__ == "__main__":
    main()
