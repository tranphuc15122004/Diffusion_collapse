"""Quick R5 decoding calibration on Dream-7B-Instruct to guarantee fluent, artifact-free generation."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_utils import format_prompt, load_frozen_dataset, load_protocol, strip_generation_text
from src.metrics.trajectory import compute_unmask_order_metrics, mask_tau_active_pre_eos
from src.model_loader import generate_dream_batch, load_model_and_tokenizer, unload_model


def main() -> None:
    proto = load_protocol()
    inf_items = load_frozen_dataset("infinity_chat100", pilot_only=True)[:2]
    # Include infchat_003 (cover letter) and infchat_009 (GoT snowball fight) or [:2]
    inf_all = load_frozen_dataset("infinity_chat100", pilot_only=True)
    items = [inf_all[1], inf_all[3]]

    model, tok = load_model_and_tokenizer(proto["models"]["primary_pair"]["dllm_instruct"], proto, is_diffusion=True)

    configs = [
        {"alg": "entropy", "temp": 0.6, "steps": 64, "mode": "unified_fewshot"},
        {"alg": "entropy", "temp": 0.6, "steps": 64, "mode": "chat_template"},
        {"alg": "topk_margin", "temp": 0.6, "steps": 64, "mode": "chat_template"},
        {"alg": "entropy", "temp": 0.4, "steps": 64, "mode": "chat_template"},
        {"alg": "entropy", "temp": 0.6, "steps": 128, "mode": "chat_template"},
    ]

    results = []
    for cfg in configs:
        prompts = []
        for it in items:
            p = format_prompt(it, proto, mode=cfg["mode"], tokenizer=tok)
            prompts.extend([p, p])  # K=2 per item -> total batch size 4

        t0 = time.perf_counter()
        out = generate_dream_batch(
            model, tok, prompts,
            max_new_tokens=128,
            steps=cfg["steps"],
            alg=cfg["alg"],
            temperature=cfg["temp"],
            top_p=0.95,
            alg_temp=0.0,
            seed=42,
        )
        wall = time.perf_counter() - t0
        gen_np = out["gen_ids"].numpy()
        tau_np = out["unmask_steps"].numpy()
        masked_tau, act_lens = mask_tau_active_pre_eos(gen_np, tau_np)

        samples = []
        for q_i, it in enumerate(items):
            s = slice(q_i * 2, (q_i + 1) * 2)
            ord_m = compute_unmask_order_metrics(masked_tau[s], min_active_tokens=4)
            texts = [strip_generation_text(t, "infinity_chat100") for t in out["texts"][s]]
            samples.append({
                "id": it["id"],
                "active_lens": act_lens[s].tolist(),
                "order": ord_m,
                "texts": [tx[:220] for tx in texts],
            })
        results.append({"config": cfg, "wall_sec": round(wall, 1), "samples": samples})
        print(f"Done {cfg} in {wall:.1f}s | lens={[x['active_lens'] for x in samples]}")

    unload_model(model)
    out_path = ROOT / "results" / "r5_decoding_calibration.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
