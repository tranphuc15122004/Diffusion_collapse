"""R2 Gate Verification: 1-batch real 7B 4-bit NF4 + fp16 + sdpa sanity & VRAM/throughput check."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_utils import format_prompt, load_frozen_dataset, load_protocol, strip_generation_text
from src.metrics.trajectory import compute_leave_entity_masked_prob, compute_unmask_order_metrics
from src.model_loader import generate_ar_batch, generate_dream_batch, load_model_and_tokenizer, unload_model


def check_dream(proto: dict) -> dict:
    model_id = proto["models"]["primary_pair"]["dllm_instruct"]
    vram_gate = float(proto["hardware"]["max_vram_gb_gate"])
    bsz = int(proto["hardware"]["batch_size"])
    steps = int(proto["decoding"]["steps_default"])
    alg = proto["decoding"]["alg_default"]
    temp = float(proto["decoding"]["temperature"])
    top_p = float(proto["decoding"]["top_p"])

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model, tok = load_model_and_tokenizer(model_id, proto, is_diffusion=True)
    load_sec = time.perf_counter() - t0
    load_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"[Dream-7B-Instruct] loaded in {load_sec:.1f}s, alloc={load_vram_gb:.2f} GB")

    cov_items = load_frozen_dataset("coverageqa", pilot_only=True)[:2]
    pop_items = load_frozen_dataset("popqa_stratified", pilot_only=True)[:2]
    qa_items = cov_items + pop_items
    qa_prompts = [format_prompt(r, proto, mode="unified_fewshot") for r in qa_items]

    inf_items = load_frozen_dataset("infinity_chat100", pilot_only=True)[:bsz]
    inf_prompts = [format_prompt(r, proto, mode="unified_fewshot") for r in inf_items]

    # Run 1: QA batch (max_new_tokens=64)
    torch.cuda.reset_peak_memory_stats()
    t1 = time.perf_counter()
    qa_out = generate_dream_batch(
        model, tok, qa_prompts,
        max_new_tokens=64, steps=steps, alg=alg, temperature=temp, top_p=top_p, seed=42,
    )
    qa_sec = time.perf_counter() - t1
    qa_peak_gb = torch.cuda.max_memory_allocated() / (1024**3)

    ent_64 = qa_out["step_mask_entropy"]
    prob_64 = qa_out["unmask_probs"]
    ustep_64 = qa_out["unmask_steps"]
    assert torch.all(torch.isfinite(ent_64)).item(), "NaN/Inf in step_mask_entropy (64)"
    assert torch.all(torch.isfinite(prob_64)).item(), "NaN/Inf in unmask_probs (64)"
    assert torch.all((ustep_64 >= 1) & (ustep_64 <= steps)).item(), "Invalid unmask_steps (64)"

    # Verify leave-entity-masked forward pass on sample 0 (span of first 3 generated tokens)
    p_len = qa_out["prompt_len"]
    p_lem = compute_leave_entity_masked_prob(
        model,
        qa_out["sequences"][:1].to(model.device),
        entity_span=(p_len, p_len + 3),
        mask_token_id=model.config.mask_token_id,
        shift_logits=True,
    )
    assert 0.0 < p_lem <= 1.0, f"Invalid leave-entity-masked prob: {p_lem}"

    # Run 2: Open-ended worst-case length batch (max_new_tokens=128)
    torch.cuda.reset_peak_memory_stats()
    t2 = time.perf_counter()
    inf_out = generate_dream_batch(
        model, tok, inf_prompts,
        max_new_tokens=128, steps=steps, alg=alg, temperature=temp, top_p=top_p, seed=42,
    )
    inf_sec = time.perf_counter() - t2
    inf_peak_gb = torch.cuda.max_memory_allocated() / (1024**3)

    ent_128 = inf_out["step_mask_entropy"]
    prob_128 = inf_out["unmask_probs"]
    assert torch.all(torch.isfinite(ent_128)).item(), "NaN/Inf in step_mask_entropy (128)"
    assert torch.all(torch.isfinite(prob_128)).item(), "NaN/Inf in unmask_probs (128)"

    order_stats_qa = compute_unmask_order_metrics(ustep_64.numpy())
    order_stats_inf = compute_unmask_order_metrics(inf_out["unmask_steps"].numpy())

    qa_clean = [strip_generation_text(t, r["dataset"]) for t, r in zip(qa_out["texts"], qa_items)]
    inf_clean = [strip_generation_text(t, r["dataset"]) for t, r in zip(inf_out["texts"], inf_items)]

    peak_overall = max(load_vram_gb, qa_peak_gb, inf_peak_gb)
    assert peak_overall <= vram_gate, f"VRAM gate failed: {peak_overall:.2f} > {vram_gate}"

    unload_model(model)
    return {
        "model_id": model_id,
        "load_sec": round(load_sec, 2),
        "load_vram_gb": round(load_vram_gb, 3),
        "qa_64": {
            "batch_size": bsz,
            "steps": steps,
            "max_new_tokens": 64,
            "wall_sec": round(qa_sec, 2),
            "sec_per_sample": round(qa_sec / bsz, 2),
            "peak_vram_gb": round(qa_peak_gb, 3),
            "leave_entity_masked_prob_ex0": round(p_lem, 6),
            "order_stats": order_stats_qa,
            "sample_outputs": qa_clean,
        },
        "inf_128": {
            "batch_size": bsz,
            "steps": steps,
            "max_new_tokens": 128,
            "wall_sec": round(inf_sec, 2),
            "sec_per_sample": round(inf_sec / bsz, 2),
            "peak_vram_gb": round(inf_peak_gb, 3),
            "order_stats": order_stats_inf,
            "sample_outputs": [s[:200] for s in inf_clean],
        },
        "gate_passed": True,
    }


def check_ar(proto: dict) -> dict:
    model_id = proto["models"]["primary_pair"]["ar_instruct"]
    vram_gate = float(proto["hardware"]["max_vram_gb_gate"])
    bsz = int(proto["hardware"]["batch_size"])
    temp = float(proto["decoding"]["temperature"])
    top_p = float(proto["decoding"]["top_p"])

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model, tok = load_model_and_tokenizer(model_id, proto, is_diffusion=False)
    load_sec = time.perf_counter() - t0
    load_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"[Qwen2.5-7B-Instruct] loaded in {load_sec:.1f}s, alloc={load_vram_gb:.2f} GB")

    cov_items = load_frozen_dataset("coverageqa", pilot_only=True)[:2]
    pop_items = load_frozen_dataset("popqa_stratified", pilot_only=True)[:2]
    qa_items = cov_items + pop_items
    qa_prompts = [format_prompt(r, proto, mode="unified_fewshot") for r in qa_items]

    inf_items = load_frozen_dataset("infinity_chat100", pilot_only=True)[:bsz]
    inf_prompts = [format_prompt(r, proto, mode="unified_fewshot") for r in inf_items]

    torch.cuda.reset_peak_memory_stats()
    t1 = time.perf_counter()
    qa_out = generate_ar_batch(model, tok, qa_prompts, max_new_tokens=64, temperature=temp, top_p=top_p, seed=42)
    qa_sec = time.perf_counter() - t1
    qa_peak_gb = torch.cuda.max_memory_allocated() / (1024**3)

    torch.cuda.reset_peak_memory_stats()
    t2 = time.perf_counter()
    inf_out = generate_ar_batch(model, tok, inf_prompts, max_new_tokens=128, temperature=temp, top_p=top_p, seed=42)
    inf_sec = time.perf_counter() - t2
    inf_peak_gb = torch.cuda.max_memory_allocated() / (1024**3)

    qa_clean = [strip_generation_text(t, r["dataset"]) for t, r in zip(qa_out["texts"], qa_items)]
    inf_clean = [strip_generation_text(t, r["dataset"]) for t, r in zip(inf_out["texts"], inf_items)]

    peak_overall = max(load_vram_gb, qa_peak_gb, inf_peak_gb)
    assert peak_overall <= vram_gate, f"VRAM gate failed: {peak_overall:.2f} > {vram_gate}"

    unload_model(model)
    return {
        "model_id": model_id,
        "load_sec": round(load_sec, 2),
        "load_vram_gb": round(load_vram_gb, 3),
        "qa_64": {
            "batch_size": bsz,
            "max_new_tokens": 64,
            "wall_sec": round(qa_sec, 2),
            "sec_per_sample": round(qa_sec / bsz, 2),
            "peak_vram_gb": round(qa_peak_gb, 3),
            "sample_outputs": qa_clean,
        },
        "inf_128": {
            "batch_size": bsz,
            "max_new_tokens": 128,
            "wall_sec": round(inf_sec, 2),
            "sec_per_sample": round(inf_sec / bsz, 2),
            "peak_vram_gb": round(inf_peak_gb, 3),
            "sample_outputs": [s[:200] for s in inf_clean],
        },
        "gate_passed": True,
    }


def main() -> None:
    proto = load_protocol()
    dream_res = check_dream(proto)
    ar_res = check_ar(proto)

    # Estimate R5 Pilot GPU budget:
    # Run 1 (Base+Inst Dream + Base+Inst AR on 100q x K=8): ~ (2*100*8 dLLM + 2*100*8 AR)
    # Run 2 (Dream-Inst T in {16, 32} on 40q x K=8; T=64 reused from Run 1)
    # Run 3 (Dream-Inst 3 extra algs on 30q x K=8; maskgit_plus reused from Run 1)
    avg_dllm_sec = (0.7 * dream_res["qa_64"]["sec_per_sample"] + 0.3 * dream_res["inf_128"]["sec_per_sample"])
    avg_ar_sec = (0.7 * ar_res["qa_64"]["sec_per_sample"] + 0.3 * ar_res["inf_128"]["sec_per_sample"])
    est_r5_hours = (
        (2 * 100 * 8 * avg_dllm_sec)
        + (2 * 100 * 8 * avg_ar_sec)
        + (40 * 8 * avg_dllm_sec * (0.25 + 0.5))
        + (3 * 30 * 8 * avg_dllm_sec)
    ) / 3600.0

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "vram_gate_gb": proto["hardware"]["max_vram_gb_gate"],
        "dream_instruct": dream_res,
        "qwen_instruct": ar_res,
        "estimated_r5_pilot_gpu_hours": round(est_r5_hours, 2),
    }
    out_path = ROOT / "results" / "r2_sanity_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
