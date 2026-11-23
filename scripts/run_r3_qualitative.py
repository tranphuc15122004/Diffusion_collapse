"""R3 Qualitative Trajectory & Diagnostic Inspection on Dream-7B-Instruct.

Checks:
1. Raw token IDs & unmasking order (tau) on R2's INFINITY-CHAT prompts under:
   - unified_fewshot + maskgit_plus
   - chat_template + maskgit_plus
   - chat_template + origin
   Diagnoses early '<' / '' outputs and compares full-length vs active-span (pre-EOS) H_order & rho_LTR.
2. Seed reproducibility check & entity extraction check on 3 CoverageQA + 1 PopQA questions (K=4).
3. Throughput & VRAM check for batch_size=8 (max_new_tokens=64) vs batch_size=4.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_utils import format_prompt, load_frozen_dataset, load_protocol, strip_generation_text
from src.metrics.knowledge import clean_qa_prediction, compute_coverageqa_metrics, compute_popqa_metrics
from src.metrics.trajectory import compute_leave_entity_masked_prob, compute_unmask_order_metrics
from src.model_loader import generate_dream_batch, load_model_and_tokenizer, unload_model

STOP_IDS = {151643, 151645}  # <|endoftext|>, <|im_end|>


def active_span_tau(gen_ids: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Zero out tau positions at/after the first EOS/IM_END token per row."""
    masked = tau.copy()
    for r in range(gen_ids.shape[0]):
        eos_pos = [i for i, tid in enumerate(gen_ids[r].tolist()) if tid in STOP_IDS]
        if eos_pos:
            cut = eos_pos[0]
            masked[r, cut:] = 0
    return masked


def summarize_open_batch(out: dict, tok, items: list[dict], k_per_q: int, label: str) -> dict:
    gen_ids = out["gen_ids"].numpy()
    tau = out["unmask_steps"].numpy()
    tau_active = active_span_tau(gen_ids, tau)

    # Split into per-question groups (prompts repeated K times)
    per_q = []
    for q_idx, item in enumerate(items):
        s = slice(q_idx * k_per_q, (q_idx + 1) * k_per_q)
        g_sub = gen_ids[s]
        t_sub = tau[s]
        ta_sub = tau_active[s]
        full_ord = compute_unmask_order_metrics(t_sub)
        act_ord = compute_unmask_order_metrics(ta_sub)

        samples = []
        for r_idx in range(k_per_q):
            ids_list = g_sub[r_idx].tolist()
            eos_pos = [i for i, tid in enumerate(ids_list) if tid in STOP_IDS]
            first_eos = eos_pos[0] if eos_pos else len(ids_list)
            raw_toks = [tok.decode([tid], skip_special_tokens=False) for tid in ids_list[:24]]
            raw_text = tok.decode(ids_list, skip_special_tokens=False)
            clean_text = strip_generation_text(raw_text, item["dataset"])
            samples.append({
                "first_eos_pos": first_eos,
                "first_24_tokens": raw_toks,
                "first_24_tau": t_sub[r_idx, :24].tolist(),
                "last_12_tau": t_sub[r_idx, -12:].tolist(),
                "clean_text_preview": clean_text[:200],
            })

        per_q.append({
            "id": item["id"],
            "query": item["query"],
            "order_full_128": full_ord,
            "order_active_pre_eos": act_ord,
            "samples": samples,
        })
    return {"label": label, "questions": per_q}


def main() -> None:
    proto = load_protocol()
    model_id = proto["models"]["primary_pair"]["dllm_instruct"]
    model, tok = load_model_and_tokenizer(model_id, proto, is_diffusion=True)

    steps = 64
    temp = float(proto["decoding"]["temperature"])
    top_p = float(proto["decoding"]["top_p"])

    # --- Part 1: Diagnose R2's open-ended queries (indices 1 and 2 from R2 inf_128 which gave '<' and 'The', plus index 0) ---
    inf_all = load_frozen_dataset("infinity_chat100", pilot_only=True)
    # In R2 we took inf_items[:4]; index 1 gave '<', index 3 gave ''
    inf_diag = [inf_all[1], inf_all[3]]
    k_open = 4

    open_runs = []
    for mode, alg in [
        ("unified_fewshot", "maskgit_plus"),
        ("chat_template", "maskgit_plus"),
        ("chat_template", "origin"),
    ]:
        prompts = []
        for item in inf_diag:
            p = format_prompt(item, proto, mode=mode, tokenizer=tok)
            prompts.extend([p] * k_open)

        # Run in 2 batches of 4 (max_new_tokens=128)
        all_gen, all_tau, all_ent = [], [], []
        t0 = time.perf_counter()
        for b_start in range(0, len(prompts), 4):
            batch_prompts = prompts[b_start : b_start + 4]
            out = generate_dream_batch(
                model, tok, batch_prompts,
                max_new_tokens=128, steps=steps, alg=alg,
                temperature=temp, top_p=top_p, seed=42 + b_start,
            )
            all_gen.append(out["gen_ids"])
            all_tau.append(out["unmask_steps"])
            all_ent.append(out["step_mask_entropy"])
        wall = time.perf_counter() - t0
        merged = {
            "gen_ids": torch.cat(all_gen, dim=0),
            "unmask_steps": torch.cat(all_tau, dim=0),
            "step_mask_entropy": torch.cat(all_ent, dim=0),
        }
        res = summarize_open_batch(merged, tok, inf_diag, k_open, f"{mode}__{alg}")
        res["wall_sec"] = round(wall, 2)
        open_runs.append(res)
        print(f"[Open {mode} / {alg}] done in {wall:.1f}s")

    # --- Part 2: Seed reproducibility & batch_size=8 throughput/VRAM test on QA ---
    cov_items = load_frozen_dataset("coverageqa", pilot_only=True)[:3]
    pop_items = load_frozen_dataset("popqa_stratified", pilot_only=True)[:1]
    qa_items = cov_items + pop_items  # 4 questions x K=4 = 16 samples

    qa_prompts = []
    for item in qa_items:
        p = format_prompt(item, proto, mode="unified_fewshot")
        qa_prompts.extend([p] * 4)

    # Test seed reproducibility on 1 batch of 4
    r_a = generate_dream_batch(model, tok, qa_prompts[:4], max_new_tokens=64, steps=steps, alg="maskgit_plus", temperature=temp, top_p=top_p, seed=123)
    r_b = generate_dream_batch(model, tok, qa_prompts[:4], max_new_tokens=64, steps=steps, alg="maskgit_plus", temperature=temp, top_p=top_p, seed=123)
    reproducible = torch.equal(r_a["gen_ids"], r_b["gen_ids"])
    print("Seed reproducibility exact match:", reproducible)

    # Test batch_size=8 on QA (16 samples = 2 batches of 8) for maskgit_plus and origin
    qa_summaries = {}
    for alg in ("maskgit_plus", "origin"):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        gen_list, tau_list, text_list = [], [], []
        bsz_test = 8
        for b_start in range(0, len(qa_prompts), bsz_test):
            out = generate_dream_batch(
                model, tok, qa_prompts[b_start : b_start + bsz_test],
                max_new_tokens=64, steps=steps, alg=alg,
                temperature=temp, top_p=top_p, seed=42 + b_start,
            )
            gen_list.append(out["gen_ids"])
            tau_list.append(out["unmask_steps"])
            text_list.extend(out["texts"])
        wall = time.perf_counter() - t0
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        gen_all = torch.cat(gen_list, dim=0).numpy()
        tau_all = torch.cat(tau_list, dim=0).numpy()
        tau_act = active_span_tau(gen_all, tau_all)

        q_evals = []
        for q_idx, item in enumerate(qa_items):
            s = slice(q_idx * 4, (q_idx + 1) * 4)
            raw_texts = text_list[s]
            cleaned = [clean_qa_prediction(strip_generation_text(t, item["dataset"])) for t in raw_texts]
            ord_act = compute_unmask_order_metrics(tau_act[s])
            if item["dataset"] == "coverageqa":
                m = compute_coverageqa_metrics(cleaned, item["answers"], category=item["category"])
            else:
                m = compute_popqa_metrics(cleaned, item["possible_answers"])
            q_evals.append({
                "id": item["id"],
                "question": item["question"],
                "preds": cleaned,
                "metrics": m,
                "order_active": ord_act,
            })

        qa_summaries[alg] = {
            "batch_size": bsz_test,
            "wall_sec": round(wall, 2),
            "sec_per_sample": round(wall / len(qa_prompts), 2),
            "peak_vram_gb": round(peak_gb, 3),
            "questions": q_evals,
        }
        print(f"[QA bsz=8 {alg}] wall={wall:.1f}s ({wall/len(qa_prompts):.2f}s/sample), peak_vram={peak_gb:.2f} GB")

    # Test leave-entity-masked on PopQA question sample 0
    pop_out = generate_dream_batch(
        model, tok, [format_prompt(pop_items[0], proto, mode="unified_fewshot")],
        max_new_tokens=64, steps=steps, alg="maskgit_plus", temperature=temp, top_p=top_p, seed=42,
    )
    p_len = pop_out["prompt_len"]
    seq = pop_out["sequences"][:1].to(model.device)
    g_ids = pop_out["gen_ids"][0].tolist()
    eos_pos = [i for i, tid in enumerate(g_ids) if tid in STOP_IDS]
    ent_end = min(eos_pos[0], 4) if eos_pos else 4
    p_final = compute_leave_entity_masked_prob(model, seq, (p_len, p_len + max(1, ent_end)), model.config.mask_token_id)
    p_unmask = float(pop_out["unmask_probs"][0, :max(1, ent_end)].mean().item())

    unload_model(model)

    report = {
        "seed_reproducible": reproducible,
        "open_ended_diagnosis": open_runs,
        "qa_batch8_comparison": qa_summaries,
        "popqa_lem_check": {
            "id": pop_items[0]["id"],
            "question": pop_items[0]["question"],
            "pred": strip_generation_text(pop_out["texts"][0], "popqa_stratified"),
            "p_at_unmask": round(p_unmask, 4),
            "p_leave_entity_masked_T": round(p_final, 4),
            "delta_c": round(p_final - p_unmask, 4),
        },
    }
    out_path = ROOT / "results" / "r3_qualitative_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
