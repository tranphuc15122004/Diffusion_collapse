"""Resumable benchmark runner for Dream-7B / LLaDA-8B / AR baselines across frozen datasets.

Saves:
  - <out_dir>/run_meta.json         (args, model, hardware, timing, peak VRAM)
  - <out_dir>/predictions.jsonl     (one JSON record per completed question, flushed atomically)
  - <out_dir>/trajectories/<id>.npz (compressed gen_ids, unmask_steps, unmask_probs, step_mask_entropy for dLLMs)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_utils import format_prompt, load_frozen_dataset, load_protocol, strip_generation_text
from src.metrics.knowledge import match_popqa
from src.metrics.trajectory import compute_eblr, compute_leave_entity_masked_prob, mask_tau_active_pre_eos
from src.model_loader import (
    generate_ar_batch,
    generate_dream_batch,
    load_model_and_tokenizer,
    unload_model,
)


def select_items(datasets: list[str], split: str) -> list[dict]:
    items: list[dict] = []
    if split == "smoke10":
        cov = load_frozen_dataset("coverageqa", pilot_only=True)[:4]
        pop = load_frozen_dataset("popqa_stratified", pilot_only=True)
        pop_head = [r for r in pop if r["stratum"] == "head"][:2]
        pop_tail = [r for r in pop if r["stratum"] == "tail"][:2]
        inf = load_frozen_dataset("infinity_chat100", pilot_only=True)[:2]
        pool = {"coverageqa": cov, "popqa_stratified": pop_head + pop_tail, "infinity_chat100": inf}
        for ds in datasets:
            items.extend(pool[ds])
        return items

    if split == "fm4_30":
        cov = load_frozen_dataset("coverageqa", pilot_only=True)[:10]
        pop_tail = [r for r in load_frozen_dataset("popqa_stratified", pilot_only=True) if r["stratum"] == "tail"][:10]
        inf = load_frozen_dataset("infinity_chat100", pilot_only=True)[:10]
        pool = {"coverageqa": cov, "popqa_stratified": pop_tail, "infinity_chat100": inf}
        for ds in datasets:
            items.extend(pool[ds])
        return items

    pilot_only = split == "pilot"
    for ds in datasets:
        items.extend(load_frozen_dataset(ds, pilot_only=pilot_only))
    return items


def load_completed_ids(pred_path: Path) -> set[str]:
    done: set[str] = set()
    if not pred_path.exists():
        return done
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done.add(rec["id"])
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--arch", choices=["dream", "ar"], required=True)
    parser.add_argument("--datasets", default="coverageqa,popqa_stratified,infinity_chat100")
    parser.add_argument("--split", choices=["smoke10", "fm4_30", "pilot", "full"], default="pilot")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--steps", type=int, default=0, help="Denoising steps; 0 = auto match dataset max_new_tokens")
    parser.add_argument("--alg", default="entropy")
    parser.add_argument("--prompt-mode", choices=["unified_fewshot", "chat_template"], default="unified_fewshot")
    parser.add_argument("--open-prompt-mode", choices=["unified_fewshot", "chat_template"], default=None)
    parser.add_argument("--batch-size-qa", type=int, default=8)
    parser.add_argument("--batch-size-open", type=int, default=4)
    parser.add_argument("--run-lem", action="store_true", help="Run leave-entity-masked pass on PopQA outputs (Dream only)")
    parser.add_argument("--max-items", type=int, default=None, help="Optional cap on newly processed items (for resume tests)")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    proto = load_protocol()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    all_items = select_items(datasets, args.split)

    out_dir = Path(args.out_dir)
    traj_dir = out_dir / "trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"

    completed_ids = load_completed_ids(pred_path)
    pending = [it for it in all_items if it["id"] not in completed_ids]
    if args.max_items is not None:
        pending = pending[: args.max_items]

    print(
        f"[run_benchmark] model={args.model} arch={args.arch} alg={args.alg} steps={args.steps} "
        f"K={args.k} total={len(all_items)} already_done={len(completed_ids)} pending={len(pending)}"
    )

    if not pending:
        print("Nothing pending — already complete.")
        return

    is_diff = args.arch == "dream"
    temp = float(proto["decoding"]["temperature"])
    top_p = float(proto["decoding"]["top_p"])
    eps = float(proto["decoding"]["eps"])
    alg_temp = float(proto["decoding"]["alg_temp"])
    bp_tokens = proto["metrics"]["eblr_boilerplate_tokens"]
    early_frac = float(proto["metrics"]["early_phase_frac"])
    base_seed = int(proto["seed"])

    torch.cuda.reset_peak_memory_stats()
    t_load0 = time.perf_counter()
    model, tok = load_model_and_tokenizer(args.model, proto, is_diffusion=is_diff)
    load_sec = time.perf_counter() - t_load0

    t_run0 = time.perf_counter()
    processed_now = 0

    with open(pred_path, "a", encoding="utf-8") as fout:
        for idx, item in enumerate(pending):
            ds_name = item["dataset"]
            max_new = int(proto["datasets"][ds_name]["max_new_tokens"])
            cur_steps = max_new if args.steps <= 0 else args.steps
            bsz = args.batch_size_open if ds_name == "infinity_chat100" else args.batch_size_qa
            pmode = (args.open_prompt_mode or args.prompt_mode) if ds_name == "infinity_chat100" else args.prompt_mode
            prompt = format_prompt(item, proto, mode=pmode, tokenizer=tok)

            raw_texts: list[str] = []
            clean_texts: list[str] = []
            gen_ids_chunks: list[np.ndarray] = []
            ustep_chunks: list[np.ndarray] = []
            uprob_chunks: list[np.ndarray] = []
            ent_chunks: list[np.ndarray] = []
            lem_records: list[dict[str, Any]] = []

            rem = args.k
            chunk_idx = 0
            while rem > 0:
                cur_n = min(bsz, rem)
                seed = base_seed + idx * 100 + chunk_idx
                prompts_batch = [prompt] * cur_n
                if is_diff:
                    out = generate_dream_batch(
                        model,
                        tok,
                        prompts_batch,
                        max_new_tokens=max_new,
                        steps=cur_steps,
                        alg=args.alg,
                        temperature=temp,
                        top_p=top_p,
                        alg_temp=alg_temp,
                        eps=eps,
                        seed=seed,
                    )
                    gen_np = out["gen_ids"].numpy()
                    ustep_np = out["unmask_steps"].numpy()
                    uprob_np = out["unmask_probs"].numpy()
                    ent_np = out["step_mask_entropy"].numpy()
                    gen_ids_chunks.append(gen_np)
                    ustep_chunks.append(ustep_np)
                    uprob_chunks.append(uprob_np)
                    ent_chunks.append(ent_np)

                    if args.run_lem and ds_name == "popqa_stratified":
                        p_len = out["prompt_len"]
                        _, act_lens = mask_tau_active_pre_eos(gen_np, ustep_np)
                        for s_i in range(cur_n):
                            alen = int(act_lens[s_i])
                            c_txt = strip_generation_text(out["texts"][s_i], ds_name)
                            is_corr = match_popqa(c_txt, item["possible_answers"])
                            if alen > 0:
                                span_len = min(alen, 6)
                                p_unmask = float(uprob_np[s_i, :span_len].mean())
                                p_lem = compute_leave_entity_masked_prob(
                                    model,
                                    out["sequences"][s_i : s_i + 1].to(model.device),
                                    entity_span=(p_len, p_len + span_len),
                                    mask_token_id=model.config.mask_token_id,
                                    shift_logits=True,
                                )
                                min_tau = int(ustep_np[s_i, :span_len].min())
                                lem_records.append({
                                    "is_correct": is_corr,
                                    "early_unmask": min_tau <= max(1, int(0.25 * cur_steps)),
                                    "p_unmask": p_unmask,
                                    "p_lem_final": p_lem,
                                    "delta_c": p_lem - p_unmask,
                                })
                else:
                    out = generate_ar_batch(
                        model,
                        tok,
                        prompts_batch,
                        max_new_tokens=max_new,
                        temperature=temp,
                        top_p=top_p,
                        seed=seed,
                    )

                for raw_t in out["texts"]:
                    raw_texts.append(raw_t)
                    clean_texts.append(strip_generation_text(raw_t, ds_name))

                rem -= cur_n
                chunk_idx += 1

            eblr_scores: list[float] = []
            if is_diff:
                gen_all = np.concatenate(gen_ids_chunks, axis=0)
                ustep_all = np.concatenate(ustep_chunks, axis=0)
                uprob_all = np.concatenate(uprob_chunks, axis=0)
                ent_all = np.concatenate(ent_chunks, axis=0)
                for r_i in range(args.k):
                    eblr_scores.append(
                        compute_eblr(
                            gen_all[r_i].tolist(),
                            ustep_all[r_i].tolist(),
                            tok,
                            bp_tokens,
                            total_steps=cur_steps,
                            early_frac=early_frac,
                        )
                    )
                np.savez_compressed(
                    traj_dir / f"{item['id']}.npz",
                    gen_ids=gen_all.astype(np.int32),
                    unmask_steps=ustep_all.astype(np.int16),
                    unmask_probs=uprob_all.astype(np.float16),
                    step_mask_entropy=ent_all.astype(np.float16),
                )

            rec_out = {
                "id": item["id"],
                "dataset": ds_name,
                "predictions": clean_texts,
                "eblr_per_sample": eblr_scores,
                "lem_records": lem_records,
            }
            fout.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            fout.flush()
            processed_now += 1
            if processed_now % 2 == 0 or processed_now == len(pending):
                elapsed = time.perf_counter() - t_run0
                print(f"  [{processed_now}/{len(pending)}] {item['id']} ({elapsed:.1f}s elapsed)")

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    total_wall = time.perf_counter() - t_run0
    unload_model(model)

    meta_path = out_dir / "run_meta.json"
    meta = {
        "model": args.model,
        "arch": args.arch,
        "alg": args.alg if is_diff else "ar",
        "steps": args.steps if is_diff else None,
        "prompt_mode": args.prompt_mode,
        "split": args.split,
        "k": args.k,
        "processed_in_last_invocation": processed_now,
        "load_sec": round(load_sec, 2),
        "run_sec": round(total_wall, 2),
        "peak_vram_gb": round(peak_vram_gb, 3),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Finished {out_dir} | peak_vram={peak_vram_gb:.2f} GB | wall={total_wall:.1f}s")


if __name__ == "__main__":
    main()
