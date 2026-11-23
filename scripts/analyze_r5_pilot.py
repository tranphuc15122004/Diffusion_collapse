"""Statistical Decision Analyzer for R5 Pilot Study across FM-1, FM-2, FM-3, and FM-4."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_utils import load_frozen_dataset
from src.metrics.knowledge import compute_coverageqa_metrics, compute_popqa_metrics, compute_thr
from src.metrics.stats import bootstrap_ci, paired_permutation_test
from src.metrics.trajectory import compute_unmask_order_metrics, mask_tau_active_pre_eos


def load_preds(run_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    p = run_dir / "predictions.jsonl"
    if not p.exists():
        return out
    for ln in p.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            rec = json.loads(ln)
            out[rec["id"]] = rec
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r5-dir", required=True)
    parser.add_argument("--eval-json", required=True)
    args = parser.parse_args()

    r5_dir = Path(args.r5_dir)
    eval_data = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
    runs_by_name = {r["run_name"]: r for r in eval_data["runs"]}

    ds_index: dict[str, dict] = {}
    for name in ("coverageqa", "popqa_stratified", "infinity_chat100"):
        for rec in load_frozen_dataset(name, pilot_only=True):
            ds_index[rec["id"]] = rec

    db_preds = load_preds(r5_dir / "dream_base_entropy")
    di_preds = load_preds(r5_dir / "dream_inst_entropy")
    qb_preds = load_preds(r5_dir / "qwen_base_ar")
    qi_preds = load_preds(r5_dir / "qwen_inst_ar")

    # --- FM-1: Paired CoverageQA & PopQA + THR + EBLR ---
    cov_ids = sorted(qid for qid, gt in ds_index.items() if gt["dataset"] == "coverageqa")
    db_rec, di_rec, qb_rec, qi_rec = [], [], [], []
    db_prec, di_prec, db_eblr, di_eblr = [], [], [], []
    for qid in cov_ids:
        gt = ds_index[qid]
        m_db = compute_coverageqa_metrics(db_preds[qid]["predictions"], gt["answers"], gt.get("category"))
        m_di = compute_coverageqa_metrics(di_preds[qid]["predictions"], gt["answers"], gt.get("category"))
        m_qb = compute_coverageqa_metrics(qb_preds[qid]["predictions"], gt["answers"], gt.get("category"))
        m_qi = compute_coverageqa_metrics(qi_preds[qid]["predictions"], gt["answers"], gt.get("category"))
        db_rec.append(m_db["recall"])
        di_rec.append(m_di["recall"])
        qb_rec.append(m_qb["recall"])
        qi_rec.append(m_qi["recall"])
        db_prec.append(m_db["precision"])
        di_prec.append(m_di["precision"])
        db_eblr.append(float(np.mean(db_preds[qid]["eblr_per_sample"])))
        di_eblr.append(float(np.mean(di_preds[qid]["eblr_per_sample"])))

    d_rec_dream = paired_permutation_test(db_rec, di_rec)
    d_rec_qwen = paired_permutation_test(qb_rec, qi_rec)
    d_eblr_dream = paired_permutation_test(db_eblr, di_eblr)
    delta_eblr_items = np.array(di_eblr) - np.array(db_eblr)
    delta_rec_items = np.array(di_rec) - np.array(db_rec)
    eblr_rec_corr = spearmanr(delta_eblr_items, delta_rec_items)

    # PopQA THR
    pop_head_ids = sorted(qid for qid, gt in ds_index.items() if gt["dataset"] == "popqa_stratified" and gt["stratum"] == "head")
    pop_tail_ids = sorted(qid for qid, gt in ds_index.items() if gt["dataset"] == "popqa_stratified" and gt["stratum"] == "tail")

    def pop_acc(preds_map: dict, ids: list[str]) -> list[float]:
        return [compute_popqa_metrics(preds_map[q]["predictions"], ds_index[q]["possible_answers"])["acc"] for q in ids]

    thr_dream = compute_thr(
        float(np.mean(pop_acc(di_preds, pop_tail_ids))),
        float(np.mean(pop_acc(db_preds, pop_tail_ids))),
        float(np.mean(pop_acc(di_preds, pop_head_ids))),
        float(np.mean(pop_acc(db_preds, pop_head_ids))),
    )
    thr_qwen = compute_thr(
        float(np.mean(pop_acc(qi_preds, pop_tail_ids))),
        float(np.mean(pop_acc(qb_preds, pop_tail_ids))),
        float(np.mean(pop_acc(qi_preds, pop_head_ids))),
        float(np.mean(pop_acc(qb_preds, pop_head_ids))),
    )

    fm1 = {
        "dream_covqa_recall_shift": d_rec_dream,
        "qwen_covqa_recall_shift": d_rec_qwen,
        "dream_eblr_shift": d_eblr_dream,
        "eblr_vs_recall_spearman": {
            "rho": float(eblr_rec_corr.statistic) if np.isfinite(eblr_rec_corr.statistic) else 0.0,
            "p_value": float(eblr_rec_corr.pvalue) if np.isfinite(eblr_rec_corr.pvalue) else 1.0,
        },
        "dream_thr": thr_dream,
        "qwen_thr": thr_qwen,
    }

    # --- FM-2: Generation Order & Diversity Collapse on INFINITY-CHAT100 ---
    inf_ids = sorted(qid for qid, gt in ds_index.items() if gt["dataset"] == "infinity_chat100")
    db_h, di_h, db_rho, di_rho = [], [], [], []
    for qid in inf_ids:
        npz_b = np.load(r5_dir / "dream_base_entropy" / "trajectories" / f"{qid}.npz")
        npz_i = np.load(r5_dir / "dream_inst_entropy" / "trajectories" / f"{qid}.npz")
        mb, _ = mask_tau_active_pre_eos(npz_b["gen_ids"], npz_b["unmask_steps"])
        mi, _ = mask_tau_active_pre_eos(npz_i["gen_ids"], npz_i["unmask_steps"])
        ob = compute_unmask_order_metrics(mb, min_active_tokens=4)
        oi = compute_unmask_order_metrics(mi, min_active_tokens=4)
        db_h.append(ob["h_order_bits"])
        di_h.append(oi["h_order_bits"])
        db_rho.append(ob["rho_ltr"])
        di_rho.append(oi["rho_ltr"])

    fm2 = {
        "dream_h_order_shift": paired_permutation_test(db_h, di_h),
        "dream_rho_ltr_shift": paired_permutation_test(db_rho, di_rho),
        "dream_base_vs05": runs_by_name["dream_base_entropy"]["infinity_chat"]["vs_0.5"],
        "dream_inst_vs05": runs_by_name["dream_inst_entropy"]["infinity_chat"]["vs_0.5"],
        "qwen_base_vs05": runs_by_name["qwen_base_ar"]["infinity_chat"]["vs_0.5"],
        "qwen_inst_vs05": runs_by_name["qwen_inst_ar"]["infinity_chat"]["vs_0.5"],
        "dream_inst_clusters": runs_by_name["dream_inst_entropy"]["infinity_chat"]["distinct_idea_clusters"],
        "qwen_inst_clusters": runs_by_name["qwen_inst_ar"]["infinity_chat"]["distinct_idea_clusters"],
        "cross_model_similarity": eval_data.get("cross_model_similarity", {}),
    }

    # --- FM-3: Step Compression & LEM Delta C_wrong ---
    fm3_runs = {}
    for label, rname in [("T64", "dream_inst_entropy"), ("T32", "dream_inst_popqa_s32"), ("T16", "dream_inst_popqa_s16")]:
        pop_sec = runs_by_name[rname]["popqa"]
        fm3_runs[label] = {
            "hallucination_rate": pop_sec["hallucination_rate"],
            "head_acc": pop_sec["head_acc"],
            "tail_acc": pop_sec["tail_acc"],
            "delta_c_wrong": pop_sec["delta_c_wrong"],
            "delta_c_right": pop_sec["delta_c_right"],
        }

    # --- FM-4: Static Remasking Tradeoff ---
    fm4_runs = {}
    for alg in ("entropy", "origin", "maskgit_plus", "topk_margin"):
        rname = f"fm4_dream_inst_{alg}"
        r = runs_by_name[rname]
        fm4_runs[alg] = {
            "covqa_recall": r["coverageqa"]["recall"],
            "covqa_precision": r["coverageqa"]["precision"],
            "inf_vs05": r["infinity_chat"]["vs_0.5"],
            "inf_clusters": r["infinity_chat"]["distinct_idea_clusters"],
            "inf_ppl": r["infinity_chat"]["external_ppl"],
            "inf_active_tokens": r["infinity_chat"]["mean_active_tokens"],
        }

    verdict = {"fm1": fm1, "fm2": fm2, "fm3": fm3_runs, "fm4": fm4_runs}
    verdict_path = r5_dir / "r5_fm_verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8")

    # Write human-readable decision memo
    memo = [
        "# R5 Pilot Study — Empirical Failure Mode Decision Memo\n",
        "## FM-1: Early Structural Lock-in & Tail Knowledge Erasure",
        f"- Dream CoverageQA Recall@8 shift (Base -> Inst): {d_rec_dream['diff_mean']:+.4f} (p={d_rec_dream['p_value']:.4f})",
        f"- Qwen CoverageQA Recall@8 shift (Base -> Inst): {d_rec_qwen['diff_mean']:+.4f} (p={d_rec_qwen['p_value']:.4f})",
        f"- Dream EBLR shift (Base -> Inst): {d_eblr_dream['diff_mean']:+.4f} (p={d_eblr_dream['p_value']:.4f})",
        f"- Spearman(ΔEBLR, ΔRecall@8): rho={fm1['eblr_vs_recall_spearman']['rho']:+.4f} (p={fm1['eblr_vs_recall_spearman']['p_value']:.4f})",
        f"- PopQA THR: Dream={thr_dream:.3f} vs Qwen={thr_qwen:.3f}\n",
        "## FM-2: Generation Order Collapse (INFINITY-CHAT100)",
        f"- Dream H_order shift (Base -> Inst): {fm2['dream_h_order_shift']['diff_mean']:+.4f} bits (p={fm2['dream_h_order_shift']['p_value']:.4f})",
        f"- Dream rho_LTR shift (Base -> Inst): {fm2['dream_rho_ltr_shift']['diff_mean']:+.4f} (p={fm2['dream_rho_ltr_shift']['p_value']:.4f})",
        f"- Vendi Score (q=0.5): Dream Base={fm2['dream_base_vs05']['mean']:.3f} -> Inst={fm2['dream_inst_vs05']['mean']:.3f} | Qwen Base={fm2['qwen_base_vs05']['mean']:.3f} -> Inst={fm2['qwen_inst_vs05']['mean']:.3f}\n",
        "## FM-3: Bidirectional Hallucination Cascade (PopQA Step Compression)",
    ]
    for lbl in ("T64", "T32", "T16"):
        x = fm3_runs[lbl]
        memo.append(
            f"- {lbl}: Halluc={x['hallucination_rate']['mean']:.3f} | "
            f"ΔC_wrong={x['delta_c_wrong']['mean']:+.4f} (n={x['delta_c_wrong']['n']}) | "
            f"ΔC_right={x['delta_c_right']['mean']:+.4f} (n={x['delta_c_right']['n']})"
        )
    memo.append("\n## FM-4: Static Remasking Pareto Tradeoff")
    for alg, x in fm4_runs.items():
        memo.append(
            f"- {alg}: Recall={x['covqa_recall']['mean']:.3f}, Prec={x['covqa_precision']['mean']:.3f}, "
            f"VS_0.5={x['inf_vs05']['mean']:.3f}, Clusters={x['inf_clusters']['mean']:.2f}, "
            f"ActiveLen={x['inf_active_tokens']['mean']:.1f}, PPL={x['inf_ppl']['mean']:.2f}"
        )

    memo_path = r5_dir / "R5_DECISION_MEMO.md"
    memo_path.write_text("\n".join(memo) + "\n", encoding="utf-8")
    print(f"Saved {verdict_path} and {memo_path}")


if __name__ == "__main__":
    main()
