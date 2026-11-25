"""Generate detailed R5 Pilot statistics and 5 publication-grade diagnostic figures."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_utils import load_frozen_dataset
from src.metrics.diversity import BGEEmbedder, compute_mtld, distinct_idea_clusters, mean_pairwise_similarity, split_into_claims, vendi_score
from src.metrics.knowledge import compute_coverageqa_metrics, compute_popqa_metrics
from src.metrics.stats import bootstrap_ci, paired_permutation_test
from src.metrics.trajectory import mask_tau_active_pre_eos

R5_DIR = ROOT / "results" / "r5_pilot"
FIG_DIR = R5_DIR / "figures"
DOCS_FIG_DIR = ROOT / "docs" / "figures"


def load_preds(run_name: str) -> dict[str, dict]:
    out = {}
    for ln in (R5_DIR / run_name / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            out[r["id"]] = r
    return out


def compute_detailed_stats(eval_data: dict) -> dict:
    ds_index = {}
    for name in ("coverageqa", "popqa_stratified", "infinity_chat100"):
        for rec in load_frozen_dataset(name, pilot_only=True):
            ds_index[rec["id"]] = rec

    db = load_preds("dream_base_entropy")
    di = load_preds("dream_inst_entropy")
    qb = load_preds("qwen_base_ar")
    qi = load_preds("qwen_inst_ar")

    # Paired tests on CoverageQA
    cov_ids = sorted(q for q, v in ds_index.items() if v["dataset"] == "coverageqa")
    cov_metrics = {k: {"rec": [], "prec": []} for k in ("db", "di", "qb", "qi")}
    for qid in cov_ids:
        gt = ds_index[qid]
        for key, pmap in [("db", db), ("di", di), ("qb", qb), ("qi", qi)]:
            m = compute_coverageqa_metrics(pmap[qid]["predictions"], gt["answers"], gt.get("category"))
            cov_metrics[key]["rec"].append(m["recall"])
            cov_metrics[key]["prec"].append(m["precision"])

    # Paired tests on PopQA Head & Tail
    head_ids = sorted(q for q, v in ds_index.items() if v["dataset"] == "popqa_stratified" and v["stratum"] == "head")
    tail_ids = sorted(q for q, v in ds_index.items() if v["dataset"] == "popqa_stratified" and v["stratum"] == "tail")
    pop_acc = {}
    for key, pmap in [("db", db), ("di", di), ("qb", qb), ("qi", qi)]:
        pop_acc[key] = {
            "head": [compute_popqa_metrics(pmap[q]["predictions"], ds_index[q]["possible_answers"])["acc"] for q in head_ids],
            "tail": [compute_popqa_metrics(pmap[q]["predictions"], ds_index[q]["possible_answers"])["acc"] for q in tail_ids],
        }

    # Paired tests on INFINITY-CHAT100 (using BGE-M3)
    inf_ids = sorted(q for q, v in ds_index.items() if v["dataset"] == "infinity_chat100")
    embedder = BGEEmbedder("BAAI/bge-m3")
    inf_per_item = {k: {"vs05": [], "pw": [], "mtld": []} for k in ("db", "di", "qb", "qi")}
    for qid in inf_ids:
        for key, pmap in [("db", db), ("di", di), ("qb", qb), ("qi", qi)]:
            preds = pmap[qid]["predictions"]
            embs = embedder.encode(preds, batch_size=8, max_length=256)
            inf_per_item[key]["vs05"].append(vendi_score(embs, q=0.5))
            inf_per_item[key]["pw"].append(mean_pairwise_similarity(embs))
            inf_per_item[key]["mtld"].append(float(np.mean([compute_mtld(p) for p in preds])))

    # FM-4 paired comparison on fm4_30 (entropy vs origin & maskgit_plus)
    fm4_runs = {
        alg: load_preds(f"fm4_dream_inst_{alg}")
        for alg in ("entropy", "origin", "maskgit_plus", "topk_margin")
    }
    fm4_cov_ids = sorted(q for q in fm4_runs["entropy"] if q.startswith("covqa_"))
    fm4_inf_ids = sorted(q for q in fm4_runs["entropy"] if q.startswith("infchat_"))
    fm4_cov_rec = {
        alg: [compute_coverageqa_metrics(fm4_runs[alg][q]["predictions"], ds_index[q]["answers"], ds_index[q].get("category"))["recall"] for q in fm4_cov_ids]
        for alg in fm4_runs
    }
    fm4_inf_vs05 = {
        alg: [vendi_score(embedder.encode(fm4_runs[alg][q]["predictions"], batch_size=8, max_length=256), q=0.5) for q in fm4_inf_ids]
        for alg in fm4_runs
    }
    embedder.unload()

    # FM-3 early vs late unmask breakdown of Delta C_wrong
    lem_breakdown = {}
    for lbl, rname in [("T64", "dream_inst_entropy"), ("T32", "dream_inst_popqa_s32"), ("T16", "dream_inst_popqa_s16")]:
        pmap = load_preds(rname)
        early_w, late_w, early_r, late_r = [], [], [], []
        for qid in head_ids + tail_ids:
            for rec in pmap[qid].get("lem_records", []):
                if rec["is_correct"]:
                    (early_r if rec["early_unmask"] else late_r).append(rec["delta_c"])
                else:
                    (early_w if rec["early_unmask"] else late_w).append(rec["delta_c"])
        lem_breakdown[lbl] = {
            "early_wrong": bootstrap_ci(early_w),
            "late_wrong": bootstrap_ci(late_w),
            "early_right": bootstrap_ci(early_r),
            "late_right": bootstrap_ci(late_r),
        }

    return {
        "paired_tests": {
            "dream_covqa_recall": paired_permutation_test(cov_metrics["db"]["rec"], cov_metrics["di"]["rec"]),
            "qwen_covqa_recall": paired_permutation_test(cov_metrics["qb"]["rec"], cov_metrics["qi"]["rec"]),
            "dream_covqa_prec": paired_permutation_test(cov_metrics["db"]["prec"], cov_metrics["di"]["prec"]),
            "qwen_covqa_prec": paired_permutation_test(cov_metrics["qb"]["prec"], cov_metrics["qi"]["prec"]),
            "dream_popqa_head": paired_permutation_test(pop_acc["db"]["head"], pop_acc["di"]["head"]),
            "dream_popqa_tail": paired_permutation_test(pop_acc["db"]["tail"], pop_acc["di"]["tail"]),
            "qwen_popqa_head": paired_permutation_test(pop_acc["qb"]["head"], pop_acc["qi"]["head"]),
            "qwen_popqa_tail": paired_permutation_test(pop_acc["qb"]["tail"], pop_acc["qi"]["tail"]),
            "dream_inf_vs05": paired_permutation_test(inf_per_item["db"]["vs05"], inf_per_item["di"]["vs05"]),
            "qwen_inf_vs05": paired_permutation_test(inf_per_item["qb"]["vs05"], inf_per_item["qi"]["vs05"]),
            "inst_dream_vs_qwen_vs05": paired_permutation_test(inf_per_item["qi"]["vs05"], inf_per_item["di"]["vs05"]),
            "dream_inf_pw_sim": paired_permutation_test(inf_per_item["db"]["pw"], inf_per_item["di"]["pw"]),
            "qwen_inf_pw_sim": paired_permutation_test(inf_per_item["qb"]["pw"], inf_per_item["qi"]["pw"]),
            "fm3_t64_vs_t16_tail": paired_permutation_test(pop_acc["di"]["tail"], [
                compute_popqa_metrics(load_preds("dream_inst_popqa_s16")[q]["predictions"], ds_index[q]["possible_answers"])["acc"]
                for q in tail_ids
            ]),
            "fm4_origin_vs_entropy_covqa_recall": paired_permutation_test(fm4_cov_rec["entropy"], fm4_cov_rec["origin"]),
            "fm4_entropy_vs_maskgit_inf_vs05": paired_permutation_test(fm4_inf_vs05["maskgit_plus"], fm4_inf_vs05["entropy"]),
            "fm4_entropy_vs_topk_inf_vs05": paired_permutation_test(fm4_inf_vs05["topk_margin"], fm4_inf_vs05["entropy"]),
        },
        "lem_breakdown": lem_breakdown,
    }


def make_figures(eval_data: dict, detailed: dict) -> list[str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    runs = {r["run_name"]: r for r in eval_data["runs"]}
    paths = []

    # --- Figure 1: Twin Comparison (Dream-7B vs Qwen2.5-7B Base->Instruct) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    models = ["Dream-Base", "Dream-Inst", "Qwen-Base", "Qwen-Inst"]
    rkeys = ["dream_base_entropy", "dream_inst_entropy", "qwen_base_ar", "qwen_inst_ar"]
    colors = ["#4C78A8", "#1F4E78", "#F58518", "#B84A00"]

    # Panel A: Knowledge Retention (CoverageQA Recall@8 & PopQA Tail Acc)
    x = np.arange(4)
    cov_rec = [runs[k]["coverageqa"]["recall"]["mean"] for k in rkeys]
    cov_err = [
        [runs[k]["coverageqa"]["recall"]["mean"] - runs[k]["coverageqa"]["recall"]["ci_low"] for k in rkeys],
        [runs[k]["coverageqa"]["recall"]["ci_high"] - runs[k]["coverageqa"]["recall"]["mean"] for k in rkeys],
    ]
    pop_tail = [runs[k]["popqa"]["tail_acc"]["mean"] for k in rkeys]
    axes[0].bar(x - 0.18, cov_rec, width=0.35, yerr=cov_err, capsize=3, color=colors, alpha=0.95, label="CovQA Recall@8")
    axes[0].bar(x + 0.18, pop_tail, width=0.35, color=colors, alpha=0.45, hatch="//", edgecolor="black", label="PopQA Tail Acc")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=15, ha="right")
    axes[0].set_ylabel("Score")
    axes[0].set_title("(a) Knowledge Coverage & Tail Retention")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.25)

    # Panel B: Meaning vs Form (VS_0.5 vs MTLD)
    vs05 = [runs[k]["infinity_chat"]["vs_0.5"]["mean"] for k in rkeys]
    vs05_err = [
        [runs[k]["infinity_chat"]["vs_0.5"]["mean"] - runs[k]["infinity_chat"]["vs_0.5"]["ci_low"] for k in rkeys],
        [runs[k]["infinity_chat"]["vs_0.5"]["ci_high"] - runs[k]["infinity_chat"]["vs_0.5"]["mean"] for k in rkeys],
    ]
    mtld = [runs[k]["infinity_chat"]["mtld"]["mean"] for k in rkeys]
    ax_b2 = axes[1].twinx()
    ax_b2.spines["top"].set_visible(False)
    b1 = axes[1].bar(x - 0.18, vs05, width=0.35, yerr=vs05_err, capsize=3, color=colors, label="Vendi VS_0.5 (left)")
    b2 = ax_b2.bar(x + 0.18, mtld, width=0.35, color="#54A24B", alpha=0.65, label="Lexical MTLD (right)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=15, ha="right")
    axes[1].set_ylabel("Vendi Score (q=0.5)")
    ax_b2.set_ylabel("Lexical MTLD")
    axes[1].set_title("(b) Meaning (VS_0.5) vs Form (MTLD)")
    axes[1].legend([b1, b2], ["Vendi VS_0.5 (left)", "Lexical MTLD (right)"], fontsize=8, loc="upper left")
    axes[1].grid(True, axis="y", alpha=0.25)

    # Panel C: Cross-Model Hivemind Matrix
    cm = eval_data["cross_model_similarity"]
    mat = np.array([[cm[a][b]["mean"] for b in rkeys] for a in rkeys])
    im = axes[2].imshow(mat, cmap="YlOrRd", vmin=0.68, vmax=0.88)
    axes[2].set_xticks(range(4))
    axes[2].set_yticks(range(4))
    axes[2].set_xticklabels(models, rotation=20, ha="right", fontsize=8)
    axes[2].set_yticklabels(models, fontsize=8)
    for i in range(4):
        for j in range(4):
            axes[2].text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color="black" if mat[i, j] < 0.83 else "white", fontsize=8.5, fontweight="bold")
    axes[2].set_title("(c) Cosine Similarity Matrix (InfChat)")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    p1 = FIG_DIR / "fig1_twin_collapse_overview.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    paths.append(p1)

    # --- Figure 2: FM-4 Static Remasking Pareto Frontier ---
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    algs = ["origin", "entropy", "maskgit_plus", "topk_margin"]
    alg_labels = ["origin (random)", "entropy", "maskgit_plus", "topk_margin"]
    alg_colors = ["#2CA02C", "#1F77B4", "#D62728", "#FF7F0E"]

    # Panel A: Fluency (External PPL, log) vs Semantic Diversity (VS_0.5)
    for alg, lbl, col in zip(algs, alg_labels, alg_colors):
        r = runs[f"fm4_dream_inst_{alg}"]["infinity_chat"]
        ppl = r["external_ppl"]["mean"]
        vs = r["vs_0.5"]["mean"]
        cl = r["distinct_idea_clusters"]["mean"]
        axes[0].scatter(ppl, vs, s=cl * 18 + 60, color=col, alpha=0.85, edgecolors="black", linewidth=1.2, label=f"{lbl} (clusters={cl:.1f})")
        offset = (-12, 6) if ppl > 200 else (10, 6)
        ha = "right" if ppl > 200 else "left"
        axes[0].annotate(lbl, (ppl, vs), xytext=offset, textcoords="offset points", ha=ha, fontsize=8.5, fontweight="bold", color=col)
    axes[0].set_xscale("log")
    axes[0].set_xlim(10, 1200)
    axes[0].set_xlabel("External Perplexity (Qwen2.5-7B-Inst, log scale, lower=better)")
    axes[0].set_ylabel("Vendi Score (q=0.5, higher=more diverse)")
    axes[0].set_title("(a) Fluency–Diversity Pareto Frontier (fm4_30)")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8, loc="lower left")

    # Panel B: CovQA Recall@8 & Distinct Idea Clusters & ActiveLen normalized
    rec_vals = [runs[f"fm4_dream_inst_{a}"]["coverageqa"]["recall"]["mean"] for a in algs]
    cl_vals = [runs[f"fm4_dream_inst_{a}"]["infinity_chat"]["distinct_idea_clusters"]["mean"] for a in algs]
    x4 = np.arange(4)
    ax2_r = axes[1].twinx()
    ax2_r.spines["top"].set_visible(False)
    b_rec = axes[1].bar(x4 - 0.18, rec_vals, width=0.35, color=alg_colors, alpha=0.9, label="CovQA Recall@8 (left)")
    b_cl = ax2_r.bar(x4 + 0.18, cl_vals, width=0.35, color=alg_colors, alpha=0.45, hatch="//", edgecolor="black", label="Distinct Idea Clusters (right)")
    axes[1].set_xticks(x4)
    axes[1].set_xticklabels(alg_labels, rotation=15, ha="right", fontsize=8.5)
    axes[1].set_ylabel("CoverageQA Recall@8")
    ax2_r.set_ylabel("Distinct Idea Clusters (InfChat)")
    axes[1].set_title("(b) Factual Coverage vs Idea Cluster Collapse")
    axes[1].legend([b_rec[0], b_cl[0]], ["CovQA Recall@8 (left)", "Distinct Idea Clusters (right)"], fontsize=8, loc="upper right")
    axes[1].grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    p2 = FIG_DIR / "fig2_fm4_pareto_frontier.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    paths.append(p2)

    # --- Figure 3: FM-3 Step Compression & Bidirectional Confidence Drift ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    steps_lbls = ["T=64", "T=32", "T=16"]
    fm3_keys = ["dream_inst_entropy", "dream_inst_popqa_s32", "dream_inst_popqa_s16"]
    head_accs = [runs[k]["popqa"]["head_acc"]["mean"] for k in fm3_keys]
    tail_accs = [runs[k]["popqa"]["tail_acc"]["mean"] for k in fm3_keys]
    hall_rates = [runs[k]["popqa"]["hallucination_rate"]["mean"] for k in fm3_keys]

    x3 = np.arange(3)
    axes[0].plot(x3, head_accs, marker="o", linewidth=2.2, color="#1F77B4", label="Head Acc (top-20% s_pop)")
    axes[0].plot(x3, tail_accs, marker="s", linewidth=2.2, color="#D62728", label="Tail Acc (bottom-20% s_pop)")
    axes[0].plot(x3, hall_rates, marker="^", linestyle="--", linewidth=1.8, color="#7F7F7F", label="Hallucination Rate")
    for xi, (h, t) in enumerate(zip(head_accs, tail_accs)):
        axes[0].text(xi, h + 0.025, f"{h:.3f}", ha="center", fontsize=8.5, color="#1F77B4")
        axes[0].text(xi, t - 0.055, f"{t:.3f}", ha="center", fontsize=8.5, color="#D62728", fontweight="bold")
    axes[0].set_xticks(x3)
    axes[0].set_xticklabels(steps_lbls)
    axes[0].set_ylim(0, 0.85)
    axes[0].set_ylabel("Proportion")
    axes[0].set_title("(a) Asymmetric Tail Collapse Under Step Compression")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25)

    dc_w = [runs[k]["popqa"]["delta_c_wrong"]["mean"] for k in fm3_keys]
    dc_w_err = [
        [runs[k]["popqa"]["delta_c_wrong"]["mean"] - runs[k]["popqa"]["delta_c_wrong"]["ci_low"] for k in fm3_keys],
        [runs[k]["popqa"]["delta_c_wrong"]["ci_high"] - runs[k]["popqa"]["delta_c_wrong"]["mean"] for k in fm3_keys],
    ]
    dc_r = [runs[k]["popqa"]["delta_c_right"]["mean"] for k in fm3_keys]
    dc_r_err = [
        [runs[k]["popqa"]["delta_c_right"]["mean"] - runs[k]["popqa"]["delta_c_right"]["ci_low"] for k in fm3_keys],
        [runs[k]["popqa"]["delta_c_right"]["ci_high"] - runs[k]["popqa"]["delta_c_right"]["mean"] for k in fm3_keys],
    ]
    axes[1].axhline(0.0, color="black", linestyle=":", linewidth=1.2)
    axes[1].bar(x3 - 0.16, dc_w, width=0.32, yerr=dc_w_err, capsize=3, color="#D62728", alpha=0.85, label="Wrong Entities (ΔC_wrong)")
    axes[1].bar(x3 + 0.16, dc_r, width=0.32, yerr=dc_r_err, capsize=3, color="#2CA02C", alpha=0.85, label="Correct Entities (ΔC_right)")
    axes[1].set_xticks(x3)
    axes[1].set_xticklabels(steps_lbls)
    axes[1].set_ylabel("Confidence Drift (P_LEM^(T) - P^(t_unmask))")
    axes[1].set_title("(b) Leave-Entity-Masked Confidence Drift (95% CI)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    p3 = FIG_DIR / "fig3_fm3_step_compression.png"
    plt.savefig(p3, dpi=300)
    plt.close()
    paths.append(p3)

    # --- Figure 4: Diffusion Trajectory Anatomy (Entropy curves + Unmasking Order Scatter) ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    curve_specs = [
        ("dream_base_entropy", "Dream-Base (entropy)", "#4C78A8", "--"),
        ("fm4_dream_inst_entropy", "Dream-Inst (entropy)", "#1F77B4", "-"),
        ("fm4_dream_inst_origin", "Dream-Inst (origin)", "#2CA02C", "-"),
        ("fm4_dream_inst_maskgit_plus", "Dream-Inst (maskgit_plus)", "#D62728", "-"),
        ("fm4_dream_inst_topk_margin", "Dream-Inst (topk_margin)", "#FF7F0E", "-."),
    ]
    for rk, lbl, col, ls in curve_specs:
        c = runs[rk]["mean_step_mask_entropy_by_dataset"]["infinity_chat100"]
        t_norm = np.linspace(0.0, 1.0, len(c))
        axes[0].plot(t_norm, c, label=lbl, color=col, linestyle=ls, linewidth=2.0)
    axes[0].set_xlabel("Normalized Denoising Time (t / T, T=128)")
    axes[0].set_ylabel("Mean Remaining Mask Entropy H_t (bits)")
    axes[0].set_title("(a) Mask Entropy Decay on INFINITY-CHAT100")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25)

    # Panel B: Real Unmasking Order Trajectory tau_{k, j} on infchat_003 (sample 0)
    for rk, lbl, col, mk in [
        ("fm4_dream_inst_entropy", "entropy (LTR locked, ρ=1.00)", "#1F77B4", "o"),
        ("fm4_dream_inst_origin", "origin (non-linear, ρ≈0.02)", "#2CA02C", "^"),
        ("fm4_dream_inst_maskgit_plus", "maskgit_plus (EOS backward flood)", "#D62728", "x"),
    ]:
        npz = np.load(R5_DIR / rk / "trajectories" / "infchat_003.npz")
        tau0 = npz["unmask_steps"][0]
        pos = np.arange(len(tau0))
        axes[1].scatter(pos, tau0, s=14, alpha=0.75, color=col, marker=mk, label=lbl)
    axes[1].set_xlabel("Token Position j (0 .. 127)")
    axes[1].set_ylabel("Unmasking Step τ (1 .. 128)")
    axes[1].set_title("(b) Token Unmasking Trajectory on infchat_003")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.25)

    plt.tight_layout()
    p4 = FIG_DIR / "fig4_trajectory_anatomy.png"
    plt.savefig(p4, dpi=300)
    plt.close()
    paths.append(p4)

    # --- Figure 5: Paired Effect Sizes (Forest Plot) & Open-Ended Task-Type Breakdown ---
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))
    pt = detailed["paired_tests"]
    forest_items = [
        ("Qwen CovQA Recall (Base->Inst)", pt["qwen_covqa_recall"], "#D62728"),
        ("Dream CovQA Recall (Base->Inst)", pt["dream_covqa_recall"], "#1F77B4"),
        ("FM-4 Origin vs Entropy Recall", pt["fm4_origin_vs_entropy_covqa_recall"], "#2CA02C"),
        ("Qwen InfChat VS_0.5 (Base->Inst)", pt["qwen_inf_vs05"], "#FF7F0E"),
        ("Dream InfChat VS_0.5 (Base->Inst)", pt["dream_inf_vs05"], "#4C78A8"),
        ("Dream-Inst vs Qwen-Inst VS_0.5", pt["inst_dream_vs_qwen_vs05"], "#9467BD"),
    ]
    y_pos = np.arange(len(forest_items))[::-1]
    axes[0].axvline(0.0, color="black", linestyle=":", linewidth=1.2)
    for yp, (lbl, stat, col) in zip(y_pos, forest_items):
        m = stat["diff_mean"]
        lo = m - stat["diff_ci_low"]
        hi = stat["diff_ci_high"] - m
        axes[0].errorbar(m, yp, xerr=[[lo], [hi]], fmt="o", color=col, ecolor=col, elinewidth=2, capsize=4, markersize=6)
        axes[0].text(stat["diff_ci_high"] + 0.03, yp, f"p={stat['p_value']:.4f}", va="center", fontsize=8)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels([x[0] for x in forest_items], fontsize=8.5)
    axes[0].set_xlim(-0.95, 1.15)
    axes[0].set_xlabel("Paired Difference Δ (with 95% Bootstrap CI)")
    axes[0].set_title("(a) Key Paired Effect Sizes & Significance")
    axes[0].grid(True, axis="x", alpha=0.25)

    # Panel B: Task-type breakdown (Creative Writing vs Other Open Chat)
    r_keys = ["dream_base_entropy", "dream_inst_entropy", "qwen_base_ar", "qwen_inst_ar"]
    r_lbls = ["Dream-Base", "Dream-Inst", "Qwen-Base", "Qwen-Inst"]
    cw_vs = [runs[k]["infinity_chat"]["by_task_type"]["creative_writing"]["vs_0.5"]["mean"] for k in r_keys]
    ot_vs = [runs[k]["infinity_chat"]["by_task_type"]["other"]["vs_0.5"]["mean"] for k in r_keys]
    xb = np.arange(4)
    axes[1].bar(xb - 0.18, cw_vs, width=0.35, color="#8C564B", alpha=0.9, label="Creative Writing (n=8)")
    axes[1].bar(xb + 0.18, ot_vs, width=0.35, color="#17BECF", alpha=0.85, label="Other Open Queries (n=20)")
    axes[1].set_xticks(xb)
    axes[1].set_xticklabels(r_lbls, rotation=15, ha="right")
    axes[1].set_ylabel("Vendi Score (q=0.5)")
    axes[1].set_title("(b) Semantic Diversity by Query Category")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    p5 = FIG_DIR / "fig5_r5_effect_sizes_and_tasks.png"
    plt.savefig(p5, dpi=300)
    plt.close()
    paths.append(p5)

    for p in paths:
        shutil.copy2(p, DOCS_FIG_DIR / p.name)
    return [str(p) for p in paths]


def main() -> None:
    eval_data = json.loads((R5_DIR / "r5_eval_report.json").read_text(encoding="utf-8"))
    detailed = compute_detailed_stats(eval_data)
    (R5_DIR / "r5_detailed_stats.json").write_text(json.dumps(detailed, indent=2, ensure_ascii=False), encoding="utf-8")
    figs = make_figures(eval_data, detailed)
    print("Detailed stats + figures generated:", figs)
    print(json.dumps(detailed, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
