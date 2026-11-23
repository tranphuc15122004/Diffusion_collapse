"""Evaluate benchmark run directories across all 3 metric tiers + claim clustering + cross-model similarity."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_utils import classify_open_query, load_frozen_dataset, load_protocol
from src.metrics.diversity import (
    BGEEmbedder,
    compute_mtld,
    cross_model_similarity,
    distinct_idea_clusters,
    mean_pairwise_similarity,
    split_into_claims,
    vendi_spectrum,
)
from src.metrics.fluency import ExternalPPLScorer
from src.metrics.knowledge import compute_coverageqa_metrics, compute_popqa_metrics
from src.metrics.stats import bootstrap_ci
from src.metrics.trajectory import compute_unmask_order_metrics, mask_tau_active_pre_eos


def load_dataset_index() -> dict[str, dict]:
    index: dict[str, dict] = {}
    for name in ("coverageqa", "popqa_stratified", "infinity_chat100"):
        for rec in load_frozen_dataset(name):
            index[rec["id"]] = rec
    return index


def evaluate_directory(
    run_dir: Path,
    ds_index: dict[str, dict],
    embedder: BGEEmbedder,
    proto: dict[str, Any],
    ppl_scorer: ExternalPPLScorer | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    pred_path = run_dir / "predictions.jsonl"
    traj_dir = run_dir / "trajectories"
    meta = json.loads((run_dir / "run_meta.json").read_text()) if (run_dir / "run_meta.json").exists() else {}

    rows = [json.loads(ln) for ln in pred_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    cov_recalls, cov_precs, cov_refs, cov_eblr = [], [], [], []
    pop_head_acc, pop_tail_acc, pop_halluc = [], [], []
    lem_wrong, lem_right = [], []

    inf_vs_05, inf_vs_1, inf_vs_inf, inf_pairwise, inf_mtld, inf_ppl = [], [], [], [], [], []
    inf_n_claims, inf_n_clusters, inf_novelty = [], [], []
    inf_h_ord, inf_rho, inf_act_len = [], [], []
    by_task: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    inf_embeddings_by_id: dict[str, np.ndarray] = {}
    entropy_curves: list[np.ndarray] = []
    entropy_curves_by_ds: dict[str, list[np.ndarray]] = defaultdict(list)

    for row in rows:
        qid = row["id"]
        gt = ds_index[qid]
        ds = row["dataset"]
        preds = row["predictions"]
        eblr_list = row.get("eblr_per_sample", [])
        npz_path = traj_dir / f"{qid}.npz"

        if npz_path.exists():
            npz = np.load(npz_path)
            ent = npz["step_mask_entropy"].astype(np.float64)
            c = ent.mean(axis=0)
            entropy_curves_by_ds[ds].append(c)
            grid_x = np.linspace(0.0, 1.0, 32)
            src_x = np.linspace(0.0, 1.0, len(c))
            entropy_curves.append(np.interp(grid_x, src_x, c))
            masked_tau, _ = mask_tau_active_pre_eos(npz["gen_ids"], npz["unmask_steps"])
            ord_m = compute_unmask_order_metrics(masked_tau, min_active_tokens=4)
        else:
            ord_m = None

        if ds == "coverageqa":
            m = compute_coverageqa_metrics(preds, gt["answers"], category=gt.get("category"))
            cov_recalls.append(m["recall"])
            cov_precs.append(m["precision"])
            cov_refs.append(m["refusal_rate"])
            if eblr_list:
                cov_eblr.append(float(np.mean(eblr_list)))

        elif ds == "popqa_stratified":
            m = compute_popqa_metrics(preds, gt["possible_answers"])
            if gt["stratum"] == "head":
                pop_head_acc.append(m["acc"])
            else:
                pop_tail_acc.append(m["acc"])
            pop_halluc.append(m["hallucination_rate"])
            for lem in row.get("lem_records", []):
                if lem["is_correct"]:
                    lem_right.append(lem["delta_c"])
                else:
                    lem_wrong.append(lem["delta_c"])

        elif ds == "infinity_chat100":
            embs = embedder.encode(preds, batch_size=8, max_length=256)
            inf_embeddings_by_id[qid] = embs
            spec = vendi_spectrum(embs, orders=(0.5, 1.0, np.inf))
            pw_sim = mean_pairwise_similarity(embs)
            mtld_val = float(np.mean([compute_mtld(p) for p in preds]))

            all_claims: list[str] = []
            for p in preds:
                all_claims.extend(split_into_claims(p))
            if all_claims:
                c_embs = embedder.encode(all_claims, batch_size=16, max_length=128)
                clust = distinct_idea_clusters(c_embs, sim_threshold=0.78)
            else:
                clust = {"n_claims": 0.0, "n_clusters": 0.0, "idea_novelty_ratio": 0.0}

            inf_vs_05.append(spec["vs_0.5"])
            inf_vs_1.append(spec["vs_1"])
            inf_vs_inf.append(spec["vs_inf"])
            inf_pairwise.append(pw_sim)
            inf_mtld.append(mtld_val)
            if ppl_scorer is not None:
                ppls = [v for v in ppl_scorer.score_texts(preds) if np.isfinite(v)]
                if ppls:
                    inf_ppl.append(float(np.median(ppls)))
            inf_n_claims.append(clust["n_claims"])
            inf_n_clusters.append(clust["n_clusters"])
            inf_novelty.append(clust["idea_novelty_ratio"])

            t_cat = classify_open_query(gt["query"])
            by_task[t_cat]["vs_0.5"].append(spec["vs_0.5"])
            by_task[t_cat]["n_clusters"].append(clust["n_clusters"])
            by_task[t_cat]["pairwise_sim"].append(pw_sim)

            if ord_m is not None:
                inf_h_ord.append(ord_m["h_order_bits"])
                if np.isfinite(ord_m["rho_ltr"]):
                    inf_rho.append(ord_m["rho_ltr"])
                inf_act_len.append(ord_m["mean_active_tokens"])

    n_boot = int(proto["metrics"]["bootstrap_resamples"])
    mean_ent_curve = (
        np.mean(np.stack(entropy_curves, axis=0), axis=0).tolist()
        if entropy_curves
        else []
    )
    mean_ent_by_ds = {
        ds_k: np.mean(np.stack(cs, axis=0), axis=0).tolist()
        for ds_k, cs in sorted(entropy_curves_by_ds.items())
    }
    task_summary = {
        cat: {k: bootstrap_ci(vals, n_boot=n_boot) for k, vals in metrics_map.items()}
        for cat, metrics_map in sorted(by_task.items())
    }

    report = {
        "run_name": run_dir.name,
        "meta": meta,
        "n_questions": len(rows),
        "coverageqa": {
            "recall": bootstrap_ci(cov_recalls, n_boot=n_boot),
            "precision": bootstrap_ci(cov_precs, n_boot=n_boot),
            "refusal_rate": bootstrap_ci(cov_refs, n_boot=n_boot),
            "eblr": bootstrap_ci(cov_eblr, n_boot=n_boot),
        },
        "popqa": {
            "head_acc": bootstrap_ci(pop_head_acc, n_boot=n_boot),
            "tail_acc": bootstrap_ci(pop_tail_acc, n_boot=n_boot),
            "hallucination_rate": bootstrap_ci(pop_halluc, n_boot=n_boot),
            "delta_c_wrong": bootstrap_ci(lem_wrong, n_boot=n_boot),
            "delta_c_right": bootstrap_ci(lem_right, n_boot=n_boot),
        },
        "infinity_chat": {
            "vs_0.5": bootstrap_ci(inf_vs_05, n_boot=n_boot),
            "vs_1": bootstrap_ci(inf_vs_1, n_boot=n_boot),
            "vs_inf": bootstrap_ci(inf_vs_inf, n_boot=n_boot),
            "pairwise_sim": bootstrap_ci(inf_pairwise, n_boot=n_boot),
            "distinct_idea_clusters": bootstrap_ci(inf_n_clusters, n_boot=n_boot),
            "total_claims": bootstrap_ci(inf_n_claims, n_boot=n_boot),
            "idea_novelty_ratio": bootstrap_ci(inf_novelty, n_boot=n_boot),
            "mtld": bootstrap_ci(inf_mtld, n_boot=n_boot),
            "external_ppl": bootstrap_ci(inf_ppl, n_boot=n_boot),
            "h_order_bits": bootstrap_ci(inf_h_ord, n_boot=n_boot),
            "rho_ltr": bootstrap_ci(inf_rho, n_boot=n_boot),
            "mean_active_tokens": bootstrap_ci(inf_act_len, n_boot=n_boot),
            "by_task_type": task_summary,
        },
        "mean_step_mask_entropy_curve": mean_ent_curve,
        "mean_step_mask_entropy_by_dataset": mean_ent_by_ds,
    }
    return report, inf_embeddings_by_id


def compute_cross_model_matrix(
    run_names: list[str],
    emb_maps: list[dict[str, np.ndarray]],
    n_boot: int,
) -> dict[str, dict[str, Any]]:
    matrix: dict[str, dict[str, Any]] = {}
    for i, name_a in enumerate(run_names):
        if not emb_maps[i]:
            continue
        matrix[name_a] = {}
        for j, name_b in enumerate(run_names):
            if not emb_maps[j]:
                continue
            shared_ids = sorted(set(emb_maps[i]) & set(emb_maps[j]))
            if not shared_ids:
                continue
            if i == j:
                sims = [mean_pairwise_similarity(emb_maps[i][qid]) for qid in shared_ids]
            else:
                sims = [cross_model_similarity(emb_maps[i][qid], emb_maps[j][qid]) for qid in shared_ids]
            matrix[name_a][name_b] = bootstrap_ci(sims, n_boot=n_boot)
    return matrix


def make_plots(results: list[dict[str, Any]], plot_dir: Path) -> list[str]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    ent_runs = [r for r in results if r["mean_step_mask_entropy_curve"]]
    if ent_runs:
        plt.figure(figsize=(7, 4))
        for r in ent_runs:
            curve = r["mean_step_mask_entropy_curve"]
            x = np.linspace(0.0, 1.0, len(curve))
            plt.plot(x, curve, label=r["run_name"], linewidth=2)
        plt.xlabel("Normalized Denoising Step (t / T)")
        plt.ylabel("Mean Mask Entropy H_t (bits)")
        plt.title("Stepwise Mask Entropy Curves")
        plt.legend()
        plt.grid(True, alpha=0.3)
        p1 = plot_dir / "stepwise_mask_entropy.png"
        plt.tight_layout()
        plt.savefig(p1, dpi=150)
        plt.close()
        saved.append(str(p1))

    names = [r["run_name"] for r in results]
    recalls = [r["coverageqa"]["recall"]["mean"] for r in results]
    precs = [r["coverageqa"]["precision"]["mean"] for r in results]
    vs05 = [r["infinity_chat"]["vs_0.5"]["mean"] for r in results]
    clusters = [r["infinity_chat"]["distinct_idea_clusters"]["mean"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    x = np.arange(len(names))
    axes[0].bar(x - 0.18, recalls, width=0.36, label="Recall@K")
    axes[0].bar(x + 0.18, precs, width=0.36, label="Precision@K")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    axes[0].set_title("CoverageQA: Recall vs Precision")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, vs05, width=0.5, color="tab:green")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    axes[1].set_title("INFINITY-CHAT: Vendi Score (q=0.5)")
    axes[1].grid(True, axis="y", alpha=0.3)

    axes[2].bar(x, clusters, width=0.5, color="tab:purple")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    axes[2].set_title("INFINITY-CHAT: Distinct Idea Clusters")
    axes[2].grid(True, axis="y", alpha=0.3)

    p2 = plot_dir / "summary_metrics.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=150)
    plt.close()
    saved.append(str(p2))
    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--plot-dir", required=True)
    parser.add_argument("--compute-ppl", action="store_true", help="Score INFINITY-CHAT fluency via ExternalPPLScorer")
    args = parser.parse_args()

    proto = load_protocol()
    ds_index = load_dataset_index()
    embedder = BGEEmbedder(proto["models"]["evaluator"]["embedder"])

    results = []
    emb_maps = []
    run_names = []
    for rd in args.run_dirs:
        rep, emap = evaluate_directory(Path(rd), ds_index, embedder, proto, ppl_scorer=None)
        results.append(rep)
        emb_maps.append(emap)
        run_names.append(rep["run_name"])

    embedder.unload()

    if args.compute_ppl:
        scorer = ExternalPPLScorer(proto["models"]["evaluator"]["external_ppl"], proto)
        n_boot = int(proto["metrics"]["bootstrap_resamples"])
        for rd, rep in zip(args.run_dirs, results):
            pred_path = Path(rd) / "predictions.jsonl"
            inf_ppls: list[float] = []
            for ln in pred_path.read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                row = json.loads(ln)
                if row["dataset"] == "infinity_chat100":
                    vals = [v for v in scorer.score_texts(row["predictions"]) if np.isfinite(v)]
                    if vals:
                        inf_ppls.append(float(np.median(vals)))
            rep["infinity_chat"]["external_ppl"] = bootstrap_ci(inf_ppls, n_boot=n_boot)
        scorer.unload()
    cross_matrix = compute_cross_model_matrix(
        run_names, emb_maps, n_boot=int(proto["metrics"]["bootstrap_resamples"])
    )
    plots = make_plots(results, Path(args.plot_dir))

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "runs": results,
        "cross_model_similarity": cross_matrix,
        "plots": plots,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Evaluation saved to {out_path}; plots: {plots}")


if __name__ == "__main__":
    main()
