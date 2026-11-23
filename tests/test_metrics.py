"""Unit tests for all three metric tiers (knowledge, diversity, trajectory)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_utils import (
    classify_open_query,
    load_frozen_dataset,
    load_protocol,
    strip_generation_text,
)
from src.metrics.diversity import (
    compute_mtld,
    cross_model_similarity,
    distinct_idea_clusters,
    mean_pairwise_similarity,
    split_into_claims,
    vendi_score,
    vendi_spectrum,
)
from src.metrics.knowledge import (
    compute_coverageqa_metrics,
    compute_popqa_metrics,
    compute_thr,
)
from src.metrics.stats import bootstrap_ci, paired_permutation_test
from src.metrics.trajectory import (
    compute_eblr,
    compute_unmask_order_metrics,
    mask_tau_active_pre_eos,
)


def test_coverageqa_alias_dedup_and_precision():
    gt = [["UAE", "United Arab Emirates"], ["Egypt"], ["Kuwait"], ["Oman"]]
    preds = [
        "UAE",
        "United Arab Emirates",  # same group as UAE -> counts once
        "Egypt",
        "Wakanda",               # invalid hallucination
        "I am not sure",         # refusal -> excluded from precision denominator
    ]
    res = compute_coverageqa_metrics(preds, gt, category="country")
    assert math.isclose(res["unique_valid"], 2.0)
    assert math.isclose(res["unique_total"], 3.0)
    assert math.isclose(res["recall"], 2.0 / 4.0)
    assert math.isclose(res["precision"], 2.0 / 3.0)
    assert math.isclose(res["refusal_rate"], 1.0 / 5.0)
    assert math.isclose(res["hallucination_rate"], 1.0 / 3.0)


def test_popqa_and_thr():
    preds = ["politician", "actor", "I don't know"]
    res = compute_popqa_metrics(preds, ["politician", "political leader"])
    assert math.isclose(res["acc"], 1.0 / 3.0)
    assert math.isclose(res["precision_non_refusal"], 0.5)
    assert math.isclose(res["hallucination_rate"], 0.5)
    assert math.isclose(res["pass_at_k"], 1.0)

    thr = compute_thr(
        acc_tail_instruct=0.10,
        acc_tail_base=0.40,
        acc_head_instruct=0.72,
        acc_head_base=0.80,
    )
    # (0.25) / (0.90)
    assert math.isclose(thr, 0.25 / 0.90, rel_tol=1e-6)


def test_vendi_score_extremes_and_monotonicity():
    identical = np.ones((8, 16), dtype=np.float64)
    assert math.isclose(vendi_score(identical, q=1.0), 1.0, abs_tol=1e-6)
    assert math.isclose(vendi_score(identical, q=0.5), 1.0, abs_tol=1e-6)
    assert math.isclose(vendi_score(identical, q=np.inf), 1.0, abs_tol=1e-6)

    orthogonal = np.eye(6, dtype=np.float64)
    for q in (0.5, 1.0, np.inf):
        assert math.isclose(vendi_score(orthogonal, q=q), 6.0, abs_tol=1e-6)

    rng = np.random.default_rng(0)
    skewed = np.vstack([np.eye(4)[0]] * 5 + [np.eye(4)[1], np.eye(4)[2], np.eye(4)[3]])
    skewed = skewed + 0.01 * rng.normal(size=skewed.shape)
    spec = vendi_spectrum(skewed)
    assert spec["vs_0.5"] >= spec["vs_1"] >= spec["vs_inf"]


def test_mtld_behavior():
    repetitive = " ".join(["cat"] * 40)
    diverse = " ".join(f"word{i}" for i in range(40))
    assert compute_mtld(diverse) > compute_mtld(repetitive)
    assert compute_mtld("") == 0.0
    assert compute_mtld("hello") == 1.0


def test_unmask_order_metrics_ltr_vs_nonlinear():
    # Rigid left-to-right across K=4, L=8
    ltr = np.tile(np.arange(1, 9), (4, 1))
    res_ltr = compute_unmask_order_metrics(ltr)
    assert math.isclose(res_ltr["h_order_bits"], 0.0, abs_tol=1e-9)
    assert math.isclose(res_ltr["rho_ltr"], 1.0, abs_tol=1e-9)

    # Diverse non-linear orders across K=4, L=8
    rng = np.random.default_rng(42)
    varied = np.vstack([rng.permutation(np.arange(1, 9)) for _ in range(4)])
    res_var = compute_unmask_order_metrics(varied)
    assert res_var["h_order_bits"] > 0.5
    assert res_var["rho_ltr"] < 0.9


class _DummyTok:
    _map = {1: "the", 2: "capital", 3: "of", 4: "Paris", 5: ",", 99: "<|endoftext|>"}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self._map.get(i, "unk") for i in ids if not (skip_special_tokens and i == 99))


def test_eblr():
    tok = _DummyTok()
    # T=8 -> early cutoff = 2
    token_ids = [1, 3, 5, 2, 4, 99]
    steps = [1, 2, 2, 4, 6, 1]
    eblr = compute_eblr(token_ids, steps, tok, boilerplate_tokens={"the", "of"}, total_steps=8, eos_ids=(99,))
    # Steps <= 2 excluding eos(99): token 1 ("the"), 3 ("of"), 5 (",") -> all 3 boilerplate/punct -> 1.0
    assert math.isclose(eblr, 1.0)


def test_frozen_datasets_and_strip():
    proto = load_protocol()
    assert len(load_frozen_dataset("coverageqa", pilot_only=True)) == 30
    assert len(load_frozen_dataset("coverageqa", core_105_only=True)) == 105
    assert len(load_frozen_dataset("popqa_stratified", pilot_only=True)) == 40
    assert len(load_frozen_dataset("infinity_chat100", pilot_only=True)) == 30
    assert len(load_frozen_dataset("coverageqa")) + len(load_frozen_dataset("popqa_stratified")) + len(load_frozen_dataset("infinity_chat100")) == 445
    assert strip_generation_text("Tokyo\nQuestion: Next?", "popqa_stratified") == "Tokyo"
    assert strip_generation_text("Hello world<|im_end|>ignored", "infinity_chat100") == "Hello world"
    assert proto["hardware"]["batch_size"] == 4
    bp = proto["metrics"]["eblr_boilerplate_tokens"]
    assert all(isinstance(t, str) for t in bp), f"Non-string boilerplate token in YAML: {bp}"


def test_mask_tau_active_pre_eos_and_bootstrap():
    # Two sequences of length 6: EOS (151643) at index 3 and index 1
    gen_ids = np.array([
        [10, 11, 12, 151643, 151643, 151643],
        [20, 151645, 151643, 151643, 151643, 151643],
    ])
    tau = np.array([
        [5, 6, 7, 1, 1, 1],
        [4, 2, 1, 1, 1, 1],
    ])
    masked, active_lens = mask_tau_active_pre_eos(gen_ids, tau)
    assert active_lens.tolist() == [3, 1]
    assert masked[0].tolist() == [5, 6, 7, 0, 0, 0]
    assert masked[1].tolist() == [4, 0, 0, 0, 0, 0]

    # With min_active_tokens=3, only row 0 contributes to rho_ltr -> +1.0 (not 0.0)
    ord_res = compute_unmask_order_metrics(masked, min_active_tokens=3)
    assert math.isclose(ord_res["rho_ltr"], 1.0, abs_tol=1e-9)
    assert ord_res["n_valid_rho_seqs"] == 1.0

    ci = bootstrap_ci([1.0, 2.0, 3.0, 4.0], n_boot=200, seed=42)
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]
    perm = paired_permutation_test([1.0, 1.0, 1.0, 1.0], [3.0, 3.0, 3.0, 3.0], n_perm=500, seed=42)
    assert perm["diff_mean"] == 2.0 and perm["p_value"] < 0.2


def test_pairwise_cross_and_claim_clustering():
    identical = np.ones((4, 8), dtype=np.float64)
    assert math.isclose(mean_pairwise_similarity(identical), 1.0, abs_tol=1e-6)
    orth = np.eye(4, dtype=np.float64)
    assert math.isclose(mean_pairwise_similarity(orth), 0.0, abs_tol=1e-6)

    a = np.array([[1.0, 0.0], [1.0, 0.0]])
    b = np.array([[0.0, 1.0], [0.0, 1.0]])
    assert math.isclose(cross_model_similarity(a, b), 0.0, abs_tol=1e-6)
    assert math.isclose(cross_model_similarity(a, a), 1.0, abs_tol=1e-6)

    claims = split_into_claims("1. First strong argument about energy.\n2. Second unrelated point on cost. Tiny.")
    assert len(claims) == 2

    # Two tight clusters of 2 claims each (2 orthogonal directions)
    c_embs = np.array([
        [1.0, 0.0, 0.0],
        [0.99, 0.05, 0.0],
        [0.0, 1.0, 0.0],
        [0.05, 0.99, 0.0],
    ])
    cl = distinct_idea_clusters(c_embs, sim_threshold=0.78)
    assert cl["n_claims"] == 4.0
    assert cl["n_clusters"] == 2.0
    assert math.isclose(cl["idea_novelty_ratio"], 0.5)

    assert classify_open_query("Write me a 200-word cover letter.") == "creative_writing"
    assert classify_open_query("Suggest 5 ideas for a startup.") == "brainstorming_ideation"
    assert classify_open_query("Explain why the sky is blue.") == "explanation_analysis"

