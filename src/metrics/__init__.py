"""Three-tier metric suite for knowledge, diversity, and diffusion trajectory diagnostics."""

from .diversity import (
    BGEEmbedder,
    compute_mtld,
    cross_model_similarity,
    distinct_idea_clusters,
    mean_pairwise_similarity,
    split_into_claims,
    vendi_score,
    vendi_spectrum,
)
from .knowledge import (
    clean_qa_prediction,
    compute_coverageqa_metrics,
    compute_popqa_metrics,
    compute_thr,
    is_refusal,
    normalize_text,
)
from .stats import bootstrap_ci, paired_permutation_test
from .trajectory import (
    compute_eblr,
    compute_leave_entity_masked_prob,
    compute_unmask_order_metrics,
    mask_tau_active_pre_eos,
)

__all__ = [
    "BGEEmbedder",
    "bootstrap_ci",
    "clean_qa_prediction",
    "compute_coverageqa_metrics",
    "compute_eblr",
    "compute_leave_entity_masked_prob",
    "compute_mtld",
    "compute_popqa_metrics",
    "compute_thr",
    "compute_unmask_order_metrics",
    "cross_model_similarity",
    "distinct_idea_clusters",
    "is_refusal",
    "mask_tau_active_pre_eos",
    "mean_pairwise_similarity",
    "normalize_text",
    "paired_permutation_test",
    "split_into_claims",
    "vendi_score",
    "vendi_spectrum",
]
