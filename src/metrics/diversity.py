"""Tier-2 spectral semantic diversity (Vendi Score), BGE-M3 embedder, claim clustering, and MTLD."""

from __future__ import annotations

import gc
import math
import re
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoModel, AutoTokenizer


class BGEEmbedder:
    """Lightweight BAAI/bge-m3 CLS-pooled normalized embedder using transformers directly."""

    def __init__(self, model_id: str = "BAAI/bge-m3", device: str | None = None) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: Sequence[str], batch_size: int = 16, max_length: int = 512) -> np.ndarray:
        if not texts:
            return np.empty((0, 1024), dtype=np.float32)
        chunks: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = [t if t.strip() else " " for t in texts[i : i + batch_size]]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            out = self.model(**enc)
            # BGE-M3 uses CLS pooling (1_Pooling/config.json: pooling_mode_cls_token=true)
            cls_emb = F.normalize(out.last_hidden_state[:, 0].float(), p=2, dim=-1)
            chunks.append(cls_emb.cpu().numpy())
        return np.vstack(chunks)

    def unload(self) -> None:
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    emb = np.asarray(embeddings, dtype=np.float64)
    if emb.ndim != 2 or emb.shape[0] == 0:
        raise ValueError(f"Expected non-empty 2D embeddings, got shape {emb.shape}")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return emb / norms


def vendi_score(embeddings: np.ndarray, q: float = 1.0) -> float:
    """Compute Vendi Score of order q from K x D embeddings."""
    emb = _l2_normalize(embeddings)
    if emb.shape[0] == 1:
        return 1.0

    k = emb.shape[0]
    sim = (emb @ emb.T) / k
    sim = 0.5 * (sim + sim.T)

    eigvals = np.linalg.eigvalsh(sim)
    eigvals = eigvals[eigvals > 1e-12]
    if eigvals.size == 0:
        return 1.0
    eigvals = eigvals / eigvals.sum()

    if np.isinf(q):
        return float(1.0 / np.max(eigvals))
    if abs(q - 1.0) < 1e-8:
        entropy = -np.sum(eigvals * np.log(eigvals))
        return float(np.exp(entropy))
    return float(np.exp((1.0 / (1.0 - q)) * np.log(np.sum(eigvals**q))))


def vendi_spectrum(embeddings: np.ndarray, orders: Sequence[float] = (0.5, 1.0, np.inf)) -> dict[str, float]:
    out: dict[str, float] = {}
    for q in orders:
        key = "vs_inf" if np.isinf(q) else f"vs_{q:g}"
        out[key] = vendi_score(embeddings, q=q)
    return out


def mean_pairwise_similarity(embeddings: np.ndarray) -> float:
    """Mean off-diagonal cosine similarity within a set of K response embeddings."""
    emb = _l2_normalize(embeddings)
    k = emb.shape[0]
    if k < 2:
        return 1.0
    sim = emb @ emb.T
    triu = sim[np.triu_indices(k, k=1)]
    return float(np.mean(triu))


def cross_model_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Mean pairwise cosine similarity between responses from two models on the same prompt."""
    a = _l2_normalize(emb_a)
    b = _l2_normalize(emb_b)
    return float(np.mean(a @ b.T))


def split_into_claims(text: str, min_chars: int = 15) -> list[str]:
    """Split a generated response into distinct sentences or bullet-point claims."""
    if not text or not text.strip():
        return []
    # Split on newlines/bullet markers or sentence-ending punctuation
    chunks = re.split(r"(?:\n+|(?<=[.!?])\s+)", text.strip())
    claims: list[str] = []
    for ch in chunks:
        cleaned = re.sub(r"^[\-\*\d\.\)\s]+", "", ch).strip()
        if len(cleaned) >= min_chars:
            claims.append(cleaned)
    if not claims and len(text.strip()) >= 3:
        return [text.strip()]
    return claims


def distinct_idea_clusters(
    claim_embeddings: np.ndarray,
    sim_threshold: float = 0.78,
) -> dict[str, float]:
    """Cluster claim-level embeddings using cosine average linkage to count distinct arguments/ideas.

    Returns:
        n_claims: Total extracted claims across K responses.
        n_clusters: Number of distinct semantic idea clusters (cosine similarity >= sim_threshold).
        idea_novelty_ratio: n_clusters / n_claims (1.0 = every claim is distinct; near 0 = heavy repetition).
    """
    if claim_embeddings.ndim != 2 or claim_embeddings.shape[0] == 0:
        return {"n_claims": 0.0, "n_clusters": 0.0, "idea_novelty_ratio": 0.0}
    n = claim_embeddings.shape[0]
    if n == 1:
        return {"n_claims": 1.0, "n_clusters": 1.0, "idea_novelty_ratio": 1.0}

    emb = _l2_normalize(claim_embeddings)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=1.0 - sim_threshold,
    )
    labels = clustering.fit_predict(emb)
    n_clusters = int(len(set(labels.tolist())))
    return {
        "n_claims": float(n),
        "n_clusters": float(n_clusters),
        "idea_novelty_ratio": float(n_clusters / n),
    }


def _mtld_one_direction(tokens: Sequence[str], threshold: float = 0.72) -> float:
    if not tokens:
        return 0.0
    factors = 0.0
    types: set[str] = set()
    token_count = 0
    for tok in tokens:
        token_count += 1
        types.add(tok)
        ttr = len(types) / token_count
        if ttr <= threshold:
            factors += 1.0
            types.clear()
            token_count = 0
    if token_count > 0:
        ttr = len(types) / token_count
        if ttr < 1.0:
            factors += (1.0 - ttr) / (1.0 - threshold)
    if factors == 0.0:
        return float(len(tokens))
    return float(len(tokens) / factors)


def compute_mtld(text: str, threshold: float = 0.72) -> float:
    """Bidirectional Measure of Textual Lexical Diversity (McCarthy & Jarvis, 2010)."""
    tokens = re.findall(r"[A-Za-z0-9']+", text.lower())
    if len(tokens) < 2:
        return float(len(tokens))
    fwd = _mtld_one_direction(tokens, threshold=threshold)
    bwd = _mtld_one_direction(list(reversed(tokens)), threshold=threshold)
    val = 0.5 * (fwd + bwd)
    return float(val) if math.isfinite(val) else 0.0
