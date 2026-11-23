"""R0 Dataset Freezing Script.

Builds deterministic, leakage-checked, stratified JSONL files in data/frozen/:
  1. coverageqa.jsonl        (145 questions; 30 marked pilot_split=True; 105 marked in_105_core=True)
  2. popqa_stratified.jsonl  (200 questions: 100 Head + 100 Tail; 40 marked pilot_split=True)
  3. infinity_chat100.jsonl  (100 open-ended queries; 30 marked pilot_split=True)
  4. manifest.json           (checksums, counts, quantiles, overlap verification)
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
RAW_COV_PATH = ROOT / "data" / "raw" / "CoverageQA.json"
FROZEN_DIR = ROOT / "data" / "frozen"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_coverageqa(rng: random.Random) -> tuple[list[dict], dict]:
    with open(RAW_COV_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for idx, (question, payload) in enumerate(sorted(raw.items(), key=lambda kv: kv[0])):
        cat = payload.get("category", "unknown")
        answers = [[a.strip() for a in group if a.strip()] for group in payload["answers"]]
        answers = [g for g in answers if g]
        rec = {
            "id": f"covqa_{idx:03d}",
            "dataset": "coverageqa",
            "question": question.strip(),
            "category": cat,
            "in_105_core": cat in ("country", "physic"),
            "answers": answers,
            "num_answer_groups": len(answers),
            "pilot_split": False,
        }
        by_cat[cat].append(rec)

    # Stratified 30-question pilot split proportional to category counts (at least 1 per category)
    total = sum(len(v) for v in by_cat.values())
    pilot_target = 30
    alloc = {cat: max(1, round(pilot_target * len(items) / total)) for cat, items in by_cat.items()}
    # Adjust rounding so sum == 30
    while sum(alloc.values()) > pilot_target:
        largest = max(alloc, key=lambda c: alloc[c])
        alloc[largest] -= 1
    while sum(alloc.values()) < pilot_target:
        largest = max(by_cat, key=lambda c: len(by_cat[c]) - alloc[c])
        alloc[largest] += 1

    records: list[dict] = []
    for cat in sorted(by_cat):
        items = list(by_cat[cat])
        rng.shuffle(items)
        for j, rec in enumerate(items):
            if j < alloc[cat]:
                rec["pilot_split"] = True
            records.append(rec)

    records.sort(key=lambda r: r["id"])
    meta = {
        "total": len(records),
        "in_105_core": sum(1 for r in records if r["in_105_core"]),
        "pilot_n": sum(1 for r in records if r["pilot_split"]),
        "categories": dict(Counter(r["category"] for r in records)),
        "pilot_categories": dict(Counter(r["category"] for r in records if r["pilot_split"])),
        "avg_answer_groups": float(np.mean([r["num_answer_groups"] for r in records])),
    }
    return records, meta


def stratified_sample_by_prop(items: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Sample n items evenly across PopQA relation properties (`prop`)."""
    by_prop: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_prop[item["prop"]].append(item)
    for prop in by_prop:
        rng.shuffle(by_prop[prop])

    selected: list[dict] = []
    props = sorted(by_prop.keys())
    rng.shuffle(props)
    idx_map = {p: 0 for p in props}
    while len(selected) < n:
        progressed = False
        for p in props:
            if idx_map[p] < len(by_prop[p]):
                selected.append(by_prop[p][idx_map[p]])
                idx_map[p] += 1
                progressed = True
                if len(selected) == n:
                    break
        if not progressed:
            break
    return selected


def build_popqa(rng: random.Random) -> tuple[list[dict], dict]:
    ds = load_dataset("akariasai/PopQA", split="test")

    # Deduplicate by normalized question string (keep lowest row order deterministically)
    seen_q: set[str] = set()
    deduped: list[dict] = []
    for row in ds:
        q_norm = " ".join(row["question"].strip().lower().split())
        ans = [a.strip() for a in json.loads(row["possible_answers"]) if a and a.strip()]
        if not ans or q_norm in seen_q:
            continue
        seen_q.add(q_norm)
        deduped.append({
            "orig_id": int(row["id"]),
            "subj": row["subj"],
            "prop": row["prop"],
            "obj": row["obj"],
            "s_pop": int(row["s_pop"]),
            "o_pop": int(row["o_pop"]),
            "question": row["question"].strip(),
            "possible_answers": ans,
        })

    spops = np.array([r["s_pop"] for r in deduped], dtype=np.float64)
    p20 = float(np.percentile(spops, 20))
    p80 = float(np.percentile(spops, 80))

    tail_pool = [r for r in deduped if r["s_pop"] <= p20]
    head_pool = [r for r in deduped if r["s_pop"] >= p80]

    tail_sample = stratified_sample_by_prop(tail_pool, 100, rng)
    head_sample = stratified_sample_by_prop(head_pool, 100, rng)

    records: list[dict] = []
    for stratum, sample in [("head", head_sample), ("tail", tail_sample)]:
        for i, item in enumerate(sample):
            records.append({
                "id": f"popqa_{stratum}_{i:03d}",
                "dataset": "popqa_stratified",
                "stratum": stratum,
                "pilot_split": i < 20,  # 20 Head + 20 Tail = 40 pilot
                **item,
            })

    meta = {
        "raw_total": len(ds),
        "deduped_total": len(deduped),
        "s_pop_p20_threshold": p20,
        "s_pop_p80_threshold": p80,
        "head_pool_size": len(head_pool),
        "tail_pool_size": len(tail_pool),
        "selected_total": len(records),
        "pilot_n": sum(1 for r in records if r["pilot_split"]),
        "head_s_pop_median": float(np.median([r["s_pop"] for r in records if r["stratum"] == "head"])),
        "tail_s_pop_median": float(np.median([r["s_pop"] for r in records if r["stratum"] == "tail"])),
    }
    return records, meta


def build_infinity_chat(rng: random.Random) -> tuple[list[dict], dict]:
    ds = load_dataset("liweijiang/infinite-chats-eval", split="train")
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    pilot_indices = set(indices[:30])

    records: list[dict] = []
    for i, row in enumerate(ds):
        records.append({
            "id": f"infchat_{i:03d}",
            "dataset": "infinity_chat100",
            "query": row["query"].strip(),
            "pilot_split": i in pilot_indices,
        })

    meta = {
        "total": len(records),
        "pilot_n": sum(1 for r in records if r["pilot_split"]),
        "avg_chars": float(np.mean([len(r["query"]) for r in records])),
    }
    return records, meta


def main() -> None:
    rng = random.Random(SEED)
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)

    cov_records, cov_meta = build_coverageqa(rng)
    pop_records, pop_meta = build_popqa(rng)
    inf_records, inf_meta = build_infinity_chat(rng)

    cov_path = FROZEN_DIR / "coverageqa.jsonl"
    pop_path = FROZEN_DIR / "popqa_stratified.jsonl"
    inf_path = FROZEN_DIR / "infinity_chat100.jsonl"

    write_jsonl(cov_path, cov_records)
    write_jsonl(pop_path, pop_records)
    write_jsonl(inf_path, inf_records)

    # Cross-dataset overlap sanity check
    cov_qs = {r["question"].lower() for r in cov_records}
    pop_qs = {r["question"].lower() for r in pop_records}
    inf_qs = {r["query"].lower() for r in inf_records}
    overlaps = {
        "cov_pop": len(cov_qs & pop_qs),
        "cov_inf": len(cov_qs & inf_qs),
        "pop_inf": len(pop_qs & inf_qs),
    }
    assert all(v == 0 for v in overlaps.values()), f"Unexpected question overlap: {overlaps}"

    manifest = {
        "seed": SEED,
        "overlaps": overlaps,
        "datasets": {
            "coverageqa": {**cov_meta, "path": str(cov_path.relative_to(ROOT)), "sha256": sha256_file(cov_path)},
            "popqa_stratified": {**pop_meta, "path": str(pop_path.relative_to(ROOT)), "sha256": sha256_file(pop_path)},
            "infinity_chat100": {**inf_meta, "path": str(inf_path.relative_to(ROOT)), "sha256": sha256_file(inf_path)},
        },
        "totals": {
            "full_questions": cov_meta["total"] + pop_meta["selected_total"] + inf_meta["total"],
            "pilot_questions": cov_meta["pilot_n"] + pop_meta["pilot_n"] + inf_meta["pilot_n"],
        },
    }
    manifest_path = FROZEN_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
