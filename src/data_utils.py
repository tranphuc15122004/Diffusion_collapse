"""Dataset loading and unified prompt construction utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_protocol(path: Path | str | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path is not None else ROOT / "configs" / "r0_protocol.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_frozen_dataset(name: str, pilot_only: bool = False, core_105_only: bool = False) -> list[dict]:
    path_map = {
        "coverageqa": ROOT / "data" / "frozen" / "coverageqa.jsonl",
        "popqa": ROOT / "data" / "frozen" / "popqa_stratified.jsonl",
        "popqa_stratified": ROOT / "data" / "frozen" / "popqa_stratified.jsonl",
        "infinity_chat100": ROOT / "data" / "frozen" / "infinity_chat100.jsonl",
    }
    if name not in path_map:
        raise ValueError(f"Unknown dataset '{name}'. Expected one of {list(path_map)}")
    records: list[dict] = []
    with open(path_map[name], "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if pilot_only and not rec.get("pilot_split", False):
                continue
            if core_105_only and name == "coverageqa" and not rec.get("in_105_core", True):
                continue
            records.append(rec)
    return records


def format_prompt(
    record: dict,
    protocol: dict[str, Any],
    mode: str = "unified_fewshot",
    tokenizer: Any = None,
) -> str:
    """Build deterministic prompt string for QA or open-ended generation.

    Args:
        record: Frozen dataset item.
        protocol: Loaded r0_protocol.yaml dict.
        mode: 'unified_fewshot' (shared across Base and Instruct) or 'chat_template' (Instruct only).
        tokenizer: Required when mode == 'chat_template'.
    """
    dataset = record["dataset"]
    text = record["question"] if dataset in ("coverageqa", "popqa_stratified") else record["query"]

    if mode == "unified_fewshot":
        if dataset in ("coverageqa", "popqa_stratified"):
            prefix = protocol["prompts"]["qa_fewshot_prefix"].rstrip()
            return f"{prefix}\n\nQuestion: {text}\nAnswer:"
        return protocol["prompts"]["open_fewshot_prefix"].format(query=text).rstrip()

    if mode == "chat_template":
        if tokenizer is None or getattr(tokenizer, "chat_template", None) is None:
            raise ValueError("tokenizer with chat_template is required for mode='chat_template'")
        messages = [{"role": "user", "content": text}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    raise ValueError(f"Unknown prompt mode: {mode}")


def strip_generation_text(text: str, dataset: str) -> str:
    """Trim special markers and (for few-shot QA) trailing extra turns."""
    for marker in ("<|endoftext|>", "<|im_end|>", "<|eot_id|>"):
        if marker in text:
            text = text.split(marker, 1)[0]
    if dataset in ("coverageqa", "popqa_stratified"):
        for stop in ("\nQuestion:", "\n\n"):
            if stop in text:
                text = text.split(stop, 1)[0]
        # Take first non-empty line for single-answer QA completion
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return lines[0] if lines else ""
    else:
        if "\nRequest:" in text:
            text = text.split("\nRequest:", 1)[0]
        return text.strip()


def classify_open_query(query: str) -> str:
    """Lightweight heuristic task classifier for INFINITY-CHAT100 open-ended queries."""
    q = query.lower()
    if any(k in q for k in ("story", "poem", "song", "verse", "joke", "letter", "dialogue", "scene", "write me", "fiction")):
        return "creative_writing"
    if any(k in q for k in ("ideas", "ways", "brainstorm", "suggest", "recommend", "list", "strategies", "alternatives", "options")):
        return "brainstorming_ideation"
    if any(k in q for k in ("how", "why", "explain", "compare", "benefit", "fit into", "analyze", "difference", "impact")):
        return "explanation_analysis"
    return "other"
