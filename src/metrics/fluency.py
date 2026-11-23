"""External Perplexity (PPL) fluency guardrail using Qwen2.5-7B-Instruct."""

from __future__ import annotations

import gc
import math
from typing import Any, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model_loader import build_bnb_config


class ExternalPPLScorer:
    """Computes response-level Perplexity using an independent AR reference model."""

    def __init__(self, model_id: str, protocol: dict[str, Any], device_index: int = 0) -> None:
        self.model_id = model_id
        bnb_cfg = build_bnb_config(protocol)
        torch_dtype = getattr(torch, protocol["hardware"]["torch_dtype"])
        attn_impl = protocol["hardware"]["attn_implementation"]

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_cfg,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
            device_map={"": device_index},
        )
        self.model.eval()

    @torch.no_grad()
    def score_texts(self, texts: Sequence[str], max_length: int = 256) -> list[float]:
        """Return per-response perplexity (NaN for texts shorter than 2 tokens)."""
        scores: list[float] = []
        for txt in texts:
            if not txt or not txt.strip():
                scores.append(float("nan"))
                continue
            enc = self.tokenizer(
                txt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(self.model.device)
            input_ids = enc["input_ids"]
            if input_ids.shape[1] < 2:
                scores.append(float("nan"))
                continue
            out = self.model(input_ids=input_ids, labels=input_ids)
            loss_val = float(out.loss.detach().cpu().item())
            ppl = math.exp(min(loss_val, 20.0))
            scores.append(ppl)
        return scores

    def unload(self) -> None:
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
