"""Model loading and batched generation wrappers under locked R0 hardware constraints."""

from __future__ import annotations

import gc
from typing import Any

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.trajectory_logger import DreamTrajectoryLogger


def build_bnb_config(protocol: dict[str, Any]) -> BitsAndBytesConfig:
    qcfg = protocol["hardware"]["quantization"]
    compute_dtype = getattr(torch, qcfg["bnb_4bit_compute_dtype"])
    return BitsAndBytesConfig(
        load_in_4bit=bool(qcfg["load_in_4bit"]),
        bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=bool(qcfg["bnb_4bit_use_double_quant"]),
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_model_and_tokenizer(
    model_id: str,
    protocol: dict[str, Any],
    is_diffusion: bool,
    device_index: int = 0,
) -> tuple[Any, Any]:
    """Load 4-bit NF4 + fp16 + sdpa model and left-padded tokenizer."""
    torch_dtype = getattr(torch, protocol["hardware"]["torch_dtype"])
    attn_impl = protocol["hardware"]["attn_implementation"]
    bnb_cfg = build_bnb_config(protocol)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=is_diffusion)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    auto_cls = AutoModel if is_diffusion else AutoModelForCausalLM
    kwargs: dict[str, Any] = {
        "quantization_config": bnb_cfg,
        "torch_dtype": torch_dtype,
        "device_map": {"": device_index},
        "trust_remote_code": is_diffusion,
    }
    if not is_diffusion:
        kwargs["attn_implementation"] = attn_impl

    model = auto_cls.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, tokenizer


def unload_model(model: Any) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


@torch.no_grad()
def generate_dream_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
    steps: int,
    alg: str,
    temperature: float,
    top_p: float,
    alg_temp: float = 0.0,
    eps: float = 1e-3,
    seed: int = 42,
) -> dict[str, Any]:
    """Run Dream diffusion_generate on a left-padded batch with online trajectory logging."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    enc = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)
    prompt_len = input_ids.shape[1]

    mask_token_id = model.config.mask_token_id
    logger = DreamTrajectoryLogger(
        prompt_len=prompt_len,
        total_steps=steps,
        mask_token_id=mask_token_id,
    )

    out = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        steps=steps,
        temperature=temperature,
        top_p=top_p,
        alg=alg,
        alg_temp=alg_temp,
        eps=eps,
        return_dict_in_generate=True,
        output_history=False,
        generation_logits_hook_func=logger.logits_hook,
        generation_tokens_hook_func=logger.tokens_hook,
    )

    gen_ids = out.sequences[:, prompt_len:].detach().cpu()
    texts = [tokenizer.decode(row.tolist(), skip_special_tokens=False) for row in gen_ids]
    traj = logger.summary()

    return {
        "sequences": out.sequences.detach().cpu(),
        "gen_ids": gen_ids,
        "texts": texts,
        "prompt_len": prompt_len,
        "step_mask_entropy": traj["step_mask_entropy"],
        "unmask_steps": traj["unmask_steps"],
        "unmask_probs": traj["unmask_probs"],
    }


@torch.no_grad()
def generate_ar_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int = 42,
) -> dict[str, Any]:
    """Run autoregressive batch generation matching Dream's stochastic decoding settings."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    enc = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)
    prompt_len = input_ids.shape[1]

    do_sample = temperature > 0.0
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    seq = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
    gen_ids = seq[:, prompt_len:].detach().cpu()
    texts = [tokenizer.decode(row.tolist(), skip_special_tokens=False) for row in gen_ids]

    return {
        "sequences": seq.detach().cpu(),
        "gen_ids": gen_ids,
        "texts": texts,
        "prompt_len": prompt_len,
    }
