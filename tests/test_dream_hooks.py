"""R1 synthetic DreamModel test verifying shifted logits, online hooks, left-padding, and leave-entity masking."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from transformers.dynamic_module_utils import get_class_from_dynamic_module

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics.trajectory import compute_leave_entity_masked_prob, compute_unmask_order_metrics
from src.trajectory_logger import DreamTrajectoryLogger

DREAM_REPO = "Dream-org/Dream-v0-Instruct-7B"


@pytest.fixture(scope="module")
def tiny_dream():
    DreamConfig = get_class_from_dynamic_module("configuration_dream.DreamConfig", DREAM_REPO)
    DreamModel = get_class_from_dynamic_module("modeling_dream.DreamModel", DREAM_REPO)

    cfg = DreamConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        mask_token_id=255,
        bos_token_id=253,
        eos_token_id=254,
        pad_token_id=254,
        _attn_implementation="sdpa",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = DreamModel(cfg).to(device=device, dtype=dtype).eval()
    return model, cfg, device


@pytest.mark.parametrize("alg", ["maskgit_plus", "origin"])
def test_dream_online_hooks_and_left_padding(tiny_dream, alg: str):
    model, cfg, device = tiny_dream
    torch.manual_seed(42)

    # Simulate left-padded batch of 2 prompts of different lengths (pad=254 on left)
    input_ids = torch.tensor(
        [
            [254, 254, 10, 11, 12, 13],
            [20, 21, 22, 23, 24, 25],
        ],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
        device=device,
    )

    steps = 8
    max_new = 12
    logger = DreamTrajectoryLogger(
        prompt_len=input_ids.shape[1],
        total_steps=steps,
        mask_token_id=cfg.mask_token_id,
    )

    out = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new,
        steps=steps,
        temperature=1.0,
        top_p=0.95,
        alg=alg,
        alg_temp=0.0,
        return_dict_in_generate=True,
        output_history=False,
        generation_logits_hook_func=logger.logits_hook,
        generation_tokens_hook_func=logger.tokens_hook,
    )

    summary = logger.summary()
    ent = summary["step_mask_entropy"]
    ustep = summary["unmask_steps"]
    uprob = summary["unmask_probs"]

    assert out.sequences.shape == (2, 6 + max_new)
    assert ent.shape == (2, steps)
    assert torch.all(torch.isfinite(ent)) and torch.all(ent >= 0.0)
    assert ustep.shape == (2, max_new)
    assert torch.all((ustep >= 1) & (ustep <= steps))
    assert uprob.shape == (2, max_new)
    assert torch.all(torch.isfinite(uprob)) and torch.all((uprob > 0.0) & (uprob <= 1.0))
    # No remaining [MASK] tokens at end of diffusion
    assert torch.all(out.sequences[:, 6:] != cfg.mask_token_id)

    order = compute_unmask_order_metrics(ustep.numpy())
    assert -1.0 <= order["rho_ltr"] <= 1.0

    # Verify leave-entity-masked Delta C_wrong helper runs and returns valid prob
    p_final = compute_leave_entity_masked_prob(
        model,
        out.sequences[:1],
        entity_span=(8, 11),
        mask_token_id=cfg.mask_token_id,
        shift_logits=True,
    )
    assert 0.0 < p_final <= 1.0
