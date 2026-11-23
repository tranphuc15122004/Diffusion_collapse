"""Online diffusion trajectory logger for Dream-7B without retaining full logit tensors."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class DreamTrajectoryLogger:
    """Captures step-wise mask entropy, unmasking order, and token probabilities at unmasking.

    Designed for `DreamGenerationMixin.diffusion_generate(..., generation_logits_hook_func=...,
    generation_tokens_hook_func=...)`. Note that Dream shifts logits BEFORE calling
    `generation_logits_hook_func(step, x, logits)`, so `logits[:, j]` aligns with `x[:, j]`.
    """

    def __init__(self, prompt_len: int, total_steps: int, mask_token_id: int) -> None:
        self.prompt_len = prompt_len
        self.total_steps = total_steps
        self.mask_token_id = mask_token_id

        self.step_mask_entropy: list[torch.Tensor] = []  # each [B]
        self.unmask_steps: torch.Tensor | None = None    # [B, L_gen] int32 (1..T)
        self.unmask_probs: torch.Tensor | None = None    # [B, L_gen] float32

        self._prev_gen_mask: torch.Tensor | None = None
        self._step_probs_cache: torch.Tensor | None = None

    def logits_hook(self, step: int, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        gen_x = x[:, self.prompt_len :]
        gen_logits = logits[:, self.prompt_len :, :]
        gen_mask = gen_x == self.mask_token_id  # [B, L_gen]

        if self.unmask_steps is None:
            bsz, gen_len = gen_x.shape
            self.unmask_steps = torch.zeros((bsz, gen_len), dtype=torch.int32, device="cpu")
            self.unmask_probs = torch.zeros((bsz, gen_len), dtype=torch.float32, device="cpu")

        # Compute Shannon entropy (bits) over masked generated positions in float32
        probs = F.softmax(gen_logits.float(), dim=-1)
        log_probs = torch.log2(probs.clamp_min(1e-12))
        tok_entropy = -(probs * log_probs).sum(dim=-1)  # [B, L_gen]

        mask_f = gen_mask.float()
        denom = mask_f.sum(dim=-1).clamp_min(1.0)
        mean_ent = (tok_entropy * mask_f).sum(dim=-1) / denom
        self.step_mask_entropy.append(mean_ent.detach().cpu())

        self._prev_gen_mask = gen_mask
        # Keep only normalized probs for the immediately following tokens_hook of the same step
        self._step_probs_cache = probs
        return logits

    def tokens_hook(self, step: int | None, x: torch.Tensor, logits: torch.Tensor | None) -> torch.Tensor:
        if step is None or self._prev_gen_mask is None or self._step_probs_cache is None:
            return x

        gen_x = x[:, self.prompt_len :]
        newly_unmasked = self._prev_gen_mask & (gen_x != self.mask_token_id)

        if torch.any(newly_unmasked):
            chosen_probs = self._step_probs_cache.gather(-1, gen_x.unsqueeze(-1)).squeeze(-1)
            new_cpu = newly_unmasked.cpu()
            assert self.unmask_steps is not None and self.unmask_probs is not None
            self.unmask_steps[new_cpu] = int(step) + 1  # 1-indexed step in [1..T]
            self.unmask_probs[new_cpu] = chosen_probs.detach().cpu()[new_cpu]

        # Release references immediately so GPU memory never accumulates across steps
        self._prev_gen_mask = None
        self._step_probs_cache = None
        return x

    def summary(self) -> dict[str, torch.Tensor]:
        ent_tensor = (
            torch.stack(self.step_mask_entropy, dim=1)
            if self.step_mask_entropy
            else torch.empty((0, 0), dtype=torch.float32)
        )
        return {
            "step_mask_entropy": ent_tensor,
            "unmask_steps": self.unmask_steps if self.unmask_steps is not None else torch.empty((0, 0), dtype=torch.int32),
            "unmask_probs": self.unmask_probs if self.unmask_probs is not None else torch.empty((0, 0), dtype=torch.float32),
        }
