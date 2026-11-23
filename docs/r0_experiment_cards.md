# R0 Experiment Cards & Feedback Ladder Protocol

**Status:** Locked at R0 prior to running GPU experiments.  
**Target Venue:** ACL 2027 Main Conference (Special Theme: *Homogenization and Knowledge Collapse in LLMs*).  
**Hardware Envelope:** 1× NVIDIA Tesla T4 (16 GB VRAM, `sm_75`), 4-bit NF4 double quantization + `torch.float16` + `sdpa`.

---

## 1. Experiment Cards (One Change Under Test per Card)

### Card FM-1: Early Structural Lock-in & Tail Knowledge Erasure
- **Question:** Does post-training alignment (Base $\to$ Instruct) induce stronger multi-answer coverage loss and tail-knowledge erosion in dLLMs (`Dream-7B`) than in twin AR models (`Qwen2.5-7B`), and is this linked to early boilerplate token lock-in?
- **Hypothesis:** Moving from `Dream-7B-Base` to `Dream-7B-Instruct` increases `EBLR` (in $t \le 0.25T$) and decreases `Recall@K` (CoverageQA) and `THR` (PopQA) more severely than `Qwen2.5-7B` Base $\to$ Instruct, while `Precision@K` remains stable.
- **Baseline:** `Dream-v0-Base-7B` and `Qwen2.5-7B` (Base) under unified few-shot completion (`T=64`, `alg=maskgit_plus`, `temperature=1.0`, `top_p=0.95`).
- **Variant:** `Dream-v0-Instruct-7B` and `Qwen2.5-7B-Instruct` under identical few-shot completion (plus secondary `chat_template` sensitivity run on Instruct).
- **Primary Metric:** $\Delta\text{Recall@K}$ (CoverageQA) & $\text{THR}$ (PopQA), paired with $\Delta\text{EBLR}$ ($t \le 0.25T$).
- **Guardrails:** `Precision@K` must not collapse artificially due to refusal/gibberish; refusal rate logged separately.
- **Data / Split:** Pilot R5 = 30 CoverageQA + 40 PopQA (20 Head / 20 Tail); Full R6 = 145 CoverageQA + 200 PopQA (100 Head / 100 Tail).
- **Seeds:** $K=8$ fixed seeds `[42..49]` in R5; $K=16$ fixed seeds `[42..57]` in R6.
- **Success Criterion:** Paired Wilcoxon / bootstrap 95% CI shows $p < 0.01$ increase in `EBLR` and statistically significant drop in `Recall@K` / `THR`, with negative per-item correlation between $\Delta\text{EBLR}$ and $\Delta\text{Recall@K}$.
- **Exploratory-only:** Absolute values under `chat_template` vs `few-shot` prompt formatting.

### Card FM-2: Generation Order Collapse (The Flexibility Trap)
- **Question:** Does post-training alignment collapse Dream-7B's non-linear unmasking order into rigid left-to-right decoding, reducing semantic diversity?
- **Hypothesis:** Moving from `Dream-7B-Base` to `Dream-7B-Instruct` decreases Unmasking Order Entropy ($H_{order}$) and shifts Left-to-Right correlation ($\rho_{LTR}$) toward $+1.0$, correlating with lower $VS_{0.5}$ on open-ended and multi-answer prompts.
- **Baseline:** `Dream-v0-Base-7B` (`T=64`, `alg=maskgit_plus`, `temperature=1.0`, `top_p=0.95`).
- **Variant:** `Dream-v0-Instruct-7B` under identical decoding config.
- **Primary Metric:** $\Delta H_{order}$ (bits) and $\Delta \rho_{LTR}$ (Spearman rank correlation).
- **Guardrails:** `MTLD` (to separate surface lexical variation from semantic diversity $VS_{0.5}$) and `External PPL`.
- **Data / Split:** Pilot R5 = 30 INFINITY-CHAT100 + 30 CoverageQA; Full R6 = 100 INFINITY-CHAT100 + 145 CoverageQA.
- **Seeds:** $K=8$ (R5) / $K=16$ (R6).
- **Success Criterion:** Significant decrease in $H_{order}$ and positive shift $\Delta \rho_{LTR} > 0$ ($p < 0.01$, non-overlapping 95% bootstrap CI), plus significant positive correlation with $\Delta VS_{0.5}$.
- **Exploratory-only:** Position-wise trajectory visualizations across prompt categories.

### Card FM-3: Bidirectional Hallucination Cascade Under Step Compression
- **Question:** Under inference step compression ($T: 128 \to 16$), does bidirectional conditioning in `Dream-7B-Instruct` amplify early entity errors into confident hallucinations ($\Delta C_{wrong} > 0$)?
- **Hypothesis:** Compressing $T$ from 128 to 16 causes a steeper `Hallucination Rate` ($1 - \text{Precision@K}$) increase on `Dream-7B-Instruct` than compute-matched AR baselines, with early-unmasked wrong entities ($t_{unmask} \le 0.25T$) gaining probability by step $T$ under leave-entity-masked re-evaluation ($\Delta C_{wrong} > 0$).
- **Baseline:** `Dream-7B-Instruct` at $T=128$ (and `Qwen2.5-7B-Instruct` single-pass AR reference).
- **Variant:** `Dream-7B-Instruct` at $T \in \{64, 32, 16\}$.
- **Primary Metric:** Step-compression hallucination slope and leave-entity-masked $\Delta C_{wrong}$ (contrasted against $\Delta C_{right}$ control group).
- **Guardrails:** Refusal rate and `External PPL` (verifying hallucinations are fluent, not random garbage).
- **Data / Split:** Pilot R5 = 40 PopQA (20 Head / 20 Tail); Full R6 = 200 PopQA + 145 CoverageQA.
- **Seeds:** $K=8$ fixed seeds.
- **Success Criterion:** Positive slope of Hallucination Rate as $T$ decreases, and median $\Delta C_{wrong} > 0$ ($p < 0.01$) on early-unmasked incorrect entities.
- **Exploratory-only:** Layer-level or token-span attention explanations.

### Card FM-4: Remasking Strategy Phase Collapse
- **Question:** Do static remasking strategies (`origin`, `maskgit_plus`, `topk_margin`, `entropy`) exhibit an inescapable Pareto tradeoff between knowledge/semantic diversity and fluency/factuality?
- **Hypothesis:** Confidence-greedy static strategies (`maskgit_plus`, `topk_margin`, `entropy`) collapse `Recall@K` and $VS_{0.5}$ relative to `origin`, while `origin` severely degrades `External PPL` and `Precision@K` — no single static rule dominates both axes.
- **Baseline:** `alg=maskgit_plus` on `Dream-7B-Instruct`.
- **Variant:** `alg` $\in \{\text{origin}, \text{topk\_margin}, \text{entropy}\}$ with identical `temperature=1.0, top_p=0.95, T=64`.
- **Primary Metric:** Two-axis Pareto dominance check: `(Recall@K, VS_0.5)` vs. `(Precision@K, External PPL)`.
- **Guardrails:** Fixed seeds and identical prompt/template across all 4 `alg` settings.
- **Data / Split:** Pilot R5 = 30 mixed (10 CovQA + 10 PopQA-Tail + 10 INFINITY-CHAT); Full R6 = 150 mixed (50 + 50 + 50).
- **Seeds:** $K=8$ (R5) / $K=16$ (R6).
- **Success Criterion:** Statistically significant tradeoff gap $\Delta_{alg}$ on both diversity (`origin` > `maskgit_plus`, $p<0.01$) and fluency/accuracy (`maskgit_plus` > `origin`, $p<0.01$).
- **Exploratory-only:** Thresholds for switching phases in downstream PA-AMR mitigation.

---

## 2. Critical Technical Findings Recorded at R0–R1 Inspection

1. **Dream-7B Logits Shifting Convention:** In `generation_utils.py`, `DreamGenerationMixin._sample` already applies `logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)` **before** calling `generation_logits_hook_func(i, x, logits)`. Therefore:
   - Inside `generation_logits_hook_func`, `logits[:, j]` directly corresponds to `x[:, j]`.
   - Outside `diffusion_generate` (e.g. direct `model(x)` forward passes for leave-entity-masked $\Delta C_{wrong}$), the caller **must** apply the same 1-position right shift.
2. **Zero-Logit-Tensor Online Extraction:** Combining `generation_logits_hook_func(i, x_before, logits)` and `generation_tokens_hook_func(i, x_after, logits)` captures $H_t$, $\tau_{k,j}$, and $P^{(t_{unmask})}(x_j)$ online without storing full `[steps, B, L, V]` tensors or setting `output_history=True`.
3. **Left-Padding Requirement for Batched Generation:** Dream-7B and Qwen2.5 tokenizers default to `padding_side='right'`. In `DreamGenerationMixin._sample`, `F.pad(input_ids, ..., value=mask_token_id)` appends `[MASK]` on the right; right-padding prompts would insert `<|endoftext|>` between prompt and `[MASK]`. All batched inference must set `tokenizer.padding_side = 'left'`.
4. **Special Token IDs:** For Dream-7B, `<|mask|>=151666`, `<|endoftext|>=151643`, `<|im_end|>=151645`. Output post-processing must truncate at both `<|endoftext|>` and `<|im_end|>`.
5. **Dataset Audit Notes:**
   - `CoverageQA.json` (from `jwong8314/simplestrat`) contains **145** questions across 6 categories (`country`: 74, `physic`: 31, `athlete`: 18, `chemical`: 11, `instrument`: 6, `park`: 5); `country + physic` = **105** questions. We preserve all 145 in `coverageqa.jsonl` with an `in_105_core` flag.
   - `akariasai/PopQA` (`test`, 14,267 rows) has 1,199 duplicate question strings; `prepare_datasets.py` deduplicates by normalized question before percentile stratification (`p20=179`, `p80=9747`).

---

## 3. R0–R7 Verification Ladder & Gates

| Rung | Scope | Gate to Promote |
|------|-------|-----------------|
| **R0** | Lock Experiment Cards, config, frozen datasets (`445` full / `100` pilot), metric unit tests | All 3 frozen JSONLs + `manifest.json` generated, all unit tests pass |
| **R1** | Environment, packages, Dream/Qwen/BGE-M3 configs, synthetic `DreamModel` hook test on T4 | `check_r1_env.py` + `pytest` pass with 0 errors |
| **R2** | 1-batch real 4-bit NF4+fp16+sdpa forward/generate on Dream-7B & Qwen2.5-7B | Peak VRAM $\le 12.5$ GB, zero `NaN/Inf`, throughput logged |
| **R3** | Qualitative inspection on 5 real questions ($K=4$) | Reproducible seeds, `origin` vs `maskgit_plus` shifts $H_{order}$, leave-entity-masked $\Delta C_{wrong}$ verified |
| **R4** | 10-question end-to-end smoke run with checkpoint/resume & full 3-tier metric + plot output | Clean resume after interrupt, all 3 metric tiers + plots emitted |
| **R5** | **Pilot Study** (100 stratified questions, $K=8$, primary Dream-7B vs Qwen2.5-7B across FM-1..4) | Rank FMs by effect size & 95% bootstrap CI; prune weak FMs, select 1–2 core FMs |
| **R6** | Full Study (all frozen questions, $K=16$ + LLaDA-8B/Llama-3.1-8B + fp16 sensitivity subset) | All planned runs complete with verified integrity |
| **R7** | Result review & PA-AMR mitigation design | Decision memo + main-paper tables/figures |
