# R5 Pilot Study — Empirical Failure Mode Decision Memo

## FM-1: Early Structural Lock-in & Tail Knowledge Erasure
- Dream CoverageQA Recall@8 shift (Base -> Inst): -0.0091 (p=0.4905)
- Qwen CoverageQA Recall@8 shift (Base -> Inst): -0.0352 (p=0.0000)
- Dream EBLR shift (Base -> Inst): -0.2417 (p=0.0000)
- Spearman(ΔEBLR, ΔRecall@8): rho=-0.1104 (p=0.5615)
- PopQA THR: Dream=0.963 vs Qwen=0.861

## FM-2: Generation Order Collapse (INFINITY-CHAT100)
- Dream H_order shift (Base -> Inst): +0.0000 bits (p=1.0000)
- Dream rho_LTR shift (Base -> Inst): +0.0000 (p=0.0000)
- Vendi Score (q=0.5): Dream Base=3.917 -> Inst=3.501 | Qwen Base=3.445 -> Inst=2.907

## FM-3: Bidirectional Hallucination Cascade (PopQA Step Compression)
- T64: Halluc=0.669 | ΔC_wrong=-0.0562 (n=214) | ΔC_right=-0.0059 (n=106)
- T32: Halluc=0.675 | ΔC_wrong=-0.0189 (n=216) | ΔC_right=-0.1232 (n=104)
- T16: Halluc=0.744 | ΔC_wrong=+0.0159 (n=238) | ΔC_right=-0.0671 (n=82)

## FM-4: Static Remasking Pareto Tradeoff
- entropy: Recall=0.109, Prec=0.680, VS_0.5=4.238, Clusters=19.60, ActiveLen=79.1, PPL=13.88
- origin: Recall=0.134, Prec=0.745, VS_0.5=4.326, Clusters=13.30, ActiveLen=45.5, PPL=28.20
- maskgit_plus: Recall=0.089, Prec=0.712, VS_0.5=2.200, Clusters=2.50, ActiveLen=12.2, PPL=578.15
- topk_margin: Recall=0.084, Prec=0.628, VS_0.5=2.882, Clusters=3.40, ActiveLen=13.2, PPL=585.75
