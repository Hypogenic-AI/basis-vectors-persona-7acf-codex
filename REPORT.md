# Basis Vectors in Persona Space

## 1. Executive Summary
This study tested whether persona representations in a language model can be decomposed into shared basis vectors and whether PCA recovers primary components in residual-stream geometry.

Using a real model (`Qwen/Qwen2.5-0.5B`) and local persona datasets, we found strong low-dimensional structure: top-10 PCs explained **57.14%** of variance, and top-20 explained **70.06%**. Subspace stability across bootstrap splits was high (mean overlap **0.905 ± 0.016**).

Causal steering evidence was mixed: one directional test (`PC-` vs base) was significant (Wilcoxon **p=0.0228**, effect size **d=-0.309**), while `PC+` improvements over baseline/random were not statistically significant. External alignment with Big-5 labels was weak and not significant after FDR correction.

## 2. Goal
### Hypothesis
Persona representations in residual streams are decomposable into reusable components, and PCA over persona vectors should reveal primary axes that are stable and behaviorally meaningful.

### Why Important
If true, persona steering can move from many brittle handcrafted vectors toward compact interpretable bases, improving controllability, transfer, and mechanistic understanding.

### Problem Solved
This work provides an empirical pipeline to test whether persona vectors exhibit low-rank geometry and whether top components support causal interventions.

## 3. Data Construction
### Dataset Description
- `proj-persona/PersonaHub` (`datasets/personahub_persona`): persona text corpus, used to build vector bank.
- `AlekseyKorshuk/persona-chat` (`datasets/persona_chat`): dialogue prompts for steering evaluation.
- `holistic-ai/Personality_mypersonality` (`datasets/mypersonality`): text + Big-5 labels for external trait validation.

### Example Samples
| Dataset | Example |
|---|---|
| PersonaHub | "A young inmate inspired by the activist's story and is striving to reform himself while inside prison" |
| PersonaHub | "A small business owner in New York with little financial knowledge" |
| PersonaChat prompt | `User: hello what are you up to this evening ?\nAssistant:` |

### Data Quality
From `results/data_quality.json`:
- Missing values: **0%** in checked subsets.
- Duplicates (PersonaHub checked slice): **0**.
- Persona length stats: mean **13.80**, std **4.20**, min **1**, max **51**, outliers (>3σ) **10**.
- myPersonality label parse failures: **0/500** checked.

### Preprocessing Steps
1. Strip and normalize whitespace in all texts.
2. Sample persona rows uniformly without replacement (`n=800`).
3. Construct persona prompts: `System: You are defined by this persona: ...`.
4. Parse myPersonality JSON answers into binary Big-5 labels (`y=1`, `n=0`).
5. Build PersonaChat evaluation prompts from validation histories.

### Train/Val/Test Splits
- Persona geometry: sampled from PersonaHub train split.
- Steering evaluation: PersonaChat validation prompts.
- Trait alignment: myPersonality train sampled subset.

## 4. Experiment Description
### Methodology
#### High-Level Approach
1. Extract hidden-state vectors for persona prompts.
2. Fit PCA on centered vectors.
3. Quantify explained variance and subspace stability.
4. Intervene during generation by adding/subtracting PC directions at a target residual layer.
5. Compare against random-direction baseline.
6. Test correlations between PC scores and external Big-5 traits.

#### Why This Method
- Directly addresses the user hypothesis: persona vectors -> PCA -> primary components.
- Uses causal intervention (not only descriptive geometry).
- Uses independent trait labels for external validation.

### Implementation Details
#### Tools and Libraries
- Python 3.12.8
- torch 2.5.1+cu124
- transformers 4.51.3
- datasets 4.6.1
- scikit-learn 1.8.0
- scipy 1.17.1
- statsmodels 0.14.6
- matplotlib 3.10.8 / seaborn 0.13.2

#### Model
- `Qwen/Qwen2.5-0.5B` (real pretrained LLM)
- Hidden dim: 896
- Layers: 24
- Main extraction layer: 23 (last), with cross-layer check vs layer 12

#### Hyperparameters
| Parameter | Value | Selection Method |
|---|---:|---|
| persona_samples | 800 | runtime/coverage tradeoff |
| steering_eval_prompts | 48 (45 usable) | held-out validation prompts |
| trait_samples | 800 | runtime/coverage tradeoff |
| max_length_extract | 128 | fixed truncation |
| max_new_tokens | 24 | speed + sufficient lexical signal |
| batch_size | 64 | GPU memory-guided (24GB class) |
| pca_components | 20 | captures dominant variance |
| steering_alpha | 8.0 | pilot-scaled intervention |
| seed | 42 | reproducibility |

### Experimental Protocol
#### Reproducibility
- Runs averaged: single deterministic run + deterministic rerun check.
- Seed: 42 for Python/NumPy/PyTorch/CUDA.
- Hardware: 2x NVIDIA GeForce RTX 3090 (24GB each).
- Mixed precision: enabled on CUDA.
- Runtime per full run: ~22s.

#### Evaluation Metrics
- Geometry: EVR top-k, PC-wise EVR.
- Stability: subspace overlap (cosine of principal angles) across bootstrap splits.
- Causal steering: lexical score (`pos_hits - neg_hits`) under base/PC+/PC-/random+.
- External validity: Pearson r between PC1-5 projections and Big-5 labels (+ BH-FDR correction).

### Raw Results
#### Geometry
| Metric | Value |
|---|---:|
| EVR top-10 | **0.5714** |
| EVR top-20 | **0.7006** |
| PC1 EVR | 0.1515 |
| PC2 EVR | 0.0872 |
| Split subspace overlap (mean ± sd) | **0.9050 ± 0.0163** |
| Random 10D basis variance ratio (mean) | 0.0113 |
| Cross-layer overlap (layer12 vs layer23, top10) | 0.2802 |

#### Steering
Score means by condition:
- Base: **0.0417**
- PC+: **0.1250**
- PC-: **-0.0417**
- Random+: **0.1042**

Statistical tests (Wilcoxon, directional):
- PC+ vs Base: mean Δ=0.0889, 95% CI [0.0000, 0.2667], p=0.1587, d=0.149
- PC+ vs Random+: mean Δ=0.0222, 95% CI [0.0000, 0.0667], p=0.1587, d=0.149
- PC- vs Base: mean Δ=-0.0889, 95% CI [-0.1778, -0.0222], **p=0.0228**, d=-0.309

#### Trait Alignment
- Significant correlations after BH-FDR: **0**
- Maximum absolute raw correlation: **|r|=0.0918** (small)

### Output Locations
- Metrics JSON: `results/metrics.json`
- Quality report: `results/data_quality.json`
- Steering rows: `results/steering_outputs.csv`
- Trait correlations: `results/trait_alignment_rows.json`
- Plots:
  - `results/plots/pca_scree.png`
  - `results/plots/steering_scores_boxplot.png`
  - `results/plots/trait_pc_heatmap.png`

## 5. Result Analysis
### Key Findings
1. **Strong low-rank persona geometry**: top PCs explain large variance quickly.
2. **High in-layer subspace stability**: bootstrap overlap ~0.905 supports reproducible component structure.
3. **Causal steering is asymmetric and weak-to-moderate**: negative-direction steering had significant effect; positive direction did not beat baseline/random significantly.
4. **Limited external trait transfer**: no corrected-significant Big-5 alignment.

### Hypothesis Testing
- H1 (low-dimensional structure): **supported**.
- H2 (causal usefulness): **partially supported** (one significant directional result, small effect).
- H3 (external trait alignment): **not supported** in this setup.
- H4 (stability): **supported within layer splits**, but cross-layer overlap was low-moderate (0.280), suggesting layer-specific bases.

### Comparison to Baselines
- PCA geometry vastly exceeded random basis variance capture.
- PC+ did not significantly outperform random+ in lexical steering score.
- PC- produced a significant directional effect relative to base.

### Surprises and Insights
- Strong geometric structure did not automatically translate to strong semantic steering gains.
- Cross-layer component mismatch suggests persona components are not fully layer-invariant.

### Error Analysis
Observed failure patterns in generated outputs:
- Generic fallback completions reduce lexical sensitivity.
- Some prompts reproduce user text rather than expanding persona content.
- Lexical metric can miss semantic persona shifts that do not use lexicon tokens.

### Limitations
- Single-model study (`Qwen2.5-0.5B`) with modest sample sizes.
- Lexicon-based steering metric is coarse and may under-detect semantics.
- No human evaluation of persona faithfulness.
- No spherical steering or SAE decomposition baseline in this run.

## 6. Conclusions
Persona vectors from diverse persona documents show clear low-dimensional organization in residual representations, and PCA reliably recovers primary geometric components. However, causal persona control from those components is weaker than geometry alone suggests, with only partial steering significance and limited transfer to external Big-5 labels.

Practical implication: PCA basis discovery is promising for analysis and compression of persona vectors, but stronger intervention objectives and richer evaluation metrics are needed before treating top PCs as robust persona control knobs.

Confidence: **moderate** for geometric claims, **low-to-moderate** for causal persona-control claims.

## 7. Next Steps
### Immediate Follow-ups
1. Replace lexical score with classifier-based persona fidelity and OOC metrics.
2. Run multi-model replication (`Qwen2.5-1.5B`, `Llama-3.1-8B` if resources permit).
3. Add spherical steering and raw-centroid steering baselines.
4. Expand alpha sweep and per-layer sweep for intervention sensitivity.

### Alternative Approaches
- Sparse autoencoder features as nonlinear persona basis.
- Canonical correlation between persona and trait datasets before intervention.

### Broader Extensions
- Cross-lingual persona basis transfer.
- Safety-focused decomposition (toxicity/helpfulness/persona entanglement).

### Open Questions
- Which PCs are causal vs merely descriptive?
- Can shared persona basis generalize across model families?
- How much of persona is nonlinear and missed by PCA?

## References
- Shai et al. (2024). *Transformers represent belief state geometry in their residual stream*.
- Cao et al. (2024). *Personalized Steering of LLMs via Bi-directional Preference Optimization*.
- Lawson et al. (2024). *Residual Stream Analysis with Multi-Layer SAEs*.
- Braun (2026). *Understanding Unreliability of Steering Vectors in Language Models*.
- You et al. (2026). *Spherical Steering*.
- Zhang et al. (2018). *Personalizing Dialogue Agents (PersonaChat)*.
