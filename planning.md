# Planning: Basis Vectors in Persona Space

## Motivation & Novelty Assessment

### Why This Research Matters
Persona control is central to safe, useful, and personalized LLM deployment, but current steering practice is mostly ad hoc and prompt-specific. If persona behavior decomposes into reusable latent components, we can build more robust, interpretable, and transferable control methods. This benefits alignment, product personalization, and mechanistic interpretability by turning many fragile steering vectors into a smaller, structured basis.

### Gap in Existing Work
Recent work shows linear residual-stream geometry and activation steering effectiveness, but there is limited direct evaluation of persona-specific basis recovery via PCA over a large, diverse persona vector bank. Existing papers emphasize individual steering directions or reliability analysis, not a systematic decomposition into primary shared persona components plus validation across datasets and prompts. Cross-layer consistency of persona components also remains underexplored.

### Our Novel Contribution
We test a direct pipeline: construct many persona vectors from diverse persona documents, perform PCA in residual space, and evaluate whether top PCs are (a) geometrically stable, (b) behaviorally meaningful under intervention, and (c) aligned with independent personality labels. We also compare against random orthonormal bases and raw per-persona vectors to isolate whether PCA gives genuinely better structure.

### Experiment Justification
- Experiment 1: Persona Vector Bank + PCA Geometry. Needed to test whether strong low-dimensional structure exists and quantify explained variance/stability.
- Experiment 2: Steering with PC Basis vs Baselines. Needed to test whether recovered components are causally meaningful, not just descriptive.
- Experiment 3: External Trait Alignment + Robustness. Needed to test generalization and rule out dataset-specific artifacts.

## Research Question
Can persona representations in a transformer residual stream be decomposed into shared basis directions, such that PCA over persona vectors reveals primary components that are stable and useful for causal persona steering?

## Background and Motivation
Prior literature and resources in this workspace indicate that steering vectors often work linearly but can be unreliable across prompts and domains. We leverage PersonaHub for broad persona diversity, Persona-Chat for behavioral testing, and myPersonality for trait-linked external validation. The core practical question is whether a compact PCA basis can replace large sets of task/persona-specific vectors while preserving controllability.

## Hypothesis Decomposition
- H1 (Geometry): Persona vectors occupy a low-dimensional subspace; top-k PCs explain substantially more variance than random controls.
- H2 (Causality): Intervening along top PCs changes persona-relevant generation metrics more than random directions and comparably or better than single raw persona vectors.
- H3 (Transfer): PC scores correlate with independent personality-trait labels (myPersonality), indicating shared components beyond one dataset.
- H4 (Stability): PC directions are consistent across data subsamples, seeds, and nearby layers.

Independent variables:
- Intervention direction type: top PC / raw persona vector / random orthonormal direction / no intervention.
- Layer: selected residual-stream layers.
- Intervention strength alpha.
- Prompt set and random seed.

Dependent variables:
- Explained variance ratio (EVR) and cumulative EVR.
- Steering effect size on persona classifier score / lexical persona adherence.
- Utility retention metric on neutral completion quality proxy.
- Correlation between PC projections and Big-5 labels.
- Subspace stability metrics (principal angle overlap, cosine similarity of matched PCs).

Success criteria:
- Top 10-20 PCs explain clearly elevated variance (target: >35% on sampled bank, model-dependent).
- PC-steering outperforms random directions with statistically significant effect and non-trivial effect size.
- At least one interpretable PC shows significant association with external trait dimensions after multiple-testing correction.

Alternative explanations considered:
- PCs capture style length/frequency artifacts rather than persona semantics.
- Effects are prompt-template specific.
- Correlations arise from dataset leakage/domain overlap.

## Proposed Methodology

### Approach
Use TransformerLens to extract residual activations from a real open model under persona-conditioned text, derive per-persona vectors (persona prompt minus neutral prompt activations), fit PCA on standardized vectors, and evaluate both geometric structure and intervention effects.

### Experimental Steps
1. Environment + reproducibility setup (seeds, versions, GPU logging): ensures repeatability.
2. Data loading and validation for PersonaHub, Persona-Chat, myPersonality: ensures schema correctness and representative sampling.
3. Build persona prompt pairs (persona vs neutral paraphrase/control): isolates persona signal.
4. Activation extraction at selected layers and token positions: produces persona vector bank.
5. PCA fit and stability analysis across bootstrap splits: tests H1 and H4.
6. Intervention experiments on held-out prompts with top PCs and baselines: tests H2.
7. Trait alignment on myPersonality via projection/regression: tests H3.
8. Statistical analysis and visualization with corrected significance tests.

### Baselines
- No intervention (base model).
- Random orthonormal directions (same norm as PCs).
- Raw persona centroid vectors (non-PCA).
- Prompt-only persona instruction baseline.

### Evaluation Metrics
- Cumulative explained variance (PCA).
- Subspace similarity (principal angles, canonical correlations).
- Persona steering score change (classifier/proxy score delta).
- Utility retention (perplexity-like proxy or neutral task score delta).
- Pearson/Spearman correlation of PC coordinates with Big-5 labels.
- Out-of-character proxy rate on Persona-Chat held-out prompts.

### Statistical Analysis Plan
Preregistered tests:
- EVR comparison vs random basis: permutation test.
- Steering effects across direction types: repeated-measures ANOVA or Friedman (assumption-dependent), post-hoc Holm correction.
- Pairwise effect size: Cohen's d (or Cliff's delta if non-normal).
- Trait alignment: multiple linear regression and permutation correlation with Benjamini-Hochberg FDR correction.
- Significance threshold alpha = 0.05 (two-sided), with 95% bootstrap CIs.

Assumption checks:
- Shapiro-Wilk on paired deltas, Levene where relevant.
- If violated, switch to non-parametric equivalents.

## Expected Outcomes
Evidence for hypothesis would be:
- Strong low-rank persona geometry (high early EVR).
- Top PCs inducing consistent persona-relevant shifts with smaller side effects than random directions.
- Non-zero, corrected-significant trait associations for some PCs.

Refuting evidence would be:
- Flat EVR spectrum near random.
- No causal steering gain over random directions.
- No stable cross-dataset trait association.

## Timeline and Milestones
- M1 (20 min): finalize plan and environment checks.
- M2 (30 min): data validation + prompt pair construction.
- M3 (45 min): activation extraction pipeline and PCA.
- M4 (45 min): steering intervention experiments.
- M5 (30 min): statistical analysis and plots.
- M6 (30 min): REPORT.md + README.md + validation rerun.

Includes ~25% buffer for debugging and reruns.

## Potential Challenges
- Activation extraction runtime on large samples: mitigate via stratified subsampling and batching.
- Persona metric noisiness: mitigate with multiple prompts/seeds and bootstrap CIs.
- Trait alignment weak signal: mitigate via regularized models and careful confound checks.
- Library compatibility: pin versions in `pyproject.toml` and log exact versions.

## Success Criteria
- End-to-end reproducible pipeline scripts in `src/`.
- Results artifacts saved in `results/` and `figures/`.
- Statistical tests with CIs/effect sizes completed.
- REPORT.md documents actual measured outcomes, limitations, and next experiments.
