# Basis Vectors in Persona Space

This project tests whether persona representations in LLM hidden states can be decomposed into shared basis vectors via PCA, then evaluated for stability and causal steering utility.

## Key Findings
- Top-10 PCs explain **57.14%** of persona-vector variance (top-20: **70.06%**).
- PCA subspace is stable across bootstrap splits (overlap **0.905 ± 0.016**).
- Steering signal is mixed: `PC-` vs base is significant (**p=0.0228**, small effect), `PC+` is not significant vs base/random.
- External Big-5 alignment is weak (no BH-FDR significant correlations).

## Reproduce
1. Activate environment:
   ```bash
   source .venv/bin/activate
   ```
2. Run experiment:
   ```bash
   python src/run_persona_pca_experiment.py
   ```
3. Inspect outputs:
   - `results/metrics.json`
   - `results/plots/`
   - `REPORT.md`

## File Structure
- `src/run_persona_pca_experiment.py`: end-to-end experiment pipeline.
- `planning.md`: motivation, novelty, and experimental plan.
- `REPORT.md`: full analysis and conclusions.
- `results/`: metrics, tables, and generated artifacts.
- `datasets/`: pre-downloaded local datasets used in the study.

For full details and interpretation, see `REPORT.md`.
