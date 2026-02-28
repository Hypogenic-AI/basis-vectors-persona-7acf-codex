## Resources Catalog

### Summary
This document catalogs all resources gathered for the project **Basis Vectors in Persona Space**, including papers, datasets, and code repositories.

### Papers
Total papers downloaded: 10

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Transformers represent belief state geometry in their residual stream | Shai et al. | 2024 | papers/2405.15943_v3_transformers_represent_belief_state_geometry_in_their_residu.pdf | Residual-stream geometry evidence |
| Understanding Unreliability of Steering Vectors... | Braun | 2026 | papers/2602.17881_v1_understanding_unreliability_of_steering_vectors_in_language_.pdf | Steering reliability diagnostics |
| Spotting Out-of-Character Behavior... | Shin et al. | 2025 | papers/2506.19352_v1_spotting_out_of_character_behavior_atomic_level_evaluation_o.pdf | Persona fidelity metrics |
| Personalizing Dialogue Agents... | Zhang et al. | 2018 | papers/1801.07243_v5_personalizing_dialogue_agents_i_have_a_dog_do_you_have_pets_.pdf | PersonaChat foundation |
| Personalized Steering of LLMs... | Cao et al. | 2024 | papers/2406.00045_v2_personalized_steering_of_large_language_models_versatile_ste.pdf | Personalized steering vectors |
| Improving Reasoning Performance via Representation Engineering | Højer et al. | 2025 | papers/2504.19483_v1_improving_reasoning_performance_in_large_language_models_via.pdf | Rep engineering interventions |
| Spherical Steering... | You et al. | 2026 | papers/2602.08169_v1_spherical_steering_geometry_aware_activation_rotation_for_la.pdf | Geometry-aware steering |
| Understanding Reasoning in Thinking LMs via Steering Vectors | Venhoff et al. | 2025 | papers/2506.18167_v4_understanding_reasoning_in_thinking_language_models_via_stee.pdf | Steering in long reasoning |
| Evaluating LLM Biases in Persona-Steered Generation | Liu et al. | 2024 | papers/2405.20253_v1_evaluating_large_language_model_biases_in_persona_steered_ge.pdf | Persona/bias analysis |
| Residual Stream Analysis with Multi-Layer SAEs | Lawson et al. | 2024 | papers/2409.04185_v3_residual_stream_analysis_with_multi_layer_saes.pdf | Cross-layer residual analysis |

See `papers/README.md` for descriptions and `papers/metadata.json` for full metadata.

### Datasets
Total datasets downloaded: 3

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| Persona-Chat | HuggingFace (`AlekseyKorshuk/persona-chat`) | 17,878 train + 1,000 val | Persona-conditioned dialogue | datasets/persona_chat/ | Includes persona profiles + utterances |
| PersonaHub (persona) | HuggingFace (`proj-persona/PersonaHub`, config `persona`) | 200,000 train | Persona text corpus | datasets/personahub_persona/ | Primary corpus for persona vector bank |
| myPersonality | HuggingFace (`holistic-ai/Personality_mypersonality`) | 7,933 train + 1,984 test | Personality-labeled text | datasets/mypersonality/ | Trait correlation checks |

See `datasets/README.md` for full download and loading instructions.

### Code Repositories
Total repositories cloned: 4

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| representation-engineering | https://github.com/andyzoujm/representation-engineering | RepE control/readout baselines | code/representation-engineering/ | Directly relevant intervention baselines |
| pyvene | https://github.com/stanfordnlp/pyvene | Model intervention library | code/pyvene/ | Flexible activation interventions |
| SAELens | https://github.com/jbloomAus/SAELens | SAE training/analysis toolkit | code/saelens/ | Complement to PCA decomposition |
| TransformerLens | https://github.com/TransformerLensOrg/TransformerLens | Transformer internal activations/hooks | code/transformerlens/ | Residual-stream extraction backbone |

See `code/README.md` for key files and usage notes.

### Resource Gathering Notes

#### Search Strategy
- Attempted local `paper-finder` service first (`find_papers.py --mode diligent`), then switched to manual arXiv API due service stall.
- Queried persona-steering, residual-stream geometry, representation engineering, and steering-vector reliability terms.
- Prioritized papers with direct relevance to persona vectors and residual geometry.

#### Selection Criteria
- Direct relation to persona-conditioned behavior, steering vectors, or residual-stream analysis.
- Recent papers (2024-2026) plus one foundational benchmark paper (PersonaChat, 2018).
- Practical reproducibility: papers/datasets/code with clear experimental utility.

#### Challenges Encountered
- Local paper-finder backend did not return within a practical timeout.
- One candidate dataset (`bavard/personachat_truecased`) used deprecated script format and was replaced by a compatible mirror (`AlekseyKorshuk/persona-chat`).

#### Gaps and Workarounds
- Some papers do not clearly advertise official code links in metadata; mitigated by collecting robust tooling repos (RepE, pyvene, SAELens, TransformerLens).
- Paper extraction relied on metadata + targeted PDF parsing; full line-by-line reading remains optional for experiment phase.

### Recommendations for Experiment Design

1. **Primary dataset(s)**: `proj-persona/PersonaHub` for vector-bank construction, validated by `persona-chat` behavior checks.
2. **Baseline methods**: prompt-only persona control, additive steering vectors, random direction controls, optional spherical steering.
3. **Evaluation metrics**: variance explained by PCs, persona fidelity/OOC rates, steering stability, utility retention.
4. **Code to adapt/reuse**: use TransformerLens for activations, RepE/pyvene for interventions, SAELens for non-PCA decomposition comparison.

## Research Execution Update (2026-02-28)

### Implemented Experiment
- Script: `src/run_persona_pca_experiment.py`
- Model used: `Qwen/Qwen2.5-0.5B`
- Core outputs: `results/metrics.json`, `results/steering_outputs.csv`, `results/trait_alignment_rows.json`, `results/plots/*.png`

### What Was Tested
1. Persona vector extraction from PersonaHub and PCA decomposition.
2. Subspace stability across bootstrap splits and random-basis comparison.
3. Causal steering using top PC direction vs random baseline on PersonaChat prompts.
4. External correlation between PC projections and myPersonality Big-5 labels.

### Outcome Snapshot
- Strong low-dimensional geometry and stability.
- Partial causal steering signal (direction-dependent).
- Weak external Big-5 alignment in this run.

See `REPORT.md` for full results and interpretation.
