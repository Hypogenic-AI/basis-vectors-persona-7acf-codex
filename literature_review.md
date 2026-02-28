# Literature Review: Basis Vectors in Persona Space

## Research Area Overview
This review targets the hypothesis that persona representations in language models can be decomposed into component vectors, and that PCA over persona vectors reveals primary axes in residual-stream geometry. The most relevant neighboring areas are:
- Persona-conditioned generation and persona fidelity evaluation
- Steering vectors / activation engineering for behavior control
- Residual-stream geometry and mechanistic interpretability
- Sparse and linear decomposition methods (PCA, SAE, linear probes)

The reviewed evidence consistently supports a strong linear-geometry signal in hidden states, while also showing important limits: steering effects can be unstable across prompts and behaviors, and geometric assumptions can break under distribution shift.

## Review Scope

### Research Question
Can persona behavior be represented as approximately linear directions/subspaces in model residual streams, and can PCA recover robust primary persona components?

### Inclusion Criteria
- Persona modeling, persona-steered generation, or persona fidelity in LLMs
- Steering vectors / representation engineering in residual activations
- Residual-stream geometry analyses in transformer models
- Papers with actionable methodology for intervention/evaluation

### Exclusion Criteria
- Non-language persona work (e.g., purely visual avatars) unless method transfers clearly
- Papers without method detail relevant to residual representations
- Non-technical opinion/commentary

### Time Frame
- Primary focus: 2024-2026
- Foundational anchor: PersonaChat paper (2018)

### Sources
- arXiv (primary)
- HuggingFace datasets (for benchmark acquisition)
- GitHub repositories for implementations

## Search Log

| Date | Query | Source | Results | Notes |
|------|-------|--------|---------|-------|
| 2026-02-28 | `persona representations residual stream PCA language models` | local `find_papers.py` | service stalled | Switched to manual search |
| 2026-02-28 | persona dialogue / persona-conditioned LM / steering vectors / residual stream geometry | arXiv API | 70 unique candidates | Curated top 10 relevant papers |
| 2026-02-28 | persona/personality datasets | HuggingFace | multiple candidates | Downloaded 3 datasets |

## Screening Results
- Title/abstract screening: 70 candidates
- Full download: 10 papers (all in `papers/`)
- Deep reading setup: chunked 3 highest-priority papers using PDF chunker

## Key Papers

### Paper 1: Transformers represent belief state geometry in their residual stream
- **Authors**: Adam S. Shai et al.
- **Year**: 2024
- **Source**: arXiv 2405.15943
- **Key Contribution**: Shows residual streams encode belief-state geometry with approximately linear structure.
- **Methodology**: Theoretical framing from belief updating + empirical geometry analysis of residual activations.
- **Datasets Used**: Task-specific hidden-state evaluations (see paper for exact setups).
- **Results**: Strong evidence for geometrically meaningful latent representations in residual space.
- **Code Available**: Not confirmed from abstract metadata.
- **Relevance to Our Research**: Core theoretical support for PCA/subspace analysis over persona vectors.

### Paper 2: Personalized Steering of LLMs via Bi-directional Preference Optimization
- **Authors**: Yuanpu Cao et al.
- **Year**: 2024
- **Source**: arXiv 2406.00045
- **Key Contribution**: Learns versatile steering vectors for personalization without full fine-tuning.
- **Methodology**: Preference-driven optimization of steering directions applied at inference-time activations.
- **Datasets Used**: Benchmark-style evaluation sets (paper reports multiple tasks).
- **Results**: Steering vectors improve personalized control while preserving base-model utility.
- **Code Available**: Not confirmed in metadata.
- **Relevance to Our Research**: Provides practical recipe for constructing persona vectors before PCA.

### Paper 3: Residual Stream Analysis with Multi-Layer SAEs
- **Authors**: Tim Lawson et al.
- **Year**: 2024
- **Source**: arXiv 2409.04185
- **Key Contribution**: Introduces multi-layer SAE to analyze information flow across residual streams.
- **Methodology**: Train single SAE over activations from multiple layers instead of layer-isolated SAEs.
- **Datasets Used**: Transformer activation corpora (see paper for exact corpus/model details).
- **Results**: Better cross-layer interpretability than layer-by-layer SAE isolation.
- **Code Available**: Related tooling available in SAE ecosystem.
- **Relevance to Our Research**: Strong complement to PCA for identifying shared axes across layers.

### Paper 4: Understanding Unreliability of Steering Vectors in Language Models
- **Authors**: Joschka Braun
- **Year**: 2026
- **Source**: arXiv 2602.17881
- **Key Contribution**: Explains why steering vectors can be inconsistent; proposes geometric predictors.
- **Methodology**: Correlates steering success with cosine similarity and training-activation geometry.
- **Datasets Used**: Multi-benchmark steering evaluation.
- **Results**: Reliability strongly depends on vector geometry and training sample alignment.
- **Code Available**: Not confirmed in metadata.
- **Relevance to Our Research**: Warns against over-interpreting top PCs without reliability diagnostics.

### Paper 5: Spherical Steering
- **Authors**: Zejia You et al.
- **Year**: 2026
- **Source**: arXiv 2602.08169
- **Key Contribution**: Replaces additive steering with norm-preserving geometric rotation.
- **Methodology**: Training-free activation rotation on a sphere.
- **Datasets Used**: Standard LM steering evaluations.
- **Results**: Better control-quality tradeoff under some settings than additive steering.
- **Code Available**: Not confirmed in metadata.
- **Relevance to Our Research**: Tests whether principal persona axes are additive or rotation-aligned.

### Paper 6: Improving Reasoning Performance via Representation Engineering
- **Authors**: Bertram Højer et al.
- **Year**: 2025
- **Source**: arXiv 2504.19483
- **Key Contribution**: Uses residual-stream-derived control vectors to improve reasoning.
- **Methodology**: Read/construct control vectors from task activations and intervene during generation.
- **Datasets Used**: Reasoning benchmarks (paper reports benchmark suite).
- **Results**: Demonstrates measurable gains under intervention.
- **Code Available**: Not confirmed in metadata.
- **Relevance to Our Research**: Provides end-to-end intervention template for vector directions.

### Paper 7: Understanding Reasoning in Thinking LMs via Steering Vectors
- **Authors**: Constantin Venhoff et al.
- **Year**: 2025
- **Source**: arXiv 2506.18167
- **Key Contribution**: Studies steering behavior in chain-of-thought-heavy models.
- **Methodology**: Analyze and inject steering vectors across reasoning tasks.
- **Datasets Used**: Multi-task reasoning evaluation.
- **Results**: Steering can modulate reasoning behavior but with controllability limits.
- **Code Available**: Not confirmed in metadata.
- **Relevance to Our Research**: Useful for testing persona vectors in long-form generation settings.

### Paper 8: Evaluating LLM Biases in Persona-Steered Generation
- **Authors**: Andy Liu, Mona Diab, Daniel Fried
- **Year**: 2024
- **Source**: arXiv 2405.20253
- **Key Contribution**: Defines incongruous multi-trait personas and evaluates generated bias effects.
- **Methodology**: Persona-steered generation evaluation against survey-derived expectations.
- **Datasets Used**: Persona and survey-based evaluation resources.
- **Results**: Shows non-trivial mismatch between intended and generated persona behavior.
- **Code Available**: Not confirmed in metadata.
- **Relevance to Our Research**: Supplies failure analyses and evaluation framing for persona-vector faithfulness.

### Paper 9: Spotting Out-of-Character Behavior
- **Authors**: Jisu Shin et al.
- **Year**: 2025
- **Source**: arXiv 2506.19352
- **Key Contribution**: Atomic-level evaluation of persona fidelity in open-ended generation.
- **Methodology**: Fine-grained OOC detection beyond single response-level scores.
- **Datasets Used**: Persona-conditioned generation evaluations.
- **Results**: Detects subtle misalignment missed by coarse metrics.
- **Code Available**: Not confirmed in metadata.
- **Relevance to Our Research**: Useful metric layer for evaluating PC-based persona interventions.

### Paper 10: Personalizing Dialogue Agents (PersonaChat)
- **Authors**: Saizheng Zhang et al.
- **Year**: 2018
- **Source**: arXiv 1801.07243
- **Key Contribution**: Introduces PersonaChat benchmark and persona-conditioned dialogue setting.
- **Methodology**: Condition dialogue models on profile sentences and dialogue context.
- **Datasets Used**: PersonaChat (ConvAI2).
- **Results**: Persona conditioning improves engagement and consistency.
- **Code Available**: Historically integrated into ParlAI ecosystem.
- **Relevance to Our Research**: Foundational benchmark/data for persona representation studies.

## Common Methodologies
- Activation steering vectors: used for behavior/persona modulation without full fine-tuning.
- Residual-stream probing: linear analyses over hidden states (projection, PCA, probing, cosine geometry).
- Representation interventions: add/rotate/project vectors at selected layers/tokens.
- Mechanistic decomposition: SAE-based factorization as nonlinear complement to PCA.

## Standard Baselines
- Prompt-only persona steering
- Fine-tuning / LoRA adaptation
- Activation addition steering vectors
- Zero-shot and few-shot prompting controls
- Random direction controls / null interventions (recommended for this project)

## Evaluation Metrics
- Persona fidelity / out-of-character rate
- Task accuracy on benchmark tasks
- Steering success rate and side-effect rate
- Cosine alignment between expected and observed direction shifts
- Utility preservation metrics (performance drop on non-target tasks)

## Datasets in the Literature
- PersonaChat / ConvAI2: classical persona-conditioned dialogue
- Synthetic/persona corpora: large-scale persona statements for controllable conditioning
- Reasoning and utility benchmarks: commonly used to verify no catastrophic capability loss after steering

## Gaps and Opportunities
- Gap 1: Limited work directly tests PCA bases over persona embedding banks as the primary decomposition method.
- Gap 2: Reliability of persona vectors across domains/prompts is under-characterized.
- Gap 3: Cross-layer consistency of persona components is not fully studied.
- Gap 4: Evaluation often mixes persona fidelity and task utility without clear Pareto analysis.

## Recommendations for Our Experiment
- **Recommended datasets**:
  - `proj-persona/PersonaHub` (`persona` config) as primary source for diverse persona text vectors.
  - `AlekseyKorshuk/persona-chat` for dialogue-level persona behavior checks.
  - `holistic-ai/Personality_mypersonality` for trait-correlated validation.
- **Recommended baselines**:
  - Prompt-only persona control
  - Additive steering vectors (RepE-style)
  - Random-direction controls
  - Optional spherical steering baseline
- **Recommended metrics**:
  - Persona fidelity / OOC detection
  - Downstream utility retention
  - Variance explained by top-K PCs
  - Intervention stability across prompts and seeds
- **Methodological considerations**:
  - Build persona vector bank from consistent layer/token extraction rules.
  - Run PCA per layer and jointly across layers to test transferability.
  - Quantify reliability: per-persona variance, confidence intervals, and failure cases.
  - Add ablations: PCA vs random orthonormal basis vs SAE features.

## Practical Resource Notes
- PDF chunks created for deep reading:
  - `papers/pages/2406.00045_*`
  - `papers/pages/2405.15943_*`
  - `papers/pages/2409.04185_*`
- Collected code tooling:
  - `code/representation-engineering/`
  - `code/pyvene/`
  - `code/saelens/`
  - `code/transformerlens/`
