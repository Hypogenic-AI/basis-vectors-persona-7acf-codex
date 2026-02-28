# Downloaded Datasets

This directory contains datasets for persona-space and representation analysis experiments.
Large data artifacts are excluded from git by design.

## Dataset 1: Persona-Chat

### Overview
- Source: `AlekseyKorshuk/persona-chat` (HuggingFace)
- Size: train 17,878 / validation 1,000
- Format: HuggingFace `DatasetDict` with nested JSON-like records
- Task: persona-conditioned dialogue generation/analysis
- Splits: train, validation
- License: see dataset card on HuggingFace

### Local Location
- `datasets/persona_chat/`

### Download Instructions

Using HuggingFace (recommended):
```python
from datasets import load_dataset

ds = load_dataset("AlekseyKorshuk/persona-chat")
ds.save_to_disk("datasets/persona_chat")
```

### Loading the Dataset
```python
from datasets import load_from_disk

ds = load_from_disk("datasets/persona_chat")
```

### Sample Data
- `datasets/persona_chat/samples/samples.json`

### Notes
- Contains persona statements and dialogue candidates.
- Useful for deriving persona vectors and persona consistency probes.

## Dataset 2: PersonaHub (persona config)

### Overview
- Source: `proj-persona/PersonaHub` with config `persona` (HuggingFace)
- Size: train 200,000
- Format: HuggingFace dataset with `persona` text field
- Task: large-scale persona text corpus for embedding/PCA
- Splits: train
- License: see dataset card on HuggingFace

### Local Location
- `datasets/personahub_persona/`

### Download Instructions
```python
from datasets import load_dataset

ds = load_dataset("proj-persona/PersonaHub", "persona")
ds.save_to_disk("datasets/personahub_persona")
```

### Loading the Dataset
```python
from datasets import load_from_disk

ds = load_from_disk("datasets/personahub_persona")
```

### Sample Data
- `datasets/personahub_persona/samples/samples.json`

### Notes
- Strong fit for this hypothesis because persona descriptions are explicit and diverse.
- Good source for creating persona vector banks before PCA.

## Dataset 3: myPersonality (text + Big-5 labels)

### Overview
- Source: `holistic-ai/Personality_mypersonality` (HuggingFace)
- Size: train 7,933 / test 1,984
- Format: HuggingFace dataset with `text` and Big-5 label object (`answer`)
- Task: personality-conditioned text analysis and supervised checks
- Splits: train, test
- License: see dataset card on HuggingFace

### Local Location
- `datasets/mypersonality/`

### Download Instructions
```python
from datasets import load_dataset

ds = load_dataset("holistic-ai/Personality_mypersonality")
ds.save_to_disk("datasets/mypersonality")
```

### Loading the Dataset
```python
from datasets import load_from_disk

ds = load_from_disk("datasets/mypersonality")
```

### Sample Data
- `datasets/mypersonality/samples/samples.json`

### Notes
- Useful for validating whether principal components correlate with personality dimensions.

## Quick Validation Performed
- Loaded each dataset and inspected first 100 examples.
- Verified schema and split accessibility.
- Saved first 10 examples for each dataset under corresponding `samples/` directories.
