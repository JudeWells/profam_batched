# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ProFam is a 251M-parameter autoregressive protein family language model trained via next-token prediction on concatenated unaligned protein sequences. Built on PyTorch Lightning + Hydra, it uses a Llama-based causal LM architecture with a 68-token vocabulary and 8192-token context window.

## Common Commands

### Installation
```bash
uv pip install -r requirements.txt          # GPU
uv pip install -r requirements-cpu.txt --index-strategy unsafe-best-match  # CPU-only
uv pip install -r requirements-dev.txt      # Dev tools
```

### Testing
```bash
pytest -k 'not example'       # Standard test run (excludes example/integration tests)
pytest -m 'not slow'          # Exclude slow tests
pytest tests/test_tokenizer.py # Run a single test file
```

### Linting & Formatting
```bash
pre-commit run --all-files    # Run all pre-commit hooks (isort, black, yaml checks)
```

### Training
```bash
python src/train.py experiment=train_profam_example                    # Lightweight example
python src/train.py model.config.hidden_size=64 trainer.max_steps=2    # Quick validation run
```

### Inference
```bash
python scripts/score_sequences.py    # Log-likelihood scoring (FASTA/A3M/A2M input)
python scripts/generate_sequences.py # Sequence generation/sampling
```

## Architecture

### Configuration (Hydra)
All training is configured via Hydra. `configs/train.yaml` is the base config; experiment overrides live in `configs/experiment/`. Key sections: `model`, `data`, `trainer`, `callbacks`, `logger`. Override any parameter from the command line.

### Models (`src/models/`)
- `base.py` — `BaseFamilyLitModule`: Core Lightning module handling training/validation/test steps, log-likelihood scoring, and sequence generation. Contains the bulk of the training logic.
- `llama.py` — `LlamaLitModule`: Thin wrapper that instantiates a HuggingFace `LlamaForCausalLM` inside the base module.
- `inference.py` — Inference utilities and pipeline.

### Data Pipeline (`src/data/`)
- `datamodule.py` — `ProteinDataMixture` Lightning DataModule.
- `tokenizers.py` — `ProFamTokenizer`: Handles sequence tokenization and packing. Wraps a HuggingFace tokenizer with protein-specific logic.
- `text_memmap_datasets.py` — Memory-mapped datasets for efficient random access to large sequence files without loading everything into memory. Uses `.idx.npy` and `.idx.info` index files.
- `objects.py` — Core dataclasses: `ProteinDocument`, `Protein`, `StringObject`.
- `collators.py` — `DocumentBatchCollator` for batch preparation.
- `processors/` — Data transformation pipeline: `preprocessing.py` (sequence cleaning), `transforms.py` (single-sample), `batch_transforms.py` (batch-level).

**ProFam-Atlas data format:**
- `.mapping` files: `>FAMILY_ID\nsequences_filename:idx0,idx1,...` (must NOT have trailing newline)
- `.sequences` files: `>ACCESSION\nSEQUENCE` (should have trailing newline)

**Preprocessing pipeline:** gap removal (`-`, `.`), uppercase conversion, non-canonical AA substitution (`U→C`, `O→K`, others → `[UNK]`).

**Sequence packing:** Multiple sequences packed into single context windows (`batch_size=1`); requires FlashAttention. Disable with `pack_to_max_tokens=null`.

### Entry Point
`src/train.py` — Main training script. Sets up Lightning Trainer, DataModule, and Model via Hydra config. Handles checkpoint loading/resuming.

### Test Fixtures (`tests/conftest.py`)
- `profam_tokenizer` — Default tokenizer
- `test_model` — Lightweight model (hidden_size=64, 1 layer) for fast tests
- `test_model_noseqpos` — Model without sequence position IDs
- `proteingym_batch` — ProteinGym batch data

## Key Constants (`src/constants.py`)
- `VOCAB_SIZE`: 68
- Feature names: `input_ids`, `attention_mask`, `aa_mask`, `original_size`, etc.

## CI
GitHub Actions runs pre-commit checks and `pytest -k 'not example'` on PRs to main, using Python 3.11 with a minimal CPU training validation (hidden_size=64, 1 layer, max_steps=2, attn=eager).
