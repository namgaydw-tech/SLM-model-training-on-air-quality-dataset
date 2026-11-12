# Air Quality SLM — Design Thinking Project

**Developing a domain-specific Small Language Model for carbon monoxide estimation from air-quality sensor observations**

> **Project status (12 November 2026):** Dataset preprocessing, instruction-response formatting, chronological splitting, Hugging Face publication, and programmatic loading have been completed. A conventional machine-learning baseline is implemented separately. **SLM fine-tuning and final SLM evaluation are not yet complete.**

## Project overview

This Design Thinking project investigates how a **Small Language Model (SLM)** can be adapted to interpret structured environmental sensor readings and generate a carbon monoxide concentration response. The initial task is to estimate the reference `CO(GT)` measurement from historical air-quality observations.

The project is being developed using **Python, JupyterLab, Hugging Face Datasets, PyTorch, and Transformers**. The immediate research goal is to create a reproducible supervised fine-tuning and evaluation pipeline, not to present a prepared dataset as an already-trained language model.

- **GitHub repository:** https://github.com/namgaydw-tech/SLM-model-training-on-air-quality-dataset
- **Published Hugging Face dataset:** https://huggingface.co/datasets/namgaydw-ai/Air_quality/tree/main
- **Original data source:** [UCI Machine Learning Repository — Air Quality](https://archive.ics.uci.edu/dataset/360/air+quality) ([DOI: 10.24432/C59K5F](https://doi.org/10.24432/C59K5F))

## Problem statement

Air-quality datasets contain numerical time-series measurements that are not immediately suited to language-model training. The project therefore transforms sensor observations into **instruction–input–response** examples. The intended SLM will receive environmental measurements as a text prompt and generate a structured `CO(GT)` value.

This is **numerical estimation expressed as text generation**. A language model is not guaranteed to outperform a specialized regression model, so the project will evaluate its numerical accuracy, output validity, and generalization to later observations.

## Completed work

| Component | Current status | Evidence / description |
|---|---|---|
| Raw dataset exploration | Completed | UCI Air Quality CSV investigated in Jupyter |
| Preprocessing | Completed | Removed empty records/columns, handled `-200` missing-value markers, filtered incomplete examples |
| Instruction formatting | Completed | Converted sensor rows into text instructions with reference `CO(GT)` responses |
| JSONL creation | Completed | Produced **7,344** instruction-response examples |
| Chronological partitioning | Completed | Prepared **train / validation / test** splits without random shuffling |
| Dataset publication | Completed | Uploaded JSONL files and dataset documentation to Hugging Face |
| Programmatic verification | Completed | Loaded the public dataset successfully with `datasets.load_dataset()` in JupyterLab |
| Local GPU configuration | Verified | PyTorch detected **NVIDIA GeForce RTX 4050 Laptop GPU** with CUDA available |
| Conventional ML baseline | Implemented separately | `train.py` uses scikit-learn **ExtraTreesRegressor**; `predict.py` loads its saved `.pkl` artifact |
| Pretrained SLM initialization | Next stage | Proposed backbone: `HuggingFaceTB/SmolLM2-135M-Instruct` |
| SLM fine-tuning / evaluation | **Not completed** | No validated fine-tuned SLM checkpoint or final SLM metrics published |

## Dataset

### Source and preprocessing

The source is the historical **UCI Air Quality** dataset, containing observations recorded during 2004–2005. The preprocessing notebook reads the CSV using its original semicolon delimiter and comma decimal convention, handles missing measurements, selects usable sensor features and the `CO(GT)` target, and creates text examples.

The prompts include the observation date/time, five `PT08` sensor readings, temperature, relative humidity, and absolute humidity. The response is a measured carbon monoxide concentration (`CO(GT)`, in mg/m³).

### Published splits

| Split | JSONL file | Examples | Purpose |
|---|---|---:|---|
| Train | `train.jsonl` | **5,875** | Update SLM parameters during fine-tuning |
| Validation | `validation.jsonl` | **734** | Monitor generalization and choose training settings |
| Test | `test.jsonl` | **735** | Independent final evaluation after model selection |
| **Total** | | **7,344** | |

The splits were constructed in **chronological order** to better approximate evaluation on future observations. The test split should remain untouched during training and hyperparameter selection.

### JSONL format

Each line is a JSON object with a single `text` field containing a complete instruction–input–response example. The following is an illustrative example from the prepared data:

```json
{"text": "### Instruction:\nPredict the carbon monoxide concentration CO(GT) from the given air-quality sensor readings.\n\n### Input:\nDate: 10/03/2004\nTime: 18.00.00\nPT08.S1(CO): 1360.0\nPT08.S2(NMHC): 1046.0\nPT08.S3(NOx): 1056.0\nPT08.S4(NO2): 1692.0\nPT08.S5(O3): 1268.0\nTemperature: 13.6 C\nRelative Humidity: 48.9 %\nAbsolute Humidity: 0.7578\n\n### Response:\nCO(GT): 2.6"}
```

The full `### Response:` is the **training label**. During evaluation, the model must receive only the instruction and input; it must generate the response itself.

### Load the published dataset in JupyterLab

Install the Hugging Face Datasets library if necessary:

```powershell
python -m pip install datasets
```

Then run in a notebook cell:

```python
from datasets import load_dataset

dataset = load_dataset("namgaydw-ai/Air_quality")
print(dataset)
print(dataset["train"][0]["text"])
```

This dataset-loading procedure has been successfully executed in the local JupyterLab environment with all three splits detected.

## Current repository components

| Path | Purpose |
|---|---|
| `air_quality_llm_training.ipynb` | Data inspection, preprocessing, instruction formatting, and original JSONL export |
| `data/` | Project data files; the maintained public train/validation/test splits are hosted on Hugging Face |
| `train.py` | **Conventional ML baseline**, using an Extra Trees regression pipeline (not SLM training) |
| `predict.py` | Command-line inference using the conventional baseline's saved `.pkl` model (not SLM inference) |
| `requirements.txt` | Existing Python dependencies for the baseline/data-processing stage |
| `.gitignore` | Excludes local environments and unnecessary artifacts |

The current GitHub repository does **not yet contain a completed fine-tuned SLM**. Separate SLM scripts, saved training receipts, and evaluation results will be added in later milestones.

## Development roadmap

### Phase 1 — Dataset engineering (complete)

- Explore and clean the UCI source data.
- Create instruction–input–response examples in JSONL.
- Partition the examples chronologically into train, validation, and test files.
- Publish the processed dataset to Hugging Face and verify loading from JupyterLab.

### Phase 2 — SLM setup and tokenization (next)

- Adopt a compact pretrained **decoder-only Transformer**, initially `HuggingFaceTB/SmolLM2-135M-Instruct` (subject to memory and compatibility testing).
- Inspect tokenizer behavior, prompt length, truncation, and output formatting.
- Prepare a tokenized training pipeline that supervises the target response while avoiding unintended learning from held-out labels.
- Record package versions, random seeds, configuration, GPU memory, and reproducibility details.

### Phase 3 — Supervised fine-tuning (planned)

- Fine-tune the pretrained SLM with **Supervised Fine-Tuning (SFT)** and **Low-Rank Adaptation (LoRA)** using the training split.
- First run a small smoke test to verify forward/backward passes and checkpoint saving.
- Select training settings using the **validation** split (for example learning rate, epochs, sequence length, and batch size).
- Save the LoRA adapter, tokenizer/configuration, training logs, and loss curves.

### Phase 4 — Independent evaluation (planned)

- Evaluate the finalized SLM **once on the held-out chronological test split**.
- Measure **numeric-output validity**, **Mean Absolute Error (MAE)**, and **Root Mean Squared Error (RMSE)**.
- Compare results with the unfine-tuned SLM and the conventional Extra Trees baseline on compatible examples.
- Document invalid outputs, error patterns, limitations, and reproducibility instructions; do not invent performance claims.

### Phase 5 — Demonstration and reporting (planned)

- Build a simple notebook or interface that accepts sensor readings and returns the generated `CO(GT)` estimate.
- Clearly label generated predictions as experimental and not real-time sensor readings or safety-grade measurements.
- Publish final code, reproducible evaluation results, and model documentation after validation.

## Development environment

The local development setup uses **Windows, PowerShell, Python virtual environments, and JupyterLab**. GPU availability was confirmed as follows:

```text
PyTorch: 2.14.1+cu126
CUDA available: True
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
```

To launch the existing project environment (adjust if the installation path changes):

```powershell
cd "D:\Design thinking LLM"
.\.venv\Scripts\Activate.ps1
jupyter lab
```

In JupyterLab, select the **Python (Air Quality SLM)** kernel. Installing `requirements.txt` alone currently installs the data-processing/baseline dependencies; the upcoming SLM phase will additionally require appropriate versions of `torch`, `transformers`, `peft`, `accelerate`, and `datasets`.

## Technical limitations and responsible use

- The dataset is historical and site-specific; results may not generalize to different sensors, locations, or contemporary air-quality conditions.
- Missing values and sensor drift can affect model quality.
- Generating a number as text does not guarantee a valid or accurate regression estimate.
- Data preparation and public dataset hosting do **not** constitute a trained SLM.
- This is an educational research prototype, **not** a certified environmental monitoring, public-health, or regulatory tool.

## Attribution

Original dataset: **Vito, S. (2008). _Air Quality_ [Dataset]. UCI Machine Learning Repository.** https://doi.org/10.24432/C59K5F

This repository documents a transformed instructional dataset built from that source. Please consult the original dataset documentation and licensing terms before redistribution or commercial use.

---

**Current milestone:** Processed and published training-ready JSONL data. **Next milestone:** Tokenize the data and fine-tune an actual SLM using LoRA, followed by independent quantitative evaluation.
