# VPR-AttLLM: LLM-Guided Attention for Visual Place Recognition

This repository provides a clean, minimal implementation of **LLM-guided spatial attention for Visual Place Recognition (VPR)**.  It pairs a standard [CosPlace](https://github.com/gmberton/cosplace) descriptor with an **AttentionCosPlace** wrapper that injects per-region importance weights — produced by a vision-language model — directly into the GeM pooling stage.

The code supports three end-to-end pipelines:

| Pipeline | Description |
|----------|-------------|
| **A** | Generate flood-simulated street-view images (SVI) with a Gemini VLM |
| **B** | Annotate query SVIs with LLM spatial attention (grid JSON files) |
| **C** | VPR matching + evaluation (CosPlace baseline & AttentionCosPlace) |

---

## Method Overview

```
Query image ──► CosPlace backbone ──► Feature map [B, C, H, W]
                                              │
                     LLM attention grid ──► Attention map [H, W]   (Pipeline B)
                          {"A1":1.2, "B3":0.5, ...}
                                              │
                                        Weighted GeM pooling
                                              │
                                      L2-normalised descriptor
                                              │
                                    FAISS nearest-neighbour search
                                              │
                                       Top-k database matches ──► Recall@k evaluation
```

**AttentionCosPlace** blends the existing GeM contribution pattern with the LLM attention map using a scalar `att_ratio` α ∈ [0, 1]:

```
w_final = existing_weights + α × (LLM_attention − existing_weights)
y_c     = ( Σ_ij [w_final_ij · x_cij^p] / Σ_ij w_final_ij )^(1/p)
```

- `att_ratio = 0.0` → standard CosPlace (no LLM influence)  
- `att_ratio = 1.0` → full LLM attention weighting

---

## Repository Structure

```
VPR_AttLLM/
├── main.py                  # Entry point — configure and run pipelines
│
├── benchmark_models.py      # Cosplace model loader (PyTorch Hub)
├── att_models.py            # AttentionCosPlace (LLM-guided GeM pooling)
├── benchmark_feature.py     # Feature extraction, database caching, FAISS index
├── benchmark_matcher.py     # End-to-end retrieval pipeline + AQE
│
├── base_vpr.py              # Abstract VPR base class
├── project_utils.py         # Coordinate parsing, attention map helpers
├── record_evaluator.py      # Recall@k evaluation and result export
├── batch_controller.py      # Resumable batch processing with checkpointing
│
├── agent_bot.py             # GeminiAgent / QwenAgent wrappers (Pipeline A & B)
├── prompts.py               # PromptManager for VLM prompts
├── svi_agent_main.py        # SviAgent — batch SVI generation & annotation
├── svi_json_clean.py        # JSON cleaning / repair utilities
├── svi_preprocess.py        # Image preprocessing helpers (grid overlay, axis labels)
│
├── requirements.txt
├── .gitignore
└── API_key.txt.template     # Copy to API_key.txt and fill in your keys
```

---

## Installation

```bash
# 1. Clone / download the repository
git clone https://github.com/your-org/VPR_AttLLM.git
cd VPR_AttLLM

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
.\.venv\Scripts\Activate.ps1       # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt
```

> **GPU note:** The code automatically uses CUDA when available.  
> For GPU-accelerated FAISS replace `faiss-cpu` with `faiss-gpu` in `requirements.txt`.

---

## Dataset Preparation

### SF-XL (San Francisco Extra-Large)

The paper experiments use the **SF-XL** dataset.

- **GitHub / paper:** [gmberton/CosPlace](https://github.com/gmberton/CosPlace)  
- **Direct download (processed splits):** follow the instructions in the SF-XL repo.

Expected folder layout after download:

```
D:/VPR_dataset/sf-xl/processed/test/
├── database_full/       ← ~1.3 M database images  (target)
│   └── @<utm_e>@<utm_n>@<zoom>@S@<lat>@<lon>@....jpg
└── sf_flood/            ← query images (flooding scenes)
    ├── <lat>_<lon>@....jpg
    └── att_axis@gemini-2.5-flash/   ← LLM attention JSONs (Pipeline B output)
```

Filename coordinate conventions used in this project:

| Split | `coord` arg | Format example |
|-------|-------------|----------------|
| Query (sf_flood) | `"dash"` | `37.761_-122.415@...jpg` |
| Database (sf-xl) | `"parse"` | `@554123.40@4178231.50@10@S@037.761@-122.415@...jpg` |

### Custom dataset

You can use any image folder.  Name your images so coordinates can be parsed
(see `project_utils.py` → `dash_coordinates` / `parse_coordinates`), or modify
`get_coordinates_from_path()` for your own naming scheme.

---

## API Keys (Pipelines A & B only)

Pipelines A and B call cloud VLM APIs.  **Pipeline C (VPR matching) requires no API key.**

```bash
# Linux / macOS
export GEMINI_API_KEY="your-google-gemini-api-key"
export DASHSCOPE_API_KEY="your-alibaba-dashscope-key"   # Qwen only

# Windows PowerShell
$env:GEMINI_API_KEY = "your-google-gemini-api-key"
$env:DASHSCOPE_API_KEY = "your-alibaba-dashscope-key"
```

Get your Gemini API key at [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).  
Get your DashScope key at [https://dashscope.console.aliyun.com/](https://dashscope.console.aliyun.com/).

---

## Running the Pipelines

All three pipelines are driven from `main.py`.  Open the file and:

1. **Edit the path variables** in each `run_*` function to match your local data layout.  
2. **Uncomment** the desired pipeline call(s) at the bottom of `main()`.  
3. Run:

```bash
python main.py
```

### Pipeline A — Generate flooding SVI images

```python
# In main.py → main():
run_svi_image_generation_pipeline()
```

| Variable | Description |
|----------|-------------|
| `test_folder` | Folder containing original (dry) SVI images |
| `output_folder` | Destination for generated flood images |

Requires `GEMINI_API_KEY`.

---

### Pipeline B — LLM attention annotation

```python
# In main.py → main():
run_svi_llm_attention_pipeline()
```

| Variable | Description |
|----------|-------------|
| `svi_folder` | Query image folder |
| `city` | City name injected into the VLM prompt |
| `json_save_folder` | Output folder for per-image attention JSON files |

Output JSON format (one file per image):
```json
{
  "result": {
    "A1": 1.2, "A2": 0.8, "A3": 1.0,
    "B1": 0.5, "B2": 1.5, "B3": 1.1,
    "C1": 0.9, "C2": 1.0, "C3": 0.7
  }
}
```

Requires `GEMINI_API_KEY` (or `DASHSCOPE_API_KEY` for Qwen).

---

### Pipeline C — VPR matching + evaluation

```python
# In main.py → main():
run_llm_match_sweep(att_ratio_list=(0.0, 0.5, 1.0))
```

Key variables inside `run_llm_match_sweep()`:

| Variable | Description |
|----------|-------------|
| `working_dir` | Root directory for query and database folders |
| `query_svi_folder` | Query images (flooding scenes) |
| `target_svi_folder` | Database images (`database_full`) |
| `llm_json_folder` | Attention JSONs from Pipeline B |
| `att_ratio_list` | List of α values to sweep |
| `distance_threshold` | Positive match radius in metres (100 m for SF-XL flood) |

**att_ratio = 0.0** runs the standard CosPlace baseline.  
**att_ratio > 0** activates AttentionCosPlace.

Results are saved to `./result/<model>_<suffix>/<query>@<database>_<att_ratio>/`:

```
result/
└── cosplace_llmatt_feature_target_axis/
    └── sf_flood@database_full_0.5/
        ├── cosplace_results.json     ← top-100 retrieved candidates per query
        └── csv_results/
            ├── individual_results.csv
            └── aggregated_results.csv   ← Recall@1/5/10
```

---

## Expected Results (SF-XL Flood, 100 m threshold)

| Model | att_ratio | Recall@1 | Recall@5 | Recall@10 |
|-------|-----------|----------|----------|-----------|
| CosPlace (baseline) | 0.0 | — | — | — |
| AttentionCosPlace (Gemini) | 0.5 | — | — | — |
| AttentionCosPlace (Gemini) | 1.0 | — | — | — |

*Fill in with your results.*

<!-- ---

## Citation

If you use this code, please cite:

```bibtex
@article{yourpaper2025,
  title   = {LLM-Guided Attention for Visual Place Recognition},
  author  = {Your Name et al.},
  journal = {arXiv},
  year    = {2025}
}
``` -->

---

## License

This project is released under the MIT License.  
CosPlace model weights are subject to the [CosPlace license](https://github.com/gmberton/cosplace/blob/main/LICENSE).
