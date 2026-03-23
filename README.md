> 🇷🇺 [Русский](README.ru.md) | 🇬🇧 English

# 🚀 LLM Training Acceleration: Optimizer Comparison

> **Solution for the Technical Assignment "LLM Training Acceleration"**  
> Comparative analysis of AdamW, Muon, and a hybrid approach for LoRA fine-tuning of the Qwen2.5-0.5B model

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📋 Project Overview

This project implements a comparative study of optimization strategies for parameter-efficient fine-tuning of large language models. The goal is to evaluate **training speed**, **GPU memory efficiency**, and **final model quality** for three optimizers:

| Optimizer | Description |
|-----------|-------------|
| **AdamW** | Standard adaptive optimizer |
| **Muon** | Matrix orthogonalization-based optimizer (local implementation) |
| **Hybrid** | Hybrid approach: 50% of 2D parameters → Muon, the rest → AdamW |

### 🔬 Experiment Parameters

- **Model**: [`Qwen/Qwen2.5-0.5B`](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- **Dataset**: [`Elriggs/openwebtext-100k`](https://huggingface.co/datasets/Elriggs/openwebtext-100k) (1% = 1,000 samples)
- **Evaluation**: [PIQA](https://github.com/EleutherAI/lm-evaluation-harness) (logical reasoning)
- **Fine-tuning**: LoRA (`r=16`, `alpha=32`, `dropout=0.05`)
- **Precision**: `bfloat16` (native support on RTX 40/50 series)
- **Framework**: Hugging Face `transformers` + `trl` (SFTTrainer)

> ⚠️ **Note**: The MeZO optimizer is **not implemented** in this version (advanced task postponed).

---

## ✨ Features

- ✅ **Hybrid optimizer**: automatic parameter splitting (50% of 2D tensors → Muon, the rest → AdamW)
- ✅ **Parameter filtering**: exclusion of `embed_tokens` and `lm_head` from the Muon group
- ✅ **Efficient memory usage**: gradient checkpointing + bf16 + LoRA
- ✅ **Detailed logging**: loss, LR, memory, and time at each step
- ✅ **Automatic PIQA evaluation** via `lm-evaluation-harness`
- ✅ **Cross-platform setup scripts** (`setup.sh` / `setup.bat`)

---

## 📦 Installation

### Requirements

- Python **3.12** (required by setup scripts)
- CUDA-capable GPU with ≥8 GB VRAM (16 GB recommended for `batch_size=4`)
- Git

### Quick Start

```bash
# Clone the repository
git clone https://github.com/Not-broken-today/llm-training-acceleration.git    
cd llm-training-acceleration

# Run setup (creates venv + installs dependencies)
# For Linux/macOS:
chmod +x setup.sh && ./setup.sh
# For Windows:
setup.bat

# Activate the environment (if not auto-activated)
# For Linux/macOS:
source .venv_llm/bin/activate
# For Windows CMD:
.venv_llm\Scripts\activate
```

### Dependencies

Core packages are listed in `requirements.txt`:
```txt
torch>=2.5.0
transformers>=4.45.0
trl>=0.12.0
datasets>=3.0.0
lm-eval>=0.4.3
peft>=0.13.0
# + utilities
```

> 💡 Custom Muon implementation is included locally — no external repository installation required.

---

## ⚙️ Configuration

Edit `config.yaml` to configure experiments:

```yaml
# Active optimizer: "adamw" | "muon" | "hybrid"
active_optimizer: "adamw"

model:
  model_name: "Qwen/Qwen2.5-0.5B"
  torch_dtype: "bfloat16"  # Use "float16" for older GPUs

lora:
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

training:
  num_train_epochs: 5
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 5e-5
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.03
  bf16: true
  gradient_checkpointing: true

optimizers:
  muon:
    ns_steps: 5
    momentum: 0.95
    nesterov: true
  adamw:
    betas: [0.9, 0.999]
    eps: 1e-8
```

---

## ▶️ Running Experiments

### Single Optimizer

```bash
# Run with AdamW (default)
python src/main.py

# Run with Muon
python src/main.py active_optimizer=muon

# Run with hybrid optimizer
python src/main.py active_optimizer=hybrid
```

### Via Ready-Made Scripts

```bash
# Linux/macOS
./run_file/AdamW_run.sh
./run_file/Muon_run.sh
./run_file/Hybrid_run.sh

# Windows
run_file\AdamW_run.bat
run_file\Moun_run.bat
run_file\Hybrid_run.bat
```

### Results

Results are saved to `outputs/checkpoints/`:
```
outputs/
├── checkpoints/
│   ├── metrics_{optimizer}.json   # Per-step metrics
│   ├── results_{optimizer}.json   # Final summary + PIQA evaluation
│   └── checkpoint-*/              # LoRA weights (PEFT format)
└── YYYY-MM-DD/
        └── hh-mm-ss/
            └── main.log           # Saved logs
```

---

## 📊 Results (Actual Data)

*Experiments conducted on RTX 5060 Ti 16 GB, Windows 11, Python 3.12*

| Optimizer | Training Time | Peak Memory | Final Loss | PIQA Accuracy | Steps/sec |
|-----------|--------------|-------------|------------|---------------|-----------|
| **AdamW** | 7m 41s | 3912 MB | 2.863 | **70.29%** | 0.682 |
| **Muon** | 9m 22s | 3846 MB | 2.903 | 70.13% | 0.560 |
| **Hybrid** | 8m 23s | 3878 MB | 2.888 | 69.86% | 0.626 |

### 🔍 Key Observations

1. **Memory**: Muon uses ~2–3% less VRAM (one momentum buffer vs. two in AdamW)
2. **Speed**: AdamW is fastest; Muon slowdown is due to Newton-Schulz iterations
3. **Quality**: All optimizers show comparable PIQA accuracy (~70%) with limited data
4. **Stability**: The hybrid approach demonstrates smooth convergence, combining advantages of both methods

---

## 🗂️ Project Structure

```
llm-training-acceleration/
├── 📄 README.md                 # This file
├── 📄 config.yaml               # Experiment configuration
├── 📄 requirements.txt          # Python dependencies
├── 📄 setup.py / setup.sh / setup.bat  # Installation scripts
├── 📄 mypy.ini / pylint_test_code.*    # Type checking and linting
│
├── 📁 src/                      # Source code
│   ├── __init__.py
│   ├── main.py                  # Entry point: training pipeline
│   ├── model.py                 # Model loading + LoRA application
│   ├── dataset.py               # Dataset loading and tokenization
│   │
│   ├── 📁 optimizers/
│   │   ├── __init__.py
│   │   ├── optimizer.py         # Factory: optimizer selection/creation
│   │   │   ├── create_optimizer(name, params, **kwargs)
│   │   │   ├── create_trainer_config(**kwargs)
│   │   │   └── create_trainer(...)
│   │   └── muon.py              # Muon implementation
│   │
│   ├── 📁 utilities/
│   │   ├── loggers/             # MetricsCallback, memory monitoring
│   │   └── system/              # GPU utilities, cleanup
│   │
│   └── 📁 evaluation/
│       └── piqa_eval.py         # PIQA dataset evaluation
│
├── 📁 run_file/                 # Ready-made launch scripts
│   ├── AdamW_run.sh / .bat
│   ├── Muon_run.sh / .bat
│   ├── Hybrid_run.sh / .bat
│   └── All_run.sh / .bat        # Sequential run of all three
│
└── 📁 outputs/                  # Generated during training (in .gitignore)
    ├── checkpoints/
    └── 📁 YYYY-MM-DD/
        └── 📁 hh-mm-ss/
            └── 📄 main.log

```

---

## 🔧 Optimizer Selection Logic (`src/optimizers/optimizer.py`)

### Parameter Filtering

```python
# Parameters for Muon: 2D+ tensors, excluding embed_tokens and lm_head
muon_params = [
    p for p in params
    if p.ndim >= 2 and "embed_tokens" not in str(p) and "lm_head" not in str(p)
]

# Remaining parameters → AdamW
adamw_params = [p for p in params if p not in muon_params]
```

### Hybrid Mode

```python
# All 2D parameters (excluding embed_tokens/lm_head)
all_2d_params = [...]  # as above

# Split in half
split_idx = len(all_2d_params) // 2
muon_params = all_2d_params[:split_idx]           # first half → Muon
adamw_params = all_2d_params[split_idx:] + other_params  # second half + rest → AdamW
```

---

## 🧪 Code Quality

The project uses:
- **`mypy`** for static type checking (settings in `mypy.ini`)
- **`pylint`** for linting (helper script `pylint_test_code.sh / .bat`)

Running checks:
```bash
# Linux/macOS
chmod +x pylint_test_code.sh && ./pylint_test_code.sh

# Windows
pylint_test_code.bat

# Or manually:
pylint src/**/*.py
```

---

## 📝 Work Plan / Known Limitations

- [ ] Implement the MeZO optimizer (advanced task)

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

## 🙏 Acknowledgements

- [MoonshotAI / Muon](https://github.com/MoonshotAI/Moonlight) — Muon optimizer implementations
- [Princeton NLP / MeZO](https://github.com/princeton-nlp/MeZO) — gradient-free optimization
- [Hugging Face](https://huggingface.co) — transformers, datasets, TRL, PEFT
- [EleutherAI](https://github.com/EleutherAI/lm-evaluation-harness) — model evaluation framework

---

*Last updated: March 2026*
