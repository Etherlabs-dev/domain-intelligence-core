# Project 03 — Domain Intelligence Core
## Learning Approach & Context for Any Agent or IDE

---

## What This Project Is

This is **IOS Risk Project 03: Domain Intelligence Core** — the fine-tuning phase of the IntelligenceOS build roadmap.

The goal is to fine-tune **Llama 3.1 8B Instruct** on **276,772 domain-specific instruction pairs** (published on HuggingFace as `Etherlabs/ios-risk-finetune-v1` from Project 02) using **QLoRA + Unsloth** to produce **IOS Risk Brain #1**: a domain-specific LLM that understands financial risk, fraud patterns, AML typologies, and regulatory language at a level no general-purpose model can match.

The output model will be published to HuggingFace under the Etherlabs org as `Etherlabs/ios-risk-llama3-v1`.

**This project is not about using AI tools. It is about building one.**

---

## Compute Strategy — Zero Spend

Training runs on **Kaggle's free GPU tier** (T4x2 = 32GB VRAM combined, 30 hrs/week free).

No paid compute is being used for this project. The T4-adjusted training config is:

```python
per_device_train_batch_size: 2
gradient_accumulation_steps: 8   # keeps effective batch size at 16
max_seq_length: 1024
max_samples: 20000               # per Kaggle session (max 9hrs)
```

Kaggle sessions save checkpoints to `/kaggle/working/` and training resumes across sessions until the full dataset is covered.

Full context on why this works without quality loss is in `build_approach_cost.md`.

---

## The Learning Approach — Non-Negotiable

**The owner of this project is learning to master fine-tuning, not just complete the project.**

Any agent or collaborator working in this context must follow this approach exactly:

### 1. One concept or one function at a time
Do not write entire files. Do not write multiple functions in a single response. The pattern is:
- Explain the concept (why it exists, what problem it solves)
- Explain the design decision (why this specific choice was made)
- Show or guide the code for that single piece
- **Wait for confirmation before moving to the next piece**

### 2. Concepts before code — always
The project plan is explicit: *"Project 3 requires you to understand three concepts before writing a line of training code. Getting them wrong wastes GPU hours."*

The three mandatory concepts are:
- **LoRA** — Low-Rank Adaptation: what rank means, why it preserves base model knowledge
- **QLoRA** — 4-bit quantization: how it reduces VRAM, why quality is not affected
- **Catastrophic Forgetting** — what causes it, why low learning rate + warmup prevents it

No training code is written until these are understood and documented in the owner's own words.

### 3. The owner writes the code
An agent's job is to guide and explain — not to produce finished code for the owner to paste. If the owner is stuck, give a hint or explain the concept differently. Do not solve it for them unless explicitly asked and the specific piece is purely mechanical (imports, boilerplate).

### 4. Answers must be defensible
Every hyperparameter in `train/config.py` must be something the owner can explain from memory to a hiring manager. If they cannot explain why `lora_r=16` or why `learning_rate=2e-4`, we have not done the teaching step yet.

### 5. No full file dumps
Even if asked "can you write the whole file," the correct response is to write one function, explain it, and wait. The exception is when the owner explicitly says "I understand all of this, just generate the boilerplate" for a non-conceptual piece (e.g., `__init__.py`).

---

## Project Structure Being Built

```
ios-risk-intelligence-core/
│
├── train/
│   ├── __init__.py
│   ├── config.py          ← all hyperparameters in one dataclass
│   ├── dataset.py         ← loads from HuggingFace, formats, splits
│   ├── model.py           ← model loading + LoRA adapter attachment
│   ├── trainer.py         ← training loop with W&B logging
│   └── run_training.py    ← entry point
│
├── eval/
│   ├── __init__.py
│   ├── domain_eval.py     ← 50 held-out domain prompts eval suite
│   ├── compare_gpt4.py    ← side-by-side vs GPT-4 on 50 prompts
│   └── benchmark_results.py
│
├── inference/
│   ├── __init__.py
│   └── predictor.py       ← clean inference API (used by Projects 4 & 5)
│
├── notebooks/
│   ├── 01_dataset_inspection.ipynb
│   ├── 02_training_run.ipynb    ← the Kaggle notebook
│   └── 03_eval_results.ipynb
│
├── configs/
│   └── training_config.yaml
│
├── tests/
│   ├── test_dataset.py
│   └── test_inference.py
│
├── requirements.txt
└── README.md
```

---

## Current Progress

| Stage | Status | Notes |
|---|---|---|
| Conceptual foundation (LoRA, QLoRA, forgetting) | ✅ Complete | Day 1 — all three concepts understood and explained in own words |
| Repo skeleton + config | ✅ Complete | train/ (config, dataset, model, trainer, run_training), tests, configs/yaml |
| Smoke test (2000 samples on Kaggle) | Not started | |
| Full training run (20k samples/session) | Not started | |
| Eval harness + domain eval | Not started | |
| HuggingFace push | Not started | |

**Update this table as stages complete.**

---

## Key References

- **Project plan:** `IOS_Risk_Project03_DomainIntelligenceCore.docx` (in this directory)
- **Compute strategy:** `build_approach_cost.md` (in this directory)
- **Dataset:** `Etherlabs/ios-risk-finetune-v1` on HuggingFace
- **LoRA paper:** arxiv.org/abs/2106.09685 — read abstract + Section 3
- **QLoRA paper:** arxiv.org/abs/2305.14314 — read abstract + Section 2
- **Karpathy attention walkthrough:** youtube.com/@AndrejKarpathy — "Let's build GPT from scratch", ~1hr mark

---

## What Hands Off to Project 04

Project 04 (Verification System) imports directly from `inference/predictor.py`. The fine-tuned model must be on HuggingFace and the `IOSRiskPredictor` class must be clean and importable before Project 04 can begin.

---

*This file is the source of truth for any agent, IDE, or collaborator entering this project mid-stream. Read it before doing anything else.*
