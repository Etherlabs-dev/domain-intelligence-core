# IOS Risk Intelligence Core
## Project 03 — Domain Intelligence Core

Fine-tuning **Llama 3.1 8B Instruct** on 276,772 domain-specific instruction pairs to create **IOS Risk Brain #1** — a model that understands financial risk, fraud patterns, AML typologies, and regulatory language.

### Method
- **Base Model:** `unsloth/Meta-Llama-3.1-8B-Instruct`
- **Technique:** QLoRA (4-bit quantization + LoRA adapters)
- **Dataset:** `Etherlabs/ios-risk-finetune-v1` (276k instruction pairs from Project 02)
- **Compute:** Kaggle T4x2 free tier

### Project Structure
```
├── train/          ← Fine-tuning pipeline (config, dataset, model, trainer)
├── eval/           ← Domain evaluation suite + GPT-4 comparison
├── inference/      ← Clean inference API (used by Projects 4 & 5)
├── notebooks/      ← Kaggle training notebooks
├── configs/        ← Training config YAML
└── tests/          ← Unit tests
```

### Quick Start
```bash
# Local setup (for eval + inference work)
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run training (on Kaggle or GPU machine)
python -m train.run_training

# Run inference
python -m inference.predictor
```

### Output
- **HuggingFace Model:** `Etherlabs/ios-risk-llama3-v1`
- **Adapter size:** ~80MB (LoRA only)
- **Merged model:** ~16GB (full 8B, for standalone deployment)

### References
- [LoRA Paper (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314)
