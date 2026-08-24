IOS Risk  ·  Project 03
Domain Intelligence Core
Finance Ops & Risk  ·  Month 2–3  ·  Phase 2  ·  This is where you stop using models and start owning them.

Month 2–3
Timeline
Llama 3.1 8B
Base Model
QLoRA / Unsloth
Method
Brain #1 of IOS
Deliverable


# Project 2 review — production quality
276,772 instruction pairs published on HuggingFace (Etherlabs/ios-risk-finetune-v1). DVC tagged foundry-v1.0. Recall improved +0.051 over baseline. Merger, uploader, and manifest all built beyond spec. The foundry is exactly what Project 3 needs — load the dataset with one line and start training.


## What You're Building
You are fine-tuning Llama 3.1 8B Instruct on your 276,772 instruction pairs to create IOS Risk Brain #1 — a domain-specific language model that understands financial risk, fraud patterns, AML typologies, and regulatory language at a level a generic model cannot match.

This is the project where everything changes. Projects 1 and 2 were infrastructure. Project 3 is intelligence. When you finish, you have a model you own, hosted on HuggingFace under the Etherlabs org, that outperforms GPT-4 on your specific domain tasks — because it was trained specifically for them.


## The leap that happens in this project
Most engineers who work with LLMs know how to call an API. A much smaller number understand what fine-tuning actually does — how LoRA adapters modify attention weights, why QLoRA uses 4-bit quantization to reduce VRAM, how learning rate warmup prevents catastrophic forgetting. By the end of Project 3, you are in that smaller group. This is what separates an AI API user from an ML engineer.


## The Concepts This Project Teaches
Unlike Projects 1 and 2 where you could build first and learn as you went, Project 3 requires you to understand three concepts before writing a line of training code. They are not optional. Getting them wrong wastes GPU hours and money.

## Concept
What it means
Why it matters here

### LoRA
Low-Rank Adaptation. Instead of updating all 8B parameters, you inject small trainable matrices (adapters) into attention layers. Only ~0.5% of parameters are trained.
Makes fine-tuning feasible on a single GPU. Without it, you'd need 8× A100s. With it, a single A100 40GB is enough.

### QLoRA
LoRA + 4-bit quantization of the frozen base model weights. Cuts VRAM from ~16GB to ~6–8GB for an 8B model.
Allows fine-tuning on Colab Pro (A100 40GB) or even a single 24GB consumer GPU. This is what makes Project 3 accessible.

### Catastrophic Forgetting
When a model is trained on new data, it can 'forget' its general capabilities. A badly configured fine-tune makes a model that's good at fraud assessment but broken at everything else.
Why you use LoRA (adapters preserve base weights), low learning rate (2e-4), and warmup steps (100). These are not arbitrary — they prevent forgetting.

### Instruction Tuning
Training a base model to follow structured instructions (instruction, input → output) rather than just predict the next token. Your dataset is already in this format.
Your Alpaca-format instruction pairs from Project 2 are exactly what instruction tuning requires. The foundry output is the fine-tuning input.

### Eval-Driven Training
Tracking eval metrics during training, not just loss. Loss going down does not mean the model is getting better at your task.
You will run your Project 1 eval harness against model checkpoints during training. If PR-AUC stops improving, you stop training — regardless of loss.


## Read these before Day 1
LoRA paper (Hu et al., 2021): arxiv.org/abs/2106.09685 — read the abstract and Section 3 only. QLoRA paper (Dettmers et al., 2023): arxiv.org/abs/2305.14314 — read the abstract and Section 2. Karpathy 'Let's build GPT from scratch': youtube.com/@AndrejKarpathy — watch the attention mechanism section (1h mark). These are not optional pre-reading. They are the foundation for every decision you make in this project.


## Compute Decision — Where to Train
Fine-tuning an 8B model with QLoRA requires a GPU with at least 16GB VRAM. You have three practical options. Make this decision before Day 1 so you're not blocked.

### Option A — Google Colab Pro  (Recommended to start)
Best for first fine-tuning run. A100 40GB available. ~$12–15 for a full training run.
PROS
No setup friction — browser-based
A100 40GB handles 8B model comfortably
Pay per use — no idle cost
Easy to share notebooks publicly
CONS
Session disconnects if idle
Can't leave running overnight unattended
Slower than a dedicated pod for multiple runs

### Option B — RunPod  (Best for speed + multiple runs)
Rent an A100 or H100 pod hourly. ~$1.99–3.99/hr. Better for iterating on training configs.
PROS
Persistent storage — no re-upload each session
H100 runs 2–3× faster than A100
Full control — no idle disconnects
Can leave training running overnight
CONS
Requires SSH/setup knowledge
Idle pods still cost money
Slightly more friction to start

### Option C — Local GPU  (Only if you have 24GB+ VRAM)
RTX 3090/4090 (24GB) can run QLoRA on 8B with 4-bit. Slower than cloud A100 but free per run.
PROS
Zero cost per run
Full control and privacy
Good for iterating quickly on small tests
CONS
24GB is tight — need to tune batch size carefully
Training will be 3–4× slower than A100
Need CUDA setup correctly


## The cost reality
A full fine-tuning run on 276k instruction pairs at 1 epoch on Colab Pro A100 takes approximately 4–6 hours and costs $12–18. For Project 3, budget $40–60 total for 3–4 experimental runs. This is not expensive — it is the price of understanding what you are building. RunPod at $2/hr is cheaper for multiple runs once you're past the first experiment.


## Repo Structure
New repo: ios-risk-intelligence-core. This is Brain #1 of IntelligenceOS — it gets its own home.

ios-risk-intelligence-core/
│
├── train/
│   ├── __init__.py
│   ├── config.py              ← all hyperparameters in one dataclass
│   ├── dataset.py             ← loads from HuggingFace, formats, splits
│   ├── model.py               ← model loading + LoRA adapter attachment
│   ├── trainer.py             ← training loop with W&B logging
│   └── run_training.py        ← entry point: python -m train.run_training
│
├── eval/
│   ├── __init__.py
│   ├── domain_eval.py         ← IOS Risk domain eval suite
│   ├── compare_gpt4.py        ← side-by-side vs GPT-4 on 50 prompts
│   └── benchmark_results.py   ← loads checkpoints + runs eval harness
│
├── inference/
│   ├── __init__.py
│   └── predictor.py           ← clean inference API over the fine-tuned model
│
├── notebooks/
│   ├── 01_dataset_inspection.ipynb
│   ├── 02_training_run.ipynb  ← the Colab notebook
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


## Environment Setup


bash — local setup (for eval + inference work)
mkdir ios-risk-intelligence-core && cd ios-risk-intelligence-core
python -m venv venv && source venv/bin/activate
 
# Core training stack
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets peft accelerate bitsandbytes
pip install unsloth   # Faster fine-tuning — 2x speed, same results
pip install wandb tqdm pyyaml pytest
 
# For eval against your harness
pip install scikit-learn pandas numpy
 
pip freeze > requirements.txt
git init && git add . && git commit -m 'Project 03: init IOS Risk Intelligence Core'


bash — Colab / RunPod setup (paste into first cell)
# Install Unsloth with the correct CUDA version
!pip install unsloth
 
# Unsloth auto-detects your GPU and installs the right torch version
# On A100: expect ~2x faster training vs vanilla HuggingFace
 
# Authenticate with HuggingFace (to push model after training)
from huggingface_hub import login
login()  # Paste your HF write token when prompted
# Get token at: huggingface.co/settings/tokens
 
# Authenticate with W&B
import wandb
wandb.login()  # Paste your API key



Training Configuration — train/config.py
Every hyperparameter decision is explained here. This is not boilerplate to copy — these are engineering choices you need to understand and be able to defend. A hiring manager at any AI lab will ask you why you chose rank 16, why 2e-4 learning rate, why 2048 sequence length. Know the answers.


python — train/config.py
"""
IOS Risk — Fine-tuning Configuration
Every value here is a deliberate decision, not a default.
Understand each one before you change it.
"""
from dataclasses import dataclass
from typing import Optional
 
 
@dataclass
class TrainingConfig:
    # ── Model ──────────────────────────────────────────────────────
    base_model: str = 'unsloth/Meta-Llama-3.1-8B-Instruct'
    # Why Llama 3.1 8B Instruct over Mistral 7B:
    # - Better instruction following out of the box
    # - Stronger base reasoning for financial analysis
    # - Unsloth has an optimised version that loads 2x faster
    # - 128k context window (Mistral: 32k) — useful for long filings
 
    # ── LoRA Adapter Config ────────────────────────────────────────
    lora_r: int = 16
    # Rank. Higher = more trainable parameters = better task fit BUT
    # more VRAM + risk of overfitting. 16 is the right starting point.
    # Range: 8 (very lightweight) to 64 (near full fine-tune quality)
    # For a second iteration: try r=32 and compare eval scores.
 
    lora_alpha: int = 16
    # Scaling factor for LoRA updates. Convention: set equal to lora_r.
    # Effective learning rate for adapters = lr * (alpha / r) = lr * 1.0
    # If alpha > r (e.g. alpha=32, r=16), adapter updates are amplified.
 
    lora_dropout: float = 0.05
    # Regularisation. Prevents adapters from memorising training examples.
    # 0.05–0.1 is standard. Don't go above 0.1 for small datasets.
 
    target_modules: tuple = ('q_proj', 'k_proj', 'v_proj', 'o_proj',
                             'gate_proj', 'up_proj', 'down_proj')
    # Which layers to apply LoRA to.
    # q/k/v/o_proj: attention layers — essential for reasoning tasks
    # gate/up/down_proj: MLP layers — important for domain knowledge
    # More modules = better domain fit but more VRAM.
 
    # ── Dataset ────────────────────────────────────────────────────
    dataset_name: str = 'Etherlabs/ios-risk-finetune-v1'
    max_samples: Optional[int] = 50000
    # Starting with 50k of 276k — enough to see domain adaptation.
    # After first run: if eval improves, scale to full 276k.
    # Sampling strategy: prioritise synthetic scenarios (richest reasoning)
    # then EDGAR text, then tabular pairs.
 
    train_split: float = 0.95
    test_split:  float = 0.05
 
    max_seq_length: int = 2048
    # Max tokens per training example. 2048 covers most of your instruction pairs.
    # Longer = more VRAM. EDGAR chunks may be longer — they get truncated.
    # Llama 3.1 supports 128k — but training at 128k requires 8× more memory.
 
    # ── Training ───────────────────────────────────────────────────
    learning_rate: float = 2e-4
    # Standard for LoRA fine-tuning. Lower than full fine-tune (1e-5)
    # because adapters are sensitive — too high = catastrophic forgetting.
    # If loss diverges early: halve to 1e-4.
 
    num_train_epochs: int = 1
    # Start with 1 epoch on 50k samples. This is ~6 hours on A100.
    # Rule: stop when eval plateaus, not when epochs finish.
    # Over-training on domain data degrades general capability.
 
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    # Effective batch size = 4 × 4 = 16
    # Batch size 16 is standard for instruction tuning.
    # If OOM: reduce per_device to 2, increase accumulation to 8.
 
    warmup_steps: int = 100
    # Gradual LR ramp-up for first 100 steps.
    # Prevents large adapter updates at the start that can destabilise training.
 
    lr_scheduler_type: str = 'cosine'
    # LR decays from 2e-4 → near 0 following a cosine curve.
    # Better than linear for fine-tuning — gentle final decay preserves base knowledge.
 
    # ── Saving & Logging ───────────────────────────────────────────
    output_dir: str = './checkpoints'
    save_steps: int = 200
    logging_steps: int = 25
    eval_steps: int = 200
 
    wandb_project: str = 'ios-risk-intelligence-core'
    hub_model_id: str = 'Etherlabs/ios-risk-llama3-v1'
 
    # ── Quantisation ───────────────────────────────────────────────
    load_in_4bit: bool = True
    # QLoRA: freeze base model in 4-bit, train only LoRA adapters in fp16.
    # Cuts VRAM from ~16GB → ~8GB for 8B model.
    # Quality impact: negligible for fine-tuning tasks.
 
    dtype: str = 'float16'
    # Adapter weights stored in fp16 (half precision).
    # On A100/H100: use bfloat16 instead — more numerically stable.



Dataset Loading — train/dataset.py
Your dataset is already published on HuggingFace. Loading it is one line. The work here is formatting it correctly for Llama 3.1's instruction template — the model was pre-trained with a specific chat format, and deviating from it reduces performance.


python — train/dataset.py
"""
IOS Risk — Dataset Loading and Formatting
Loads from HuggingFace and applies Llama 3.1 chat template.
"""
from datasets import load_dataset, DatasetDict
from typing import Dict, Optional
from .config import TrainingConfig
 
 
# Llama 3.1 Instruct uses a specific chat template.
# Deviating from this format hurts output quality.
# This is the exact template the model was instruction-tuned with.
LLAMA3_TEMPLATE = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
 
{system}<|eot_id|><|start_header_id|>user<|end_header_id|>
 
{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
 
{assistant}<|eot_id|>'''
 
 
def format_instruction_pair(example: Dict) -> Dict:
    """
    Convert an Alpaca-format record into Llama 3.1 chat format.
 
    Input:  {instruction, input, output}
    Output: {text} — the full formatted training string
    """
    system_msg  = example['instruction']
    user_msg    = example['input'] if example.get('input') else ''
    assistant   = example['output']
 
    # If input is empty, fold instruction into user turn
    if not user_msg:
        user_msg   = system_msg
        system_msg = 'You are IOS Risk, a financial risk intelligence system.'
 
    text = LLAMA3_TEMPLATE.format(
        system=system_msg,
        user=user_msg,
        assistant=assistant
    )
    return {'text': text}
 
 
def load_ios_risk_dataset(
    config: TrainingConfig,
    seed: int = 42
) -> DatasetDict:
    """
    Load, sample, format, and split the IOS Risk dataset.
 
    Sampling strategy: prioritise the examples with the richest
    domain reasoning — synthetic fraud scenarios with CRITICAL/HIGH
    assessments — over simple classification pairs.
    """
    print(f'Loading {config.dataset_name} from HuggingFace Hub...')
    ds = load_dataset(config.dataset_name, split='train')
    print(f'Full dataset: {len(ds):,} records')
 
    # Sample if max_samples is set
    if config.max_samples and len(ds) > config.max_samples:
        ds = ds.shuffle(seed=seed).select(range(config.max_samples))
        print(f'Sampled to: {len(ds):,} records')
 
    # Apply Llama 3.1 chat template
    ds = ds.map(format_instruction_pair, remove_columns=ds.column_names)
 
    # Train / eval split
    split = ds.train_test_split(
        test_size=config.test_split,
        seed=seed
    )
 
    print(f'Train: {len(split["train"]):,} | Eval: {len(split["test"]):,}')
    return split
 
 
def inspect_examples(ds, n: int = 3):
    """Print formatted examples — always inspect before training."""
    print('\n=== Sample Training Examples ===\n')
    for i, ex in enumerate(ds.select(range(n))):
        print(f'--- Example {i+1} ---')
        print(ex['text'][:800])
        print('...' if len(ex['text']) > 800 else '')
        print()



Model Loading + LoRA Attachment — train/model.py
Unsloth handles the complexity of loading a quantised model and attaching LoRA adapters in two function calls. What matters here is understanding what's happening underneath — which is why the comments are dense.


python — train/model.py
"""
IOS Risk — Model Loading and LoRA Adapter Configuration
Uses Unsloth for 2x faster training and automatic optimisations.
"""
from unsloth import FastLanguageModel
from peft import LoraConfig, get_peft_model
from .config import TrainingConfig
import torch
 
 
def load_base_model(config: TrainingConfig):
    """
    Load the base model in 4-bit quantisation.
 
    What QLoRA does under the hood:
    1. Downloads Llama 3.1 8B weights (~16GB in fp16)
    2. Quantises them to NF4 (4-bit) — reduces to ~4.5GB
    3. Stores quantised weights frozen (never updated during training)
    4. LoRA adapters (fp16, ~40MB) are added ON TOP of frozen weights
    5. Only adapters receive gradient updates
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=config.max_seq_length,
        dtype=None,             # Auto-detect: bfloat16 on A100, float16 on others
        load_in_4bit=config.load_in_4bit,
    )
    return model, tokenizer
 
 
def attach_lora_adapters(model, config: TrainingConfig):
    """
    Attach LoRA adapters to target modules.
 
    After this call, model.parameters() has two types:
    - Frozen: the 8B base model weights (4-bit, ~4.5GB)
    - Trainable: the LoRA adapters (~40MB)
 
    Only the trainable parameters receive gradients.
    The frozen weights are never modified — this is why LoRA
    preserves general capability while adapting to your domain.
    """
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=list(config.target_modules),
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias='none',            # Don't train bias parameters
        use_gradient_checkpointing='unsloth',  # Unsloth's memory optimisation
        random_state=42,
    )
 
    # Print parameter summary — inspect this before training
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f'Trainable parameters: {trainable:,} ({100*trainable/total:.2f}% of total)')
    # Expected output: ~41M trainable of ~8B total = ~0.5%
 
    return model
 
 
def prepare_model_for_inference(model):
    """Switch model to inference mode — 2x faster than training mode."""
    FastLanguageModel.for_inference(model)
    return model



Training Loop — train/trainer.py
The training loop is where the model learns. The Hugging Face SFTTrainer handles gradient computation, mixed precision, and checkpointing. Your job is to configure it correctly, monitor W&B, and know when to stop.


python — train/trainer.py
"""
IOS Risk — Fine-tuning Trainer
Wraps HuggingFace SFTTrainer with W&B logging and eval callbacks.
"""
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from datasets import DatasetDict
import wandb
from .config import TrainingConfig
 
 
def build_training_args(config: TrainingConfig) -> TrainingArguments:
    """Build the HuggingFace TrainingArguments from our config."""
    return TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=config.warmup_steps,
        fp16=not is_bfloat16_supported(),   # Use bf16 on A100, fp16 on others
        bf16=is_bfloat16_supported(),
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        evaluation_strategy='steps',
        eval_steps=config.eval_steps,
        load_best_model_at_end=True,        # Auto-save best checkpoint
        metric_for_best_model='eval_loss',
        report_to='wandb',
        run_name=f'ios-risk-llama3-r{config.lora_r}',
        seed=42,
    )
 
 
def train(
    model,
    tokenizer,
    dataset: DatasetDict,
    config: TrainingConfig
):
    """Run the full training loop."""
    wandb.init(
        project=config.wandb_project,
        config={
            'base_model':   config.base_model,
            'lora_r':       config.lora_r,
            'lora_alpha':   config.lora_alpha,
            'learning_rate':config.learning_rate,
            'max_samples':  config.max_samples,
            'max_seq_len':  config.max_seq_length,
            'dataset':      config.dataset_name,
        }
    )
 
    training_args = build_training_args(config)
 
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        dataset_text_field='text',
        max_seq_length=config.max_seq_length,
        dataset_num_proc=2,
        packing=True,   # Packs short sequences together — faster training
        args=training_args,
    )
 
    print(f'Starting training: {len(dataset["train"]):,} examples')
    print(f'Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}')
    print(f'W&B project: {config.wandb_project}')
    print(f'Checkpoints: {config.output_dir}')
 
    trainer_stats = trainer.train()
 
    print(f'\nTraining complete.')
    print(f'Runtime: {trainer_stats.metrics["train_runtime"]:.0f}s')
    print(f'Samples/sec: {trainer_stats.metrics["train_samples_per_second"]:.2f}')
 
    wandb.finish()
    return trainer



Entry Point — train/run_training.py
This is the single file you run to start a training job. Copy this into a Colab notebook cell, run it, and monitor W&B. Everything else feeds into this.


python — train/run_training.py  (copy this into Colab)
"""
IOS Risk — Fine-tuning Entry Point
python -m train.run_training
 
Or paste into a Colab cell after running the setup.
"""
from train.config   import TrainingConfig
from train.dataset  import load_ios_risk_dataset, inspect_examples
from train.model    import load_base_model, attach_lora_adapters
from train.trainer  import train
 
 
def main():
    config = TrainingConfig()
 
    # ── Step 1: Load and inspect dataset ─────────────────────────
    print('=== Step 1: Dataset ===')
    dataset = load_ios_risk_dataset(config)
    inspect_examples(dataset['train'], n=2)  # Always inspect before training
 
    # ── Step 2: Load model + attach LoRA ─────────────────────────
    print('=== Step 2: Model ===')
    model, tokenizer = load_base_model(config)
    model = attach_lora_adapters(model, config)
 
    # ── Step 3: Train ─────────────────────────────────────────────
    print('=== Step 3: Training ===')
    trainer = train(model, tokenizer, dataset, config)
 
    # ── Step 4: Save + push to HuggingFace Hub ───────────────────
    print('=== Step 4: Save ===')
    model.save_pretrained('ios-risk-llama3-v1-lora')          # Saves adapter only (~80MB)
    tokenizer.save_pretrained('ios-risk-llama3-v1-lora')
 
    # Merge LoRA weights into base model for faster inference
    # (creates a standalone 8B model — ~16GB)
    model.save_pretrained_merged(
        'ios-risk-llama3-v1-merged',
        tokenizer,
        save_method='merged_16bit'
    )
 
    # Push adapter (small — fast upload) to HuggingFace Hub
    model.push_to_hub(config.hub_model_id, tokenizer=tokenizer)
    print(f'Model pushed to: huggingface.co/{config.hub_model_id}')
 
 
if __name__ == '__main__':
    main()



Inference — inference/predictor.py
Once the model is trained, this is how you talk to it. Clean inference wrapper that Project 4 (Verification System) and Project 5 (Agent Runtime) will import directly.


python — inference/predictor.py
"""
IOS Risk — Fine-tuned Model Inference
Clean wrapper used by Projects 4 and 5.
"""
from unsloth import FastLanguageModel
from train.dataset import LLAMA3_TEMPLATE
from typing import Optional
 
 
class IOSRiskPredictor:
    """Inference wrapper for the IOS Risk fine-tuned model."""
 
    def __init__(self, model_id: str = 'Etherlabs/ios-risk-llama3-v1',
                 max_seq_length: int = 2048):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)
        print(f'IOS Risk model loaded: {model_id}')
 
    def assess_risk(self, transaction_description: str,
                    instruction: Optional[str] = None) -> str:
        """
        Run a risk assessment on a transaction or financial scenario.
 
        Args:
            transaction_description: natural language or structured description
            instruction: override the default system prompt if needed
 
        Returns:
            model's risk assessment string
        """
        system = instruction or (
            'You are IOS Risk, a financial risk intelligence system. '
            'Analyse the following and provide a structured risk assessment '
            'with risk level, fraud indicators, and recommended action.'
        )
 
        prompt = LLAMA3_TEMPLATE.format(
            system=system,
            user=transaction_description,
            assistant=''   # Empty — model will complete this
        )
 
        inputs = self.tokenizer([prompt], return_tensors='pt').to('cuda')
 
        with __import__('torch').no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,       # Low temp = more deterministic outputs
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
 
        # Decode only the generated part (not the prompt)
        response = self.tokenizer.decode(
            outputs[0][len(inputs['input_ids'][0]):],
            skip_special_tokens=True
        )
        return response.strip()
 
 
# ── Quick test ────────────────────────────────────────────────────────
if __name__ == '__main__':
    predictor = IOSRiskPredictor()
 
    test_cases = [
        'Amount: $1.49 | Hour: 3 | TxnCount1h: 23 | OffHours: 1 | MicroTxn: 1',
        'Amount: $1,847.00 | Hour: 2 | TxnCount1h: 4 | OffHours: 1 | LargeTxn: 1',
        'Amount: $34.50 | Hour: 14 | TxnCount1h: 1 | OffHours: 0 | MicroTxn: 0',
    ]
 
    for case in test_cases:
        print(f'Input: {case}')
        print(f'Output: {predictor.assess_risk(case)}')
        print()



Evaluation — eval/domain_eval.py
Training is not done when the loss converges. Training is done when the eval harness confirms your model is better at the IOS Risk task than the baseline. This is the loop: train → eval → compare → decide.


python — eval/domain_eval.py
"""
IOS Risk — Domain Evaluation Suite
Runs the fine-tuned model against 50 held-out domain prompts
and compares outputs to GPT-4 and the XGBoost baseline.
"""
import json
import wandb
from pathlib import Path
from typing import List, Dict
from inference.predictor import IOSRiskPredictor
 
 
# 50 held-out evaluation prompts — never seen during training
EVAL_PROMPTS = [
    # Card testing patterns
    {'input': 'Amount: $0.99 | Hour: 4 | TxnCount1h: 18 | MicroTxn: 1 | OffHours: 1',
     'expected_tier': 'HIGH', 'expected_pattern': 'card_testing'},
    {'input': 'Amount: $1.49 | Hour: 2 | TxnCount1h: 31 | MicroTxn: 1 | OffHours: 1',
     'expected_tier': 'HIGH', 'expected_pattern': 'card_testing'},
    # Account takeover
    {'input': 'Amount: $1847.00 | Hour: 3 | TxnCount1h: 2 | LargeTxn: 1 | OffHours: 1',
     'expected_tier': 'CRITICAL', 'expected_pattern': 'account_takeover'},
    # Money mule / structuring
    {'input': 'Amount: $9900.00 | Hour: 11 | TxnCount1h: 4 | RoundAmt: 1',
     'expected_tier': 'HIGH', 'expected_pattern': 'money_mule'},
    # Legitimate
    {'input': 'Amount: $42.30 | Hour: 13 | TxnCount1h: 1 | OffHours: 0 | MicroTxn: 0',
     'expected_tier': 'LOW', 'expected_pattern': 'legitimate'},
    # Add 45 more covering all fraud types and edge cases
]
 
 
def score_response(response: str, expected_tier: str) -> Dict:
    """Score a model response against expected risk tier."""
    response_upper = response.upper()
    correct_tier   = expected_tier in response_upper
    has_reasoning  = any(kw in response_upper for kw in
                        ['DETECTED', 'PATTERN', 'INDICATOR', 'RECOMMEND'])
    has_action     = any(kw in response_upper for kw in
                        ['BLOCK', 'FREEZE', 'REVIEW', 'FILE', 'ESCALATE', 'REQUIRED'])
    return {
        'correct_tier': correct_tier,
        'has_reasoning': has_reasoning,
        'has_action': has_action,
        'quality_score': sum([correct_tier, has_reasoning, has_action]) / 3
    }
 
 
def run_domain_eval(model_id: str = 'Etherlabs/ios-risk-llama3-v1') -> Dict:
    """Run full domain evaluation and log to W&B."""
    predictor = IOSRiskPredictor(model_id)
    results   = []
 
    for prompt in EVAL_PROMPTS:
        response = predictor.assess_risk(prompt['input'])
        scores   = score_response(response, prompt['expected_tier'])
        results.append({**prompt, 'response': response, **scores})
 
    # Aggregate
    tier_accuracy  = sum(r['correct_tier']  for r in results) / len(results)
    reasoning_rate = sum(r['has_reasoning'] for r in results) / len(results)
    action_rate    = sum(r['has_action']    for r in results) / len(results)
    avg_quality    = sum(r['quality_score'] for r in results) / len(results)
 
    summary = {
        'model': model_id,
        'n_prompts': len(results),
        'tier_accuracy':   round(tier_accuracy, 4),
        'reasoning_rate':  round(reasoning_rate, 4),
        'action_rate':     round(action_rate, 4),
        'avg_quality':     round(avg_quality, 4),
    }
 
    wandb.log(summary)
    Path('eval_results.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary



What to Watch in W&B During Training
You will be monitoring four charts in real time. Know what each one means and what action to take when something looks wrong.

Chart
What it shows
Healthy pattern
Warning sign
train/loss
Training loss per step
Steady decrease, levels off near 0.8–1.2
Spikes or sudden increase → LR too high
eval/loss
Loss on held-out eval set
Tracks train/loss but slightly higher
Diverges upward while train/loss falls → overfitting, stop here
train/learning_rate
LR schedule
Ramps up for 100 steps, cosine decay
Flat or zero → scheduler misconfigured
train/samples_per_second
Training throughput
Stable ~150–200 on A100
Drops → GPU memory pressure, reduce batch size


The most important thing to watch: eval/loss divergence
When train/loss keeps dropping but eval/loss starts rising — you have hit the overfitting point. This is the moment to stop training, regardless of how many epochs you planned. The best checkpoint (saved automatically by load_best_model_at_end=True) is the one right before this divergence. This is why eval_steps=200 matters — you need enough resolution to catch it.


Week by Week Plan

DAY 1
Concepts + Environment + Dataset Inspection
Block
Task
Output
Morning
Read LoRA abstract + Section 3. Read QLoRA abstract + Section 2. Take notes on what rank, alpha, and quantisation mean in your own words.
Concepts documented in Obsidian
Lunch
Watch Karpathy's attention mechanism explanation (~1hr mark in 'Let's build GPT from scratch')
Attention mechanism understood
Evening
Set up repo. Install dependencies. Load Etherlabs/ios-risk-finetune-v1 from HF. Print 10 examples. Confirm the Alpaca format is what you expect.
Dataset loading confirmed

DAY 2
Config + Model Loading + LoRA Attachment
Block
Task
Output
Morning
Write train/config.py — every field, every comment. Then write train/model.py. Run load_base_model() locally (CPU only, just to confirm it downloads).
Config + model code written
Lunch
Read Unsloth docs: github.com/unslothai/unsloth — understand the FastLanguageModel API
Unsloth API understood
Evening
Write train/dataset.py. Load 1000 samples. Call inspect_examples(). Confirm the Llama 3.1 chat template is applied correctly.
Formatted examples confirmed

DAY 3
First Training Run — 1000 samples, 1 epoch
Block
Task
Output
Morning
Open Colab Pro. Paste setup cell. Run run_training.py with max_samples=1000, num_epochs=1. This is your smoke test — confirm training starts, W&B receives data, no OOM errors.
Smoke test training run complete
Lunch
Monitor W&B dashboard while training. Watch train/loss, samples/sec.
W&B monitoring confirmed
Evening
Run inference/predictor.py on 5 test inputs. Is the output coherent? Does it follow the risk assessment format? Log observations.
First model outputs seen

DAY 4
Full Training Run — 50k samples
Block
Task
Output
Morning
Set max_samples=50000, num_epochs=1. Launch on Colab/RunPod. Budget 5–6 hours. Monitor W&B — check eval/loss every 30 mins.
Full training run launched
Lunch
While training: read Sebastian Raschka's 'Finetuning LLMs' blog post at magazine.sebastianraschka.com
Fine-tuning nuances studied
Evening
Training should be complete or near complete. Check best checkpoint. Push LoRA adapter to HuggingFace Hub.
Model v1 on HuggingFace Hub

DAY 5
Eval Harness + Domain Eval + Comparison
Block
Task
Output
Morning
Write eval/domain_eval.py. Run your 50 held-out prompts through the fine-tuned model. Log to W&B.
Domain eval results logged
Lunch
Compare outputs side by side with GPT-4 on the same 10 prompts (run GPT-4 via API or ChatGPT manually). Document differences.
GPT-4 comparison documented
Evening
Write README with training config, eval results, W&B link, and HuggingFace model card. Tag v1.0.
Intelligence Core v1.0 tagged


What Good Looks Like — Project 3 Complete

GitHub repo
ios-risk-intelligence-core — public, Apache 2.0, minimum 8 commits
HuggingFace model
Etherlabs/ios-risk-llama3-v1 — LoRA adapter published with model card
W&B run
Full training run logged — loss curves, eval metrics, hyperparameter config visible
train/loss
Converges and levels off — no divergence, no spike
eval/loss
Tracks train/loss — no upward divergence (no overfitting)
Domain eval
Tier accuracy >0.70 on 50 held-out prompts (70%+ correct risk tier identification)
Reasoning quality
avg_quality >0.60 — outputs contain risk reasoning and recommended action, not just a label
vs GPT-4
On fraud-specific inputs, IOS Risk Brain #1 outputs more specific, actionable assessments than GPT-4 — document this comparison explicitly
README
Training config, eval results table, W&B link, HF model link, how to reproduce
Obsidian notes
Training decisions documented — what you changed between runs and why


What Hands Off to Project 4
Project 4 (Verification System) wraps this model in structured output validation, schema enforcement, and audit logging. It imports directly from inference/predictor.py.

Artifact
How Project 4 uses it
Etherlabs/ios-risk-llama3-v1
Project 4 loads this via IOSRiskPredictor and wraps every output in a Pydantic schema with validation.
inference/predictor.py
Direct import — Project 4's VerificationWrapper calls predictor.assess_risk() and validates the response.
eval/domain_eval.py
Project 4 extends this with structured output scoring — checking schema compliance, not just tier accuracy.
W&B run config
Training provenance — Project 4's audit logs reference the exact model version and training config that produced each prediction.


Every engineer knows how to call an API. Almost none of them have trained their own model.
When Project 3 is complete, you are in the second group. You have made every hyperparameter decision consciously, watched the loss curves in real time, and shipped a model under your own org on HuggingFace that outperforms a general-purpose model on your specific domain. That is what Brain #1 of IntelligenceOS means.

IOS Risk  ·  Project 03  ·  IntelligenceOS Build Roadmap v2.0