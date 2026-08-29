"""
IOS Risk Intelligence Core — Fine-Tuning Configuration
Every parameter is a deliberate engineering decision.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainingConfig:
    # ── Model & Quantization ─────────────────────────────────────────
    base_model: str = "unsloth/Meta-Llama-3.1-8B-Instruct"
    max_seq_length: int = 1024
    load_in_4bit: bool = True
    resume_from_adapter: Optional[str] = None
    # Path to a previous session's adapter. None = fresh LoRA (session 1).

    # ── LoRA Adapter Configuration ───────────────────────────────────
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    bias: str = "none"

    # ── Dataset ──────────────────────────────────────────────────────
    dataset_name: str = "Etherlabs/ios-risk-finetune-v3"
    max_samples: Optional[int] = None  # v3 is 20,606 rows; one session covers it
    sample_offset: int = 0
    # Sessions walk forward through the dataset: session 1 takes rows
    # [0, 20000), session 2 [20000, 40000), and so on. Without this every
    # session retrains on the same head of the data.

    val_size: int = 1000
    # A FIXED held-out set taken from the tail of the dataset, identical in
    # every session. A random per-slice split would make eval loss
    # incomparable between sessions, which defeats tracking it.

    # ── Training & Optimization ──────────────────────────────────────
    num_train_epochs: float = 1.0
    # One pass over the data. TRL's default is 3, which silently triples both
    # runtime and overfitting risk on a large instruction set.
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8  # 2 * 8 = effective batch size of 16
    warmup_steps: int = 100
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    optim: str = "adamw_8bit"
    seed: int = 42

    # Dataset quality gates. These run before trainer construction so a bad
    # export cannot consume GPU time merely because it is loadable.
    min_unique_instructions: int = 3
    min_unique_outputs: int = 100

    # ── Checkpointing & Saving ───────────────────────────────────────
    output_dir: str = "./outputs/Llama-3.1-8B-IOS-Risk-v1"
    save_strategy: str = "steps"
    save_steps: int = 250
    eval_steps: int = 250
    # Cadence is per-run, not universal: eval over a 5% split of 20k samples is
    # 1,000 forward passes. At save/eval every 50 steps that overhead dominates.
    save_total_limit: int = 3

    # ── Logging & Monitoring ─────────────────────────────────────────
    logging_steps: int = 10
    report_to: str = "wandb"
    wandb_project: str = "ios-risk-domain-core"
    wandb_run_name: Optional[str] = "llama3-8b-ios-risk-v1"

    # ── Helper Method ────────────────────────────────────────────────
    def to_dict(self) -> dict:
        """Converts configuration into a dictionary for logging and saving."""
        import dataclasses

        return dataclasses.asdict(self)
