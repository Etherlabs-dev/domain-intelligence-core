"""
IOS Risk Intelligence Core — SFT Training Pipeline
Integrates Unsloth, TRL SFTTrainer, and Weights & Biases tracking.
"""
import dataclasses
import inspect
import os
from typing import Optional

from trl import SFTConfig, SFTTrainer

from train.config import TrainingConfig
from train.dataset import load_and_prepare_dataset
from train.model import load_model_and_tokenizer


def build_sft_config(config: TrainingConfig, has_eval: bool) -> SFTConfig:
    """
    Builds the TRL SFTConfig.

    TRL moved the dataset and sequence-length knobs off SFTTrainer and onto
    SFTConfig, and renamed some of them along the way (`max_seq_length` ->
    `max_length`, `evaluation_strategy` -> `eval_strategy`). Kaggle resolves
    whatever TRL version Unsloth pulls in, so reconcile against the installed
    dataclass instead of hardcoding one version's spelling.
    """
    fields = {f.name for f in dataclasses.fields(SFTConfig)}

    kwargs = dict(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        optim=config.optim,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        report_to=config.report_to if config.report_to else "none",
        run_name=config.wandb_run_name,
        seed=config.seed,
        dataset_text_field="text",
        dataset_num_proc=2,
        packing=False,  # Faster for short sequences, but off for safety.
    )

    eval_key = "eval_strategy" if "eval_strategy" in fields else "evaluation_strategy"
    kwargs[eval_key] = "steps" if has_eval else "no"
    if has_eval:
        kwargs["eval_steps"] = config.eval_steps

    length_key = "max_length" if "max_length" in fields else "max_seq_length"
    kwargs[length_key] = config.max_seq_length

    dropped = [k for k in kwargs if k not in fields]
    for k in dropped:
        kwargs.pop(k)
    if dropped:
        print(f"[Trainer] Dropped args unsupported by this TRL version: {dropped}")

    return SFTConfig(**kwargs)


def create_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    config: TrainingConfig,
) -> SFTTrainer:
    """
    Constructs and configures the SFTTrainer with all hyperparameters.
    """
    training_args = build_sft_config(config, has_eval=eval_dataset is not None)

    # SFTTrainer renamed `tokenizer` -> `processing_class`.
    params = inspect.signature(SFTTrainer.__init__).parameters
    tokenizer_kwarg = "processing_class" if "processing_class" in params else "tokenizer"

    return SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        **{tokenizer_kwarg: tokenizer},
    )


def is_bfloat16_supported() -> bool:
    """Checks if the current GPU supports bf16 (Ampere/Ada/Hopper). T4 does not."""
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        return False


def run_training_pipeline(config: Optional[TrainingConfig] = None):
    """
    End-to-end execution: dataset loading, model setup, training, and artifact saving.
    """
    if config is None:
        config = TrainingConfig()

    print("\n" + "=" * 60)
    print("  IOS RISK INTELLIGENCE CORE — TRAINING PIPELINE")
    print("=" * 60)
    print(f"Base Model:  {config.base_model}")
    print(f"Dataset:     {config.dataset_name}")
    print(f"LoRA Rank:   {config.lora_r} (alpha={config.lora_alpha})")
    print(f"Output Dir:  {config.output_dir}\n")

    # 1. Dataset
    train_data, eval_data = load_and_prepare_dataset(config)

    # 2. Model & Tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # 3. Trainer
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=eval_data,
        config=config,
    )

    # 4. Train
    print("\n[Training] Starting training run...")
    train_result = trainer.train()

    # 5. Save Adapter Artifacts
    print(f"\n[Saving] Saving trained LoRA adapter to '{config.output_dir}'...")
    os.makedirs(config.output_dir, exist_ok=True)
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    print("[Training] Training completed successfully!")
    return train_result
