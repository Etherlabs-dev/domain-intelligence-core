"""
IOS Risk Intelligence Core — Model Loading & LoRA Adapter Setup
Uses Unsloth FastLanguageModel for memory-efficient 4-bit fine-tuning.
"""

from train.config import TrainingConfig


def load_model_and_tokenizer(config: TrainingConfig):
    """
    Loads Llama 3.1 8B in 4-bit and returns it with LoRA adapters attached.

    Two modes:
      * Fresh  — load the base model, then attach new randomly-initialised
                 LoRA adapters. This is session 1.
      * Resume — load a previously trained adapter directory. Unsloth restores
                 the base model AND the trained adapter weights, so training
                 continues from where the last session stopped. Calling
                 get_peft_model() here would wrap a second adapter around the
                 first and silently discard the training you already paid for.
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise ImportError(
            "Unsloth is required for model loading. "
            "Please run this on GPU (e.g. Kaggle/Colab) where Unsloth is installed."
        )

    resuming = bool(config.resume_from_adapter)
    source = config.resume_from_adapter if resuming else config.base_model

    if resuming:
        print(f"[Model] RESUMING from trained adapter '{source}' in 4-bit...")
    else:
        print(
            f"[Model] Loading base model '{source}' in 4-bit (max_seq_length={config.max_seq_length})..."
        )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=source,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
    )

    if resuming:
        print("[Model] Trained adapter restored — skipping fresh LoRA attachment.")
        FastLanguageModel.for_training(model)
    else:
        print(
            f"[Model] Attaching LoRA adapters (rank={config.lora_r}, alpha={config.lora_alpha})..."
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_r,
            target_modules=config.target_modules,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias=config.bias,
            use_gradient_checkpointing="unsloth",
            random_state=config.seed,
        )

    trainable_params, all_params = model.get_nb_trainable_parameters()
    trainable_pct = 100 * trainable_params / all_params
    print(
        f"[Model] Parameter Summary: {trainable_params:,} trainable / {all_params:,} total ({trainable_pct:.3f}% trainable)"
    )

    return model, tokenizer
