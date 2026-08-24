"""
IOS Risk Intelligence Core — Dataset Pipeline
Loads, formats, and splits domain-specific instruction datasets.
"""
from typing import Dict, Optional, Tuple
from datasets import Dataset, DatasetDict, load_dataset
from train.config import TrainingConfig


# The standard Alpaca/Llama prompt template for instruction tuning
ALPACA_PROMPT = """Below is an instruction that describes a financial risk analysis task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

ALPACA_NO_INPUT_PROMPT = """Below is an instruction that describes a financial risk analysis task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""


def format_instruction_pair(example: Dict[str, str]) -> Dict[str, str]:
    """
    Formats a single dataset row into the standard prompt structure.
    Works whether an 'input' context field is present or empty.
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text and input_text.strip():
        text = ALPACA_PROMPT.format(instruction, input_text, output)
    else:
        text = ALPACA_NO_INPUT_PROMPT.format(instruction, output)

    return {"text": text}


def load_and_prepare_dataset(config: TrainingConfig) -> Tuple[Dataset, Dataset]:
    """
    Loads raw instruction pairs from HuggingFace, formats them into training
    prompts, and returns a (train, eval) pair.

    Multi-session aware. Kaggle caps a session at 9 hours, so the 276k dataset
    is covered across many sessions. Two invariants make that work:

      * The validation set is the LAST `val_size` rows, every session. Fixed,
        never trained on, so eval loss is comparable run to run.
      * The training slice is [sample_offset, sample_offset + max_samples),
        so each session advances through data it has not seen.
    """
    print(f"[Dataset] Loading dataset '{config.dataset_name}' from HuggingFace...")
    dataset = load_dataset(config.dataset_name, split="train")
    total = len(dataset)

    # Held-out tail — identical in every session.
    val_start = total - config.val_size
    if val_start <= 0:
        raise ValueError(
            f"val_size={config.val_size} is >= dataset size {total}"
        )
    eval_raw = dataset.select(range(val_start, total))

    # Training slice walks forward through everything before the val tail.
    start = config.sample_offset
    if start >= val_start:
        raise ValueError(
            f"sample_offset={start} is past the training pool (ends at {val_start}). "
            "The dataset has been fully covered."
        )
    end = val_start if config.max_samples is None else min(start + config.max_samples, val_start)
    train_raw = dataset.select(range(start, end))

    print(f"[Dataset] Train slice: rows [{start:,}, {end:,}) of {val_start:,} available")
    print(f"[Dataset] Val set:     rows [{val_start:,}, {total:,}) — fixed across sessions")

    print("[Dataset] Formatting instruction pairs into Alpaca format...")
    train_data = train_raw.map(format_instruction_pair, desc="Formatting train")
    eval_data = eval_raw.map(format_instruction_pair, desc="Formatting val")

    print(f"[Dataset] Ready: {len(train_data):,} train samples, {len(eval_data):,} validation samples.")
    return train_data, eval_data
