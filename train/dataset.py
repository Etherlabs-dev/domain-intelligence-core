"""
IOS Risk Intelligence Core — Dataset Pipeline
Loads, formats, and splits domain-specific instruction datasets.
"""

from collections import Counter
from typing import Dict, Tuple
from datasets import Dataset, load_dataset
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


def validate_raw_dataset(dataset: Dataset, config: TrainingConfig) -> dict:
    """Fail fast on the defects that made the v1 training run worthless."""
    required = {"instruction", "input", "output"}
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    instructions = Counter(str(value).strip() for value in dataset["instruction"])
    outputs = Counter(str(value).strip() for value in dataset["output"])
    empty_instructions = instructions.get("", 0)
    empty_outputs = outputs.get("", 0)
    if empty_instructions or empty_outputs:
        raise ValueError(
            f"Dataset contains empty targets: instruction={empty_instructions}, "
            f"output={empty_outputs}"
        )
    if len(instructions) < config.min_unique_instructions:
        raise ValueError(
            f"Only {len(instructions)} unique instructions; minimum is "
            f"{config.min_unique_instructions}. Refusing to train."
        )
    if len(outputs) < config.min_unique_outputs:
        raise ValueError(
            f"Only {len(outputs)} unique outputs; minimum is "
            f"{config.min_unique_outputs}. Refusing to train."
        )

    summary = {
        "records": len(dataset),
        "unique_instructions": len(instructions),
        "unique_outputs": len(outputs),
        "top_outputs": outputs.most_common(5),
    }
    print(f"[Dataset] Quality gate passed: {summary}")
    return summary


def append_eos_token(dataset: Dataset, eos_token: str) -> Dataset:
    """Make answer boundaries explicit and idempotent before tokenization."""
    if not eos_token:
        raise ValueError("Tokenizer has no eos_token; training cannot proceed safely")

    def add_eos(example: Dict[str, str]) -> Dict[str, str]:
        text = example["text"]
        return {"text": text if text.endswith(eos_token) else text + eos_token}

    return dataset.map(add_eos, desc="Appending EOS")


def verify_tokenized_eos(trainer, tokenizer) -> None:
    """Prove the actual trainer input ends at EOS before spending GPU time."""
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id")
    ids = trainer.train_dataset[0].get("input_ids")
    if not ids:
        raise ValueError(
            "Trainer did not expose tokenized input_ids for EOS verification"
        )
    if ids[-1] != eos_id:
        raise ValueError(
            f"EOS preflight failed: final token is {ids[-1]}, expected {eos_id}. "
            "Training was stopped before the first optimizer step."
        )
    print(f"[Preflight] EOS verified on tokenized training data (token id {eos_id}).")


def load_and_prepare_dataset(config: TrainingConfig) -> Tuple[Dataset, Dataset]:
    """
    Loads raw instruction pairs from HuggingFace, formats them into training
    prompts, and returns a (train, eval) pair.

    The final `val_size` shuffled rows monitor optimization loss and are not
    trained. This is separate from the frozen Project 03 domain evaluation,
    whose records and regulatory sections are excluded upstream in Foundry v3.

    `sample_offset` and `max_samples` remain available for smoke tests, but the
    release notebook trains the 20,606-row v3 dataset in one fresh run.
    """
    print(f"[Dataset] Loading dataset '{config.dataset_name}' from HuggingFace...")
    dataset = load_dataset(config.dataset_name, split="train")
    total = len(dataset)
    validate_raw_dataset(dataset, config)

    # Held-out tail — identical in every session.
    val_start = total - config.val_size
    if val_start <= 0:
        raise ValueError(f"val_size={config.val_size} is >= dataset size {total}")
    eval_raw = dataset.select(range(val_start, total))

    # Training slice walks forward through everything before the val tail.
    start = config.sample_offset
    if start >= val_start:
        raise ValueError(
            f"sample_offset={start} is past the training pool (ends at {val_start}). "
            "The dataset has been fully covered."
        )
    end = (
        val_start
        if config.max_samples is None
        else min(start + config.max_samples, val_start)
    )
    train_raw = dataset.select(range(start, end))

    print(
        f"[Dataset] Train slice: rows [{start:,}, {end:,}) of {val_start:,} available"
    )
    print(
        f"[Dataset] Val set:     rows [{val_start:,}, {total:,}) — fixed across sessions"
    )

    print("[Dataset] Formatting instruction pairs into Alpaca format...")
    train_data = train_raw.map(format_instruction_pair, desc="Formatting train")
    eval_data = eval_raw.map(format_instruction_pair, desc="Formatting val")

    print(
        f"[Dataset] Ready: {len(train_data):,} train samples, {len(eval_data):,} validation samples."
    )
    return train_data, eval_data
