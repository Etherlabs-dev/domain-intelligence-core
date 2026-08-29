"""
Unit tests for the dataset formatting pipeline.
"""

import pytest
from datasets import Dataset

from train.config import TrainingConfig
from train.dataset import (
    append_eos_token,
    format_instruction_pair,
    validate_raw_dataset,
    verify_tokenized_eos,
)


def test_format_instruction_pair_with_input():
    sample = {
        "instruction": "Analyze this transaction sequence for AML risk.",
        "input": "Account 1234: Cash deposit of $9,500 followed by rapid wire transfer.",
        "output": "High risk: Structuring pattern detected just below reporting threshold.",
    }
    result = format_instruction_pair(sample)
    formatted_text = result["text"]

    assert (
        "### Instruction:\nAnalyze this transaction sequence for AML risk."
        in formatted_text
    )
    assert "### Input:\nAccount 1234: Cash deposit of $9,500" in formatted_text
    assert "### Response:\nHigh risk: Structuring pattern" in formatted_text


def test_format_instruction_pair_without_input():
    sample = {
        "instruction": "What is the filing deadline for a SAR?",
        "input": "",
        "output": "Under FinCEN regulations, a SAR must be filed within 30 calendar days.",
    }
    result = format_instruction_pair(sample)
    formatted_text = result["text"]

    assert "### Instruction:\nWhat is the filing deadline for a SAR?" in formatted_text
    assert "### Input:" not in formatted_text
    assert "### Response:\nUnder FinCEN regulations" in formatted_text


def test_append_eos_token_is_idempotent():
    ds = Dataset.from_list([{"text": "answer"}, {"text": "answer<eos>"}])
    result = append_eos_token(ds, "<eos>")
    assert result["text"] == ["answer<eos>", "answer<eos>"]


def test_quality_gate_rejects_v1_shape():
    ds = Dataset.from_list(
        [
            {"instruction": "same", "input": "a", "output": "LEGITIMATE"},
            {"instruction": "same", "input": "b", "output": "FRAUD"},
        ]
    )
    config = TrainingConfig(min_unique_instructions=3, min_unique_outputs=3)
    with pytest.raises(ValueError, match="unique instructions"):
        validate_raw_dataset(ds, config)


class _Tokenizer:
    eos_token_id = 2


class _Trainer:
    def __init__(self, ids):
        self.train_dataset = [{"input_ids": ids}]


def test_tokenized_eos_preflight_accepts_real_eos_boundary():
    verify_tokenized_eos(_Trainer([7, 8, 2]), _Tokenizer())


def test_tokenized_eos_preflight_stops_before_training():
    with pytest.raises(ValueError, match="EOS preflight failed"):
        verify_tokenized_eos(_Trainer([7, 8, 9]), _Tokenizer())
