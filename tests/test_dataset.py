"""
Unit tests for the dataset formatting pipeline.
"""
from train.dataset import format_instruction_pair


def test_format_instruction_pair_with_input():
    sample = {
        "instruction": "Analyze this transaction sequence for AML risk.",
        "input": "Account 1234: Cash deposit of $9,500 followed by rapid wire transfer.",
        "output": "High risk: Structuring pattern detected just below reporting threshold."
    }
    result = format_instruction_pair(sample)
    formatted_text = result["text"]

    assert "### Instruction:\nAnalyze this transaction sequence for AML risk." in formatted_text
    assert "### Input:\nAccount 1234: Cash deposit of $9,500" in formatted_text
    assert "### Response:\nHigh risk: Structuring pattern" in formatted_text


def test_format_instruction_pair_without_input():
    sample = {
        "instruction": "What is the filing deadline for a SAR?",
        "input": "",
        "output": "Under FinCEN regulations, a SAR must be filed within 30 calendar days."
    }
    result = format_instruction_pair(sample)
    formatted_text = result["text"]

    assert "### Instruction:\nWhat is the filing deadline for a SAR?" in formatted_text
    assert "### Input:" not in formatted_text
    assert "### Response:\nUnder FinCEN regulations" in formatted_text
