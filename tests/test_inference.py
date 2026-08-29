from inference.predictor import IOSRiskPredictor


def test_prompt_forbids_unsupported_precision():
    prompt = IOSRiskPredictor.build_prompt("Amount: $10.00 | NewDevice: 0")
    assert "Do not invent probabilities, scores" in prompt
    assert "Amount: $10.00" in prompt
    assert prompt.endswith("### Response:\n")
