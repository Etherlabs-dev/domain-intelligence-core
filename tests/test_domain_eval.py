from eval.domain_eval import extract_label, extract_tier, score_risk, unsupported_claims


def test_classification_requires_exact_label():
    assert extract_label("FRAUD") == "FRAUD"
    assert extract_label("The transaction is FRAUD") is None


def test_tier_must_open_the_response():
    assert extract_tier("HIGH RISK — CARD TESTING DETECTED") == "HIGH"
    assert extract_tier("The result is HIGH RISK") is None


def test_unsupported_score_and_missing_device_are_flagged():
    response = "HIGH RISK. Probability of fraud: 89%. A new device was used."
    claims = unsupported_claims(response, "Amount: $1.00")
    assert "invented_score_or_probability" in claims
    assert "unsupported_device" in claims


def test_negative_action_text_does_not_get_generic_review_credit():
    case = {
        "expected_tier": "LOW",
        "expected_pattern": "legitimate",
        "expected_evidence": ["documented", "verified"],
        "expected_action": "no_action",
        "input": "Source: documented | Beneficiary: verified",
    }
    result = score_risk(
        "LOW RISK — LEGITIMATE. Source is documented and beneficiary verified. "
        "No action; continue monitoring.",
        case,
    )
    assert result["correct_action"]
    assert result["quality_score"] == 1.0
