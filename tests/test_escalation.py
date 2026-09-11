"""
Unit tests for conservative, explainable escalation policies.
"""

import pytest
from src.escalation import EscalationPolicy


def test_escalation_security_risk():
    policy = EscalationPolicy()
    evidence = [{"similarity": 0.85, "tweet_id": "1", "historical_reply": "test"}]
    
    # Apple ID lock
    res = policy.evaluate(
        customer_text="My Apple ID is locked and I cannot reset my password",
        predicted_intent="account_apple_id",
        intent_confidence=0.90,
        retrieved_evidence=evidence
    )
    assert res["decision"] == "ESCALATE"
    assert res["escalation_type"] == "ACCOUNT_SECURITY"


def test_escalation_financial_refund():
    policy = EscalationPolicy()
    evidence = [{"similarity": 0.85, "tweet_id": "1", "historical_reply": "test"}]

    res = policy.evaluate(
        customer_text="I was charged $49 without my authorization and demand a refund!",
        predicted_intent="app_store_billing",
        intent_confidence=0.90,
        retrieved_evidence=evidence
    )
    assert res["decision"] == "ESCALATE"
    assert res["escalation_type"] == "FINANCIAL_REFUND"


def test_escalation_physical_hardware():
    policy = EscalationPolicy()
    evidence = [{"similarity": 0.85, "tweet_id": "1", "historical_reply": "test"}]

    res = policy.evaluate(
        customer_text="I dropped my phone and the screen is completely shattered",
        predicted_intent="hardware_physical",
        intent_confidence=0.88,
        retrieved_evidence=evidence
    )
    assert res["decision"] == "ESCALATE"
    assert res["escalation_type"] == "PHYSICAL_HARDWARE"


def test_escalation_abusive_hostile():
    policy = EscalationPolicy()
    evidence = [{"similarity": 0.85, "tweet_id": "1", "historical_reply": "test"}]

    res = policy.evaluate(
        customer_text="Your piece of shit update broke my phone you fucking scammers",
        predicted_intent="ios_update_bugs",
        intent_confidence=0.85,
        retrieved_evidence=evidence
    )
    assert res["decision"] == "ESCALATE"
    assert res["escalation_type"] == "ABUSIVE_HOSTILE"


def test_escalation_low_confidence():
    policy = EscalationPolicy(similarity_threshold=0.55)
    # Low similarity evidence
    weak_evidence = [{"similarity": 0.32, "tweet_id": "1", "historical_reply": "test"}]

    res = policy.evaluate(
        customer_text="Can you tell me something obscure about old Mac models?",
        predicted_intent="general_inquiry",
        intent_confidence=0.60,
        retrieved_evidence=weak_evidence
    )
    assert res["decision"] == "ESCALATE"
    assert res["escalation_type"] == "LOW_CONFIDENCE"


def test_auto_handle_standard_troubleshooting():
    policy = EscalationPolicy(similarity_threshold=0.55)
    strong_evidence = [{"similarity": 0.78, "tweet_id": "1", "historical_reply": "Please check settings."}]

    res = policy.evaluate(
        customer_text="My battery seems to drain a bit faster after updating to iOS 11. Any settings to change?",
        predicted_intent="battery_power",
        intent_confidence=0.85,
        retrieved_evidence=strong_evidence
    )
    assert res["decision"] == "AUTO_HANDLE"
    assert res["escalation_type"] == "NONE"
