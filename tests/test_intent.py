"""
Unit tests for intent taxonomy, rule heuristics, and classification models.
"""

import pytest
from src.intent import (
    ALL_INTENTS,
    rule_match_intent,
    TrivialIntentClassifier,
    TFIDFLogisticIntentClassifier,
    ProposedIntentClassifier
)


def test_intent_taxonomy():
    assert len(ALL_INTENTS) == 9
    assert "battery_power" in ALL_INTENTS
    assert "account_apple_id" in ALL_INTENTS
    assert "hardware_physical" in ALL_INTENTS
    assert "app_store_billing" in ALL_INTENTS


def test_rule_match_intent_domains():
    intent, conf = rule_match_intent("My battery is dying in 20 minutes on my iPhone!")
    assert intent == "battery_power"
    assert conf > 0.5

    intent, conf = rule_match_intent("Apple ID is locked and I cannot reset my password")
    assert intent == "account_apple_id"
    assert conf > 0.6

    intent, conf = rule_match_intent("I want a refund for the unauthorized in-app purchase")
    assert intent == "app_store_billing"
    assert conf > 0.6

    intent, conf = rule_match_intent("My phone screen is completely cracked and shattered")
    assert intent == "hardware_physical"
    assert conf > 0.6

    intent, conf = rule_match_intent("AirPods audio cuts out and speaker has no sound")
    assert intent == "audio_media"
    assert conf > 0.5


def test_intent_classifiers():
    train_texts = [
        "iPhone battery drains fast",
        "Battery percentage drops quickly",
        "Updated to iOS 11 and it is lagging",
        "iOS update causes phone freeze",
        "AirPods volume sound is not working",
        "Apple ID locked password reset"
    ]
    train_labels = [
        "battery_power",
        "battery_power",
        "ios_update_bugs",
        "ios_update_bugs",
        "audio_media",
        "account_apple_id"
    ]

    # 1. Trivial Classifier
    trivial = TrivialIntentClassifier()
    trivial.fit(train_texts, train_labels)
    preds = trivial.predict(["Some new battery problem", "Screen lag issue"])
    assert len(preds) == 2
    assert preds[0] in ALL_INTENTS

    # 2. Simple TF-IDF Classifier
    tfidf_clf = TFIDFLogisticIntentClassifier()
    tfidf_clf.fit(train_texts, train_labels)
    pred_intent, conf = tfidf_clf.predict_with_confidence("Battery dies so fast")
    assert pred_intent == "battery_power"
    assert conf > 0.2

    # 3. Proposed Classifier
    proposed = ProposedIntentClassifier(tfidf_model=tfidf_clf)
    res = proposed.predict_with_reasoning("Battery dies so fast")
    assert res["intent"] == "battery_power"
    assert "confidence" in res
    assert res["confidence"] > 0.5
