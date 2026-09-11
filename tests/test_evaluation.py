"""
Unit tests for evaluation metrics, grounding coverage, and judge agreement math.
"""

import pytest
import pandas as pd
from src.evaluation import (
    evaluate_intent_metrics,
    evaluate_escalation_metrics,
    evaluate_reply_grounding,
    score_reply_deterministic_judge,
    calculate_judge_agreement
)


def test_intent_metrics_calculation():
    y_true = ["battery_power", "ios_update_bugs", "connectivity", "battery_power"]
    y_pred = ["battery_power", "ios_update_bugs", "battery_power", "battery_power"]

    res = evaluate_intent_metrics(y_true, y_pred)
    assert res["accuracy"] == 0.75
    assert "macro_f1" in res
    assert "per_intent" in res
    assert "battery_power" in res["per_intent"]


def test_escalation_metrics_calculation():
    y_true = ["ESCALATE", "AUTO_HANDLE", "ESCALATE", "AUTO_HANDLE"]
    y_pred = ["ESCALATE", "AUTO_HANDLE", "AUTO_HANDLE", "AUTO_HANDLE"]

    res = evaluate_escalation_metrics(y_true, y_pred)
    assert res["true_positives"] == 1
    assert res["false_negatives"] == 1
    assert res["recall"] == 0.5
    assert res["precision"] == 1.0


def test_reply_grounding():
    preds = [
        {
            "reply": "Please check your battery health in Settings > Battery.",
            "evidence": [
                {
                    "tweet_id": "1",
                    "similarity": 0.85,
                    "customer_text": "battery problem",
                    "historical_reply": "Check battery in Settings > Battery."
                }
            ]
        }
    ]
    res = evaluate_reply_grounding(preds)
    assert res["citation_coverage"] == 1.0
    assert res["mean_top_similarity"] == 0.85
    assert res["mean_lexical_grounding"] > 0.0


def test_judge_agreement_calculation():
    df_sample = pd.DataFrame({"id": range(10)})
    llm_scores = [4.5, 4.0, 3.5, 5.0, 2.0, 4.5, 4.0, 3.5, 5.0, 2.0]
    human_scores = [4.8, 4.2, 3.8, 5.0, 2.5, 4.8, 4.2, 3.8, 5.0, 2.5]
    llm_decisions = ["AUTO_HANDLE", "AUTO_HANDLE", "ESCALATE", "AUTO_HANDLE", "ESCALATE"] * 2
    human_decisions = ["AUTO_HANDLE", "AUTO_HANDLE", "ESCALATE", "AUTO_HANDLE", "ESCALATE"] * 2

    res = calculate_judge_agreement(df_sample, llm_scores, human_scores, llm_decisions, human_decisions)
    assert res["cohens_kappa_escalation"] == 1.0
    assert res["spearman_correlation_quality"] > 0.9
