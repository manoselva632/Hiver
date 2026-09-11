"""
Unit tests for structured agent JSON prediction schemas across Trivial, Simple, and Proposed systems.
"""

import pytest
import pandas as pd
from src.intent import TFIDFLogisticIntentClassifier, ProposedIntentClassifier
from src.retrieval import RetrievalEngine
from src.escalation import EscalationPolicy
from src.agent import TrivialBaselineAgent, SimpleBaselineAgent, ProposedAgent


@pytest.fixture
def mock_pipeline():
    ref_df = pd.DataFrame([
        {
            "brand_tweet_id": "201",
            "customer_tweet_id": "101",
            "customer_text": "Battery draining so fast after iOS update",
            "brand_reply_text": "Battery life is important. DM us your iOS version.",
            "intent": "battery_power"
        },
        {
            "brand_tweet_id": "202",
            "customer_tweet_id": "102",
            "customer_text": "Need help with locked Apple ID password",
            "brand_reply_text": "Please visit iforgot.apple.com to reset your password.",
            "intent": "account_apple_id"
        }
    ])

    retriever = RetrievalEngine(use_dense=False)
    retriever.index(ref_df, text_col="customer_text")

    tfidf_clf = TFIDFLogisticIntentClassifier()
    tfidf_clf.fit(ref_df["customer_text"].tolist(), ref_df["intent"].tolist())

    proposed_clf = ProposedIntentClassifier(tfidf_model=tfidf_clf)
    policy = EscalationPolicy(similarity_threshold=0.50)

    return {
        "trivial": TrivialBaselineAgent(),
        "simple": SimpleBaselineAgent(tfidf_clf, retriever),
        "proposed": ProposedAgent(proposed_clf, retriever, policy)
    }


def test_agent_structured_json_schema(mock_pipeline):
    query = "My battery drains quickly during phone calls"
    required_keys = {"intent", "reply", "decision", "escalation_reason", "evidence"}

    for name, agent in mock_pipeline.items():
        res = agent.predict(query)
        assert isinstance(res, dict), f"{name} failed to return a dict"
        assert required_keys.issubset(res.keys()), f"{name} output missing required keys: {required_keys - set(res.keys())}"
        assert res["decision"] in ["AUTO_HANDLE", "ESCALATE"]
        assert len(res["reply"].strip()) > 5
        assert isinstance(res["evidence"], list)


def test_proposed_agent_evidence_citations(mock_pipeline):
    agent = mock_pipeline["proposed"]
    res = agent.predict("My battery is dying fast")
    assert len(res["evidence"]) > 0
    citation = res["evidence"][0]
    assert "tweet_id" in citation
    assert "similarity" in citation
    assert "historical_reply" in citation
