"""
Unit tests for retrieval engine, indexing, and Hit@k evaluation.
"""

import pytest
import pandas as pd
import numpy as np

from src.retrieval import RetrievalEngine


def test_retrieval_engine_sparse():
    # Use sparse TF-IDF mode for ultra-fast unit testing
    ref_data = pd.DataFrame([
        {
            "brand_tweet_id": "201",
            "customer_tweet_id": "101",
            "customer_text": "My iPhone battery drains completely in one hour",
            "brand_reply_text": "Battery health is important. DM us your iOS version.",
            "intent": "battery_power"
        },
        {
            "brand_tweet_id": "202",
            "customer_tweet_id": "102",
            "customer_text": "Wi-Fi is not connecting on my iPad",
            "brand_reply_text": "Try resetting network settings in Settings > General.",
            "intent": "connectivity"
        },
        {
            "brand_tweet_id": "203",
            "customer_tweet_id": "103",
            "customer_text": "Updated to iOS 11 and everything is stuttering and freezing",
            "brand_reply_text": "We can help. What device model are you using?",
            "intent": "ios_update_bugs"
        }
    ])

    engine = RetrievalEngine(use_dense=False)
    engine.index(ref_data, text_col="customer_text")

    # Test top-1 retrieval
    results = engine.retrieve("battery dying fast", top_k=1)
    assert len(results) == 1
    assert results[0]["intent"] == "battery_power"
    assert results[0]["tweet_id"] == "201"
    assert results[0]["similarity"] > 0.1

    # Test Hit@k evaluation
    eval_df = pd.DataFrame([
        {"customer_text": "battery is dead", "intent": "battery_power"},
        {"customer_text": "wi-fi keeps dropping", "intent": "connectivity"}
    ])
    metrics = engine.evaluate_hit_at_k(eval_df, k_list=[1, 2])
    assert "hit@1" in metrics
    assert "mrr" in metrics
    assert metrics["hit@1"] > 0.0
