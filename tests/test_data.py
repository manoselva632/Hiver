"""
Unit tests for data preprocessing, thread reconstruction, and zero-leakage splits.
"""

import pytest
import pandas as pd
import numpy as np

from src.data import clean_text, normalize_id, reconstruct_conversations, create_splits


def test_clean_text():
    raw = "  Hello &amp; welcome &lt;user&gt;   to Apple   "
    cleaned = clean_text(raw)
    assert cleaned == "Hello & welcome <user> to Apple"


def test_normalize_id():
    assert normalize_id(12345) == "12345"
    assert normalize_id("12345.0") == "12345"
    assert normalize_id(None) == ""
    assert normalize_id(np.nan) == ""
    assert normalize_id("nan") == ""


def test_reconstruct_conversations():
    df_raw = pd.DataFrame([
        {
            "tweet_id": "101",
            "author_id": "cust_1",
            "inbound": True,
            "created_at": "2017-10-10",
            "text": "@AppleSupport My battery is draining super fast on iOS 11!",
            "response_tweet_id": "102",
            "in_response_to_tweet_id": ""
        },
        {
            "tweet_id": "102",
            "author_id": "AppleSupport",
            "inbound": False,
            "created_at": "2017-10-10",
            "text": "@cust_1 We can help! Send us a DM with your iOS version.",
            "response_tweet_id": "",
            "in_response_to_tweet_id": "101"
        }
    ])
    
    pairs = reconstruct_conversations(df_raw, brand="AppleSupport")
    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert row["customer_tweet_id"] == "101"
    assert row["brand_tweet_id"] == "102"
    assert "battery is draining" in row["customer_text"]
    assert "Send us a DM" in row["brand_reply_text"]


def test_zero_leakage_splits():
    # Construct 100 sample pairs with distinct thread_ids
    data = []
    for i in range(100):
        data.append({
            "customer_tweet_id": str(1000 + i),
            "customer_author_id": f"user_{i}",
            "customer_text": f"Sample inquiry {i}",
            "brand_tweet_id": str(2000 + i),
            "brand_reply_text": f"Sample reply {i}",
            "thread_id": f"thread_{i // 2}",  # pairs share thread roots
            "created_at": "2017-10-10"
        })
    df_pairs = pd.DataFrame(data)

    ref_df, train_df, eval_df = create_splits(
        df_pairs, ref_ratio=0.70, train_ratio=0.15, eval_ratio=0.15, random_seed=42
    )

    # Check non-empty
    assert len(ref_df) > 0
    assert len(train_df) > 0
    assert len(eval_df) > 0
    assert len(ref_df) + len(train_df) + len(eval_df) == len(df_pairs)

    # Check strict thread-level disjointness (zero leakage)
    ref_threads = set(ref_df["thread_id"])
    train_threads = set(train_df["thread_id"])
    eval_threads = set(eval_df["thread_id"])

    assert len(ref_threads.intersection(train_threads)) == 0, "Leakage between Reference and Train!"
    assert len(ref_threads.intersection(eval_threads)) == 0, "Leakage between Reference and Eval!"
    assert len(train_threads.intersection(eval_threads)) == 0, "Leakage between Train and Eval!"
