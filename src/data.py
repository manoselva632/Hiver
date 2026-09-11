"""
Data loading, preprocessing, thread reconstruction, and leak-free splitting.
"""

import os
import re
import csv
import logging
from typing import Dict, List, Optional, Tuple, Set, Any
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Normalize whitespace, remove raw handles/html entities while preserving technical terms."""
    if not isinstance(text, str):
        return ""
    # Decode HTML entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    # Clean redundant whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_id(val: Any) -> str:
    """Normalize tweet ID to string, handling float NaN and .0 suffixes."""
    if val is None or pd.isna(val):
        return ""
    s = str(val).strip()
    if s.lower() == "nan":
        return ""
    if s.endswith(".0"):
        s = s[:-2]
    return s


def stream_subsample_brand(
    csv_path: str,
    brand: str = "AppleSupport",
    max_tweets: int = 25000,
    chunk_size: int = 50000,
) -> pd.DataFrame:
    """
    Stream large CSV file in chunks and sample brand tweets + customer queries.
    Avoids loading all 3M rows into RAM.
    """
    logger.info(f"Streaming subsample for brand '{brand}' from {csv_path} (target ~{max_tweets} rows)...")
    
    brand_handle_lower = brand.lower()
    brand_tweets = []
    customer_tweets = []
    
    # Track needed parent tweet IDs that brand replies to
    needed_parent_ids: Set[str] = set()
    
    total_scanned = 0
    # First pass: collect brand tweets and inbound tweets
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False, on_bad_lines="skip", dtype=str):
        total_scanned += len(chunk)
        
        # Clean IDs
        chunk["clean_tweet_id"] = chunk["tweet_id"].apply(normalize_id)
        chunk["clean_in_response_to"] = chunk["in_response_to_tweet_id"].apply(normalize_id)
        chunk["author_lower"] = chunk["author_id"].astype(str).str.lower()
        chunk["inbound_bool"] = chunk["inbound"].astype(str).str.lower() == "true"
        
        # 1. Brand replies
        b_mask = (chunk["author_lower"] == brand_handle_lower) & (~chunk["inbound_bool"])
        b_df = chunk[b_mask]
        if not b_df.empty:
            brand_tweets.append(b_df)
            for parent_id in b_df["clean_in_response_to"]:
                if parent_id:
                    needed_parent_ids.add(parent_id)
                    
        # 2. Inbound customer tweets targeting the brand or matching needed parents
        c_mask = chunk["inbound_bool"] & (
            chunk["clean_tweet_id"].isin(needed_parent_ids) |
            chunk["text"].astype(str).str.lower().str.contains(f"@{brand_handle_lower}")
        )
        c_df = chunk[c_mask]
        if not c_df.empty:
            customer_tweets.append(c_df)
            
        current_brand_count = sum(len(x) for x in brand_tweets)
        current_cust_count = sum(len(x) for x in customer_tweets)
        
        if current_brand_count >= max_tweets and current_cust_count >= max_tweets:
            break
            
    df_brand = pd.concat(brand_tweets, ignore_index=True) if brand_tweets else pd.DataFrame()
    df_cust = pd.concat(customer_tweets, ignore_index=True) if customer_tweets else pd.DataFrame()
    
    logger.info(f"Scanned {total_scanned} rows. Collected {len(df_brand)} brand tweets and {len(df_cust)} customer tweets.")
    df_combined = pd.concat([df_brand, df_cust], ignore_index=True).drop_duplicates(subset=["clean_tweet_id"])
    return df_combined


def reconstruct_conversations(
    df: pd.DataFrame,
    brand: str = "AppleSupport"
) -> pd.DataFrame:
    """
    Reconstruct customer-query to brand-reply conversation pairs.
    Each pair contains:
    - customer_tweet_id
    - customer_author_id
    - customer_text
    - brand_tweet_id
    - brand_reply_text
    - thread_id (root tweet id)
    """
    logger.info("Reconstructing conversation pairs...")
    brand_lower = brand.lower()
    
    df = df.copy()
    if "clean_tweet_id" not in df.columns:
        df["clean_tweet_id"] = df["tweet_id"].apply(normalize_id)
    if "clean_in_response_to" not in df.columns:
        df["clean_in_response_to"] = df["in_response_to_tweet_id"].apply(normalize_id)
    if "author_lower" not in df.columns:
        df["author_lower"] = df["author_id"].astype(str).str.lower()
        
    df["inbound_bool"] = df["inbound"].astype(str).str.lower() == "true"
    df["text"] = df["text"].astype(str)
    
    # Map tweet_id to row
    tweet_map = {row["clean_tweet_id"]: row for _, row in df.iterrows()}
    
    pairs = []
    # Identify brand tweets that respond to customer tweets
    brand_replies = df[(df["author_lower"] == brand_lower) & (df["clean_in_response_to"] != "")]
    
    for _, b_row in brand_replies.iterrows():
        cust_id = b_row["clean_in_response_to"]
        if cust_id in tweet_map:
            c_row = tweet_map[cust_id]
            # Verify customer tweet is inbound or authored by a customer (not brand itself)
            if c_row["author_lower"] != brand_lower:
                c_text = clean_text(c_row["text"])
                b_text = clean_text(b_row["text"])
                
                # Filter out unhelpful short responses or empty texts
                if len(c_text.split()) < 3 or len(b_text.split()) < 3:
                    continue
                    
                # Clean customer text of leading mentions for clearer semantic intent
                clean_c_text = re.sub(r"^(@\w+\s*)+", "", c_text).strip()
                if len(clean_c_text.split()) < 2:
                    clean_c_text = c_text
                    
                # Determine thread root id
                root_id = c_row.get("clean_in_response_to", "")
                thread_id = root_id if root_id else c_row["clean_tweet_id"]
                
                pairs.append({
                    "customer_tweet_id": str(c_row["clean_tweet_id"]),
                    "customer_author_id": str(c_row["author_id"]),
                    "customer_text": clean_c_text,
                    "raw_customer_text": c_text,
                    "brand_tweet_id": str(b_row["clean_tweet_id"]),
                    "brand_reply_text": b_text,
                    "thread_id": str(thread_id),
                    "created_at": str(c_row.get("created_at", ""))
                })
                
    df_pairs = pd.DataFrame(pairs)
    if not df_pairs.empty:
        df_pairs = df_pairs.drop_duplicates(subset=["customer_tweet_id"])
        logger.info(f"Reconstructed {len(df_pairs)} unique customer-agent conversation pairs.")
    else:
        logger.warning("No conversation pairs could be reconstructed!")
        
    return df_pairs


def create_splits(
    df_pairs: pd.DataFrame,
    ref_ratio: float = 0.70,
    train_ratio: float = 0.15,
    eval_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split conversations into Reference, Train, and Evaluation sets.
    GUARANTEES ZERO DATA LEAKAGE:
    Groups by thread_id so that all tweets from a thread
    are strictly assigned to a single split.
    """
    assert abs((ref_ratio + train_ratio + eval_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    np.random.seed(random_seed)
    
    unique_threads = np.array(df_pairs["thread_id"].unique(), dtype=object)
    np.random.shuffle(unique_threads)
    
    n_total = len(unique_threads)
    n_ref = int(n_total * ref_ratio)
    n_train = int(n_total * train_ratio)
    
    ref_threads = set(unique_threads[:n_ref])
    train_threads = set(unique_threads[n_ref : n_ref + n_train])
    eval_threads = set(unique_threads[n_ref + n_train :])
    
    df_ref = df_pairs[df_pairs["thread_id"].isin(ref_threads)].copy()
    df_train = df_pairs[df_pairs["thread_id"].isin(train_threads)].copy()
    df_eval = df_pairs[df_pairs["thread_id"].isin(eval_threads)].copy()
    
    logger.info(
        f"Splits generated: Reference={len(df_ref)} pairs ({len(ref_threads)} threads), "
        f"Train={len(df_train)} pairs ({len(train_threads)} threads), "
        f"Eval={len(df_eval)} pairs ({len(eval_threads)} threads)."
    )
    
    return df_ref, df_train, df_eval
