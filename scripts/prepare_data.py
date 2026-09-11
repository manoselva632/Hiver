"""
Script: prepare_data.py
Extracts reproducible representative subsample of AppleSupport conversations,
reconstructs thread relationships, labels intents via domain taxonomy,
and generates zero-leakage splits (reference, train, eval).
"""

import os
import sys
import argparse
import logging
import pandas as pd

# Add repo root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import stream_subsample_brand, reconstruct_conversations, create_splits
from src.intent import assign_intent_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Prepare and split AppleSupport customer conversations.")
    parser.add_argument("--input_csv", type=str, default="../twcs/twcs.csv", help="Path to raw twcs.csv")
    parser.add_argument("--brand", type=str, default="AppleSupport", help="Target brand handle")
    parser.add_argument("--max_tweets", type=int, default=20000, help="Max tweets to scan for subsample")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for splits")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Check input path (support running from repo/ or repo/scripts/)
    input_path = args.input_csv
    if not os.path.exists(input_path):
        alt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../twcs/twcs.csv"))
        if os.path.exists(alt_path):
            input_path = alt_path
        elif os.path.exists("twcs/twcs.csv"):
            input_path = "twcs/twcs.csv"
        else:
            raise FileNotFoundError(f"Cannot locate twcs.csv at {input_path} or {alt_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Using input dataset at: {input_path}")
    logger.info(f"Target brand: {args.brand}, Random Seed: {args.seed}")

    # 1. Stream subsample for brand
    df_raw = stream_subsample_brand(input_path, brand=args.brand, max_tweets=args.max_tweets)
    
    # 2. Reconstruct conversation pairs
    df_pairs = reconstruct_conversations(df_raw, brand=args.brand)
    if df_pairs.empty:
        logger.error("Failed to reconstruct any conversation pairs.")
        sys.exit(1)

    # 3. Assign domain intent labels
    logger.info("Discovering and assigning intent labels to conversation pairs...")
    df_labeled = assign_intent_labels(df_pairs, text_col="customer_text")

    # Log intent distribution
    logger.info("Intent distribution across reconstructed pairs:")
    print(df_labeled["intent"].value_counts().to_string())

    # 4. Generate leak-free splits
    df_ref, df_train, df_eval = create_splits(
        df_labeled, ref_ratio=0.70, train_ratio=0.15, eval_ratio=0.15, random_seed=args.seed
    )

    # Save to disk
    ref_file = os.path.join(args.output_dir, "reference_set.csv")
    train_file = os.path.join(args.output_dir, "train_set.csv")
    eval_file = os.path.join(args.output_dir, "eval_set.csv")

    df_ref.to_csv(ref_file, index=False)
    df_train.to_csv(train_file, index=False)
    df_eval.to_csv(eval_file, index=False)

    logger.info(f"Successfully saved splits:")
    logger.info(f"  Reference Set : {ref_file} ({len(df_ref)} rows)")
    logger.info(f"  Train Set     : {train_file} ({len(df_train)} rows)")
    logger.info(f"  Eval Set      : {eval_file} ({len(df_eval)} rows)")


if __name__ == "__main__":
    main()
