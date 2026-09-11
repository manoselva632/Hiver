"""
Script: run_demo.py
Interactive CLI demonstration of the Proposed AI Customer Support Agent.
Tests input customer tweets against retrieval, intent classification,
grounded reply generation, and conservative escalation.
"""

import os
import sys
import json
import argparse
import logging
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Add repo root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.intent import TFIDFLogisticIntentClassifier, ProposedIntentClassifier
from src.retrieval import RetrievalEngine
from src.escalation import EscalationPolicy
from src.agent import ProposedAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_DEMO_QUERIES = [
    "My iPhone 7 battery is draining from 100% to 15% in less than an hour after the new update!",
    "Someone hacked into my Apple ID account and changed my password and 2FA phone number, help!",
    "I was charged $99 for an in-app subscription that I cancelled last week! I demand a full refund immediately.",
    "Dropped my iPhone on the sidewalk and the entire glass screen shattered into pieces. Can I get it fixed at Genius Bar?",
    "Every time I connect my AirPods to my Mac, the audio is distorted and cuts out completely after 30 seconds.",
    "I am so fucking sick of this piece of shit phone constantly crashing! Your engineers are total scammers and I'm going to sue Apple!"
]


def print_formatted_result(query: str, res: dict):
    print("\n" + "=" * 75)
    print("CUSTOMER INQUIRY:")
    print(f"  \"{query}\"")
    print("-" * 75)
    print(f"IDENTIFIED INTENT : {res['intent'].upper()}")
    print(f"TRIAGE DECISION   : {res['decision']}  (Reason: {res['escalation_reason']})")
    print("-" * 75)
    print("RETRIEVED HISTORICAL CITATIONS:")
    if res.get("evidence"):
        for i, ev in enumerate(res["evidence"], 1):
            print(f"  [{i}] Tweet #{ev.get('tweet_id')} (Sim: {ev.get('similarity', 0):.3f})")
            print(f"      Customer: \"{ev.get('customer_text', '')[:90]}...\"")
            print(f"      Apple   : \"{ev.get('historical_reply', '')[:90]}...\"")
    else:
        print("  None.")
    print("-" * 75)
    print("GENERATED AGENT REPLY:")
    print(f"  \"{res['reply']}\"")
    print("-" * 75)
    print("STRUCTURED JSON OUTPUT:")
    print(json.dumps(res, indent=2))
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run interactive or batch demo of the AI Support Agent.")
    parser.add_argument("--query", type=str, default="", help="Single customer query to triage")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory path")
    args = parser.parse_args()

    ref_path = os.path.join(args.data_dir, "reference_set.csv")
    train_path = os.path.join(args.data_dir, "train_set.csv")

    if not os.path.exists(ref_path) or not os.path.exists(train_path):
        raise FileNotFoundError("Reference or train set missing. Run prepare_data.py first.")

    logger.info("Initializing Proposed Support Agent...")
    df_ref = pd.read_csv(ref_path)
    df_train = pd.read_csv(train_path)

    # Train classifier & index retriever
    tfidf_clf = TFIDFLogisticIntentClassifier()
    tfidf_clf.fit(df_train["customer_text"].astype(str).tolist(), df_train["intent"].astype(str).tolist())

    retrieval_engine = RetrievalEngine(cache_dir=args.data_dir, use_dense=True)
    retrieval_engine.index(df_ref, text_col="customer_text")

    proposed_clf = ProposedIntentClassifier(tfidf_model=tfidf_clf)
    policy = EscalationPolicy(similarity_threshold=0.55, intent_confidence_threshold=0.50)

    agent = ProposedAgent(
        intent_classifier=proposed_clf,
        retrieval_engine=retrieval_engine,
        escalation_policy=policy
    )

    if args.query:
        queries = [args.query]
    else:
        logger.info("No custom query provided. Demonstrating across diverse test cases...")
        queries = SAMPLE_DEMO_QUERIES

    for q in queries:
        res = agent.predict(q)
        print_formatted_result(q, res)


if __name__ == "__main__":
    main()
