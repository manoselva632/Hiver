"""
Dense and sparse retrieval engine with embedding caching and Hit@k evaluation.
"""

import os
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """
    Retrieval engine over historical reference conversations.
    Supports SentenceTransformer dense embeddings with disk caching,
    plus an automated fallback to TF-IDF cosine similarity.
    """
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: str = "data",
        use_dense: bool = True
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.use_dense = use_dense
        self.encoder = None
        self.tfidf_vectorizer = None
        
        self.ref_df = None
        self.embeddings = None
        
    def _init_encoder(self):
        """Lazy load SentenceTransformer encoder."""
        if self.encoder is None and self.use_dense:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading SentenceTransformer model '{self.model_name}'...")
                self.encoder = SentenceTransformer(self.model_name)
                logger.info("SentenceTransformer loaded successfully.")
            except Exception as e:
                logger.warning(f"Failed to load SentenceTransformer ({e}). Falling back to TF-IDF sparse retrieval.")
                self.use_dense = False

    def _get_cache_path(self, texts: List[str]) -> str:
        """Create a deterministic cache filename based on text hash and model name."""
        content_hash = hashlib.sha256(("".join(texts[:50]) + str(len(texts)) + self.model_name).encode()).hexdigest()[:12]
        return os.path.join(self.cache_dir, f"embeddings_{self.model_name}_{content_hash}.npz")

    def index(self, ref_df: pd.DataFrame, text_col: str = "customer_text"):
        """Index historical customer queries and their brand responses."""
        self.ref_df = ref_df.reset_index(drop=True).copy()
        texts = self.ref_df[text_col].astype(str).tolist()
        os.makedirs(self.cache_dir, exist_ok=True)
        
        if self.use_dense:
            cache_path = self._get_cache_path(texts)
            if os.path.exists(cache_path):
                try:
                    logger.info(f"Loading cached embeddings from {cache_path}...")
                    data = np.load(cache_path)
                    self.embeddings = data["embeddings"]
                    logger.info(f"Loaded {len(self.embeddings)} embeddings from cache.")
                    return self
                except Exception as e:
                    logger.warning(f"Failed to load cache ({e}). Recomputing embeddings...")
                    
            self._init_encoder()
            if self.encoder is not None:
                logger.info(f"Computing dense embeddings for {len(texts)} reference queries...")
                self.embeddings = self.encoder.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
                np.savez_compressed(cache_path, embeddings=self.embeddings)
                logger.info(f"Saved embeddings to cache: {cache_path}")
                return self
                
        # Sparse TF-IDF fallback
        logger.info(f"Building sparse TF-IDF index for {len(texts)} reference queries...")
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
        self.embeddings = self.tfidf_vectorizer.fit_transform(texts)
        return self

    def retrieve(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most similar historical conversations for a given customer query.
        Returns list of citations with tweet_id, similarity, customer_text, and brand_reply.
        """
        if self.ref_df is None or self.embeddings is None:
            raise ValueError("RetrievalEngine has not been indexed yet. Call index() first.")
            
        if self.use_dense:
            self._init_encoder()
            if self.encoder is not None:
                query_vec = self.encoder.encode([query_text], normalize_embeddings=True)[0]
                # Cosine similarity for normalized vectors is just dot product
                scores = np.dot(self.embeddings, query_vec)
            else:
                if self.tfidf_vectorizer is None:
                    raise ValueError("No retrieval index available.")
                query_vec = self.tfidf_vectorizer.transform([query_text])
                scores = cosine_similarity(query_vec, self.embeddings)[0]
        else:
            if self.tfidf_vectorizer is None:
                raise ValueError("No retrieval index available.")
            query_vec = self.tfidf_vectorizer.transform([query_text])
            scores = cosine_similarity(query_vec, self.embeddings)[0]
            
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            row = self.ref_df.iloc[idx]
            results.append({
                "tweet_id": str(row.get("brand_tweet_id", row.get("customer_tweet_id", ""))),
                "customer_tweet_id": str(row.get("customer_tweet_id", "")),
                "customer_text": str(row.get("customer_text", "")),
                "historical_reply": str(row.get("brand_reply_text", "")),
                "similarity": round(float(scores[idx]), 4),
                "intent": str(row.get("intent", ""))
            })
            
        return results

    def evaluate_hit_at_k(self, eval_df: pd.DataFrame, k_list: List[int] = [1, 3, 5]) -> Dict[str, float]:
        """
        Evaluate Hit@k and Mean Reciprocal Rank (MRR) where a hit means
        retrieved example shares the query's ground-truth intent.
        """
        if "intent" not in eval_df.columns:
            logger.warning("eval_df missing 'intent' column. Skipping Hit@k evaluation.")
            return {}
            
        hits = {f"hit@{k}": 0 for k in k_list}
        reciprocal_ranks = []
        max_k = max(k_list)
        
        valid_queries = 0
        for _, row in eval_df.iterrows():
            q_text = str(row["customer_text"])
            q_intent = str(row["intent"])
            if not q_intent or q_intent == "nan":
                continue
                
            retrieved = self.retrieve(q_text, top_k=max_k)
            valid_queries += 1
            
            # Find first rank of matching intent
            hit_ranks = [i + 1 for i, r in enumerate(retrieved) if r.get("intent") == q_intent]
            if hit_ranks:
                first_rank = hit_ranks[0]
                reciprocal_ranks.append(1.0 / first_rank)
                for k in k_list:
                    if first_rank <= k:
                        hits[f"hit@{k}"] += 1
            else:
                reciprocal_ranks.append(0.0)
                
        metrics = {}
        if valid_queries > 0:
            for k in k_list:
                metrics[f"hit@{k}"] = round(hits[f"hit@{k}"] / valid_queries, 4)
            metrics["mrr"] = round(float(np.mean(reciprocal_ranks)), 4)
            metrics["total_evaluated"] = valid_queries
            
        return metrics
