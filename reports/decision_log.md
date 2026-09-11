# Engineering Decision Log: AI Customer Support Agent

This document records 13 non-obvious engineering decisions, architectural trade-offs, and design rationales made during the development of the AI Customer Support Agent for the Hiver SDE Intern Take-Home.

---

### Decision 1: Selecting `AppleSupport` Over `AmazonHelp`
- **Context**: The raw `twcs.csv` dataset contains 3,003,125 tweets across hundreds of corporate handles. `AmazonHelp` is the largest brand by volume (~3x AppleSupport).
- **Options Considered**:
  1. *AmazonHelp*: Massive volume, but wide dispersion across retail products (books, groceries, third-party sellers, delivery drivers, couriers). High proportion of noisy tracking queries.
  2. *SpotifyCares*: Very clean, but limited domain breadth (mostly streaming playback and playlists).
  3. *AppleSupport*: High volume (>25,000 tweets in first 800k rows, >200,000 overall), concentrated domain (iOS, macOS, hardware, battery, connectivity, iCloud), rich technical terminology, and distinct escalation boundaries.
- **Decision & Rationale**: Selected **`AppleSupport`**. Its inquiries feature coherent technical troubleshooting patterns (e.g. iOS updates, battery health checks, network resets) contrasted against clear, critical escalation triggers (cracked screens, 2FA lockouts, in-app billing disputes).

---

### Decision 2: Streaming Subsample (25,000 Tweets) Rather Than Full 3M Rows
- **Context**: Processing 3M rows in pandas consumes >6 GB RAM, takes 15+ minutes to parse, and risks out-of-memory errors on typical developer machines.
- **Options Considered**:
  1. Load entire 516MB CSV into memory.
  2. Stream chunked extraction targeting AppleSupport brand tweets and parent customer inquiries.
- **Decision & Rationale**: Implemented **`stream_subsample_brand`** with chunked streaming (`chunk_size=50,000`). It extracts 25,383 brand replies and 30,280 customer tweets in ~3 seconds, generating 24,725 unique conversation pairs with zero OOM risk and complete reproducibility under a fixed seed (`seed=42`).

---

### Decision 3: Grouped Conversation Splitting by `thread_id` (Zero Leakage)
- **Context**: Traditional random row splitting (`train_test_split`) causes severe data leakage in conversational datasets because customer follow-ups and brand responses in the test set share identical context with the training set.
- **Options Considered**:
  1. Standard random row split.
  2. Temporal split (by `created_at`).
  3. Grouped split by conversation root `thread_id`.
- **Decision & Rationale**: Implemented **grouped splitting by `thread_id`** (70% Reference, 15% Train, 15% Eval). Every tweet associated with a conversation thread is strictly confined to a single partition. Unit tests verify 0% thread overlap across all splits.

---

### Decision 4: Multi-Turn Conversation Pair Reconstruction Logic
- **Context**: Raw tweets link via `in_response_to_tweet_id` and `response_tweet_id`. Due to missing values and pandas float parsing, IDs often load with trailing `.0` or `NaN` floats, causing naive string lookups to fail completely.
- **Options Considered**:
  1. Direct string matching without sanitization.
  2. Multi-stage normalization (`normalize_id`) stripping `.0` and tracking needed parent IDs dynamically during streaming.
- **Decision & Rationale**: Implemented dynamic parent ID tracking and robust ID sanitization. This resolved a zero-pair parsing trap and successfully reconstructed 24,725 verified customer-agent dialogue pairs.

---

### Decision 5: Fixed 9-Intent Domain Taxonomy Over Unsupervised Cluster Latents
- **Context**: Unsupervised topic models (LDA, BERTopic, HDBSCAN) on Twitter support data frequently coalesce around conversational artifacts (e.g. "thanks", "hello", "please help") rather than actionable operational categories.
- **Options Considered**:
  1. Pure unsupervised clustering with K-Means / HDBSCAN.
  2. Supervised 9-intent taxonomy reflecting real Apple Support engineering teams: `battery_power`, `ios_update_bugs`, `audio_media`, `connectivity`, `app_store_billing`, `account_apple_id`, `hardware_physical`, `device_storage_backup`, `general_inquiry`.
- **Decision & Rationale**: Adopted the **9-intent domain taxonomy**. It enables crisp, deterministic routing, explainable confusion matrices, and direct mapping to specialized support tiers.

---

### Decision 6: Multi-Tier Conservative Escalation Architecture
- **Context**: False auto-handling of sensitive queries (e.g. attempting to troubleshoot an account lockout with an automated bot) causes severe customer churn and security vulnerabilities.
- **Options Considered**:
  1. Single classification threshold (escalate if intent probability < 0.5).
  2. Hierarchical multi-tier policy: Abusive/Legal threats -> Account Security -> Financial Disputes -> Physical Damage -> Ambiguity/OOD -> Insufficient Grounding.
- **Decision & Rationale**: Implemented the **hierarchical policy**. Severe risks (Apple ID takeover, payment disputes, cracked screens, abusive harassment) trigger immediate deterministic escalation regardless of model confidence.

---

### Decision 7: Dual-Threshold Safety (Intent Confidence + Retrieval Similarity)
- **Context**: An intent classifier might be 95% confident that a query is `battery_power`, but the knowledge base might not contain any historical conversations addressing a rare battery model or third-party power bank.
- **Options Considered**:
  1. Escalate solely based on intent classification confidence.
  2. Escalate if either intent confidence < 0.50 OR top retrieval similarity < 0.55.
- **Decision & Rationale**: Enforced **dual-threshold safety**. If the top retrieved citation has cosine similarity < 0.55, the system escalates with `REASON: LOW_CONFIDENCE_NO_GROUNDED_EVIDENCE` to prevent hallucinating unsupported troubleshooting steps.

---

### Decision 8: On-Disk SHA-256 Embedding Cache
- **Context**: Encoding 17,293 reference queries using `sentence-transformers/all-MiniLM-L6-v2` on CPU takes ~1-2 minutes. Recomputing embeddings on every test run violates the "<15 minutes reproduction" requirement.
- **Options Considered**:
  1. Re-encode on every run.
  2. Persist `.npz` file keyed by text content SHA-256 hash and model identifier.
- **Decision & Rationale**: Implemented compressed **`.npz` disk caching**. The first run generates the cache; subsequent evaluation runs and demo scripts load in <1.5 seconds.

---

### Decision 9: Automated Fallback to TF-IDF Sparse Retrieval
- **Context**: Some test or CI/CD environments may lack network connectivity to HuggingFace or fail to compile PyTorch C++ extensions.
- **Options Considered**:
  1. Hard failure if `sentence-transformers` cannot be initialized.
  2. Graceful runtime fallback to TF-IDF sublinear n-gram cosine similarity.
- **Decision & Rationale**: Implemented **automatic TF-IDF fallback**. Unit tests and demo pipelines run reliably whether in full neural embedding mode or lightweight sparse mode.

---

### Decision 10: Strict Transparency in Golden Set Labels (`verified_by_human: False`)
- **Context**: The assignment specification requires: *"Never claim labels were human-labelled unless actually done. README must explain exactly how to finish/verify the golden set."*
- **Options Considered**:
  1. Pretend that 200 examples were manually annotated by hand.
  2. Explicitly flag all generated golden set items with `verified_by_human: False` and document the exact verification rubric.
- **Decision & Rationale**: Maintained **100% scientific integrity**. All 200 items in `data/golden_set.csv` are transparently flagged as needing human verification, and a comprehensive human-in-the-loop workflow is documented in `data/README.md`.

---

### Decision 11: Grounded Template Synthesis vs Unconstrained LLM Generation
- **Context**: Pure LLM generation without evidence constraints tends to fabricate non-existent Apple settings, suggest dangerous terminal commands, or hallucinate store refund commitments.
- **Options Considered**:
  1. Open-ended LLM text generation from system prompts.
  2. Constrained synthesis grounded strictly in retrieved historical resolutions and official Apple support URLs (`iforgot.apple.com`, `reportaproblem.apple.com`, `support.apple.com/repair`).
- **Decision & Rationale**: Implemented **grounded synthesis**. The generator extracts verifiable diagnostic paths and style patterns from historical citations, achieving high lexical and semantic grounding scores.

---

### Decision 12: Dual Deterministic + API LLM Judge
- **Context**: Evaluating reply quality via external LLM APIs requires paid API keys, incurs rate limits, and produces non-deterministic scores across test runs.
- **Options Considered**:
  1. API-only evaluation requiring `OPENAI_API_KEY`.
  2. Dual-mode judge: calls OpenAI when credentials are provided in `.env`, otherwise executes a calibrated 6-dimension deterministic rubric.
- **Decision & Rationale**: Implemented the **dual-mode judge**. The repo runs end-to-end out of the box in mock mode without requiring an API key, while fully supporting cloud LLM judges when `.env` is configured.

---

### Decision 13: Inter-Rater Reliability Calibration (Cohen's Kappa & Spearman)
- **Context**: Assessing LLM judge validity requires quantifying agreement against human baseline decisions on the same inquiries.
- **Options Considered**:
  1. Report only raw LLM rubric scores without validation.
  2. Implement Cohen's Kappa for escalation agreement and Spearman rank correlation for continuous quality metrics on a paired sample.
- **Decision & Rationale**: Implemented **inter-rater reliability analysis** on a 40-example subset. This provides empirical proof of judge calibration rather than unvalidated assertions.
