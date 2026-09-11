# Data Directory & Golden Benchmark Specification

This directory houses the processed subsets, splits, and evaluation benchmarks derived from the Kaggle **Customer Support on Twitter** (`thoughtvector/customer-support-on-twitter`) dataset.

---

## Dataset Artifacts

1. `reference_set.csv` (17,293 rows, 70%):
   - Historical resolved customer-agent conversation pairs for AppleSupport.
   - Serves as the indexed knowledge base for dense vector retrieval.
2. `train_set.csv` (3,731 rows, 15%):
   - Historical conversation pairs used to train baseline TF-IDF vectorizers and logistic classifiers.
3. `eval_set.csv` (3,701 rows, 15%):
   - Held-out conversation pairs reserved strictly for testing and candidate sampling.
4. `golden_set.csv` (200 rows):
   - Stratified evaluation benchmark with representative distribution across all 9 discovered intents, varied conversation depths, and safety escalation boundaries.

---

## Zero Data Leakage Guarantee

To prevent cross-set leakage:
- Rows are grouped by conversation `thread_id`.
- All interactions in a thread (initial inquiry, brand response, follow-up turns) reside exclusively in **one** split.
- The evaluation and golden sets contain zero customer tweet IDs or thread IDs present in the train or reference sets.

---

## Intent Taxonomy (Apple Support Domain)

| Intent Key | Description | Example Query |
| :--- | :--- | :--- |
| `battery_power` | Rapid battery drain, charging failure, overheating | *"Phone battery dropped 15% in 5 minutes after unplugging."* |
| `ios_update_bugs` | iOS release glitches, UI stutter, post-update lag | *"Updated to iOS 11 and now keyboard freezes in iMessage."* |
| `audio_media` | Apple Music streaming, AirPods, speaker distortion | *"Music keeps pausing every 30 seconds on Apple Music."* |
| `connectivity` | Wi-Fi disconnects, Bluetooth pairing, cellular loss | *"Wi-Fi disconnects every time the phone goes to sleep."* |
| `app_store_billing` | In-app purchases, billing charges, refund requests | *"Charged twice for my monthly iCloud subscription, want refund."* |
| `account_apple_id` | Locked Apple ID, forgotten passcode, 2FA codes | *"Apple ID is locked for security reasons, cannot log in."* |
| `hardware_physical` | Broken screen, water immersion, swollen battery | *"Dropped phone on concrete and the front glass shattered."* |
| `device_storage_backup` | iCloud backup failure, storage full notifications | *"Cannot back up to iCloud, says storage full but I bought 50GB."* |
| `general_inquiry` | Setup questions, compatibility, store policies | *"Is Apple Watch Series 3 compatible with iPhone 6?"* |

---

## Golden Set Schema

| Field | Type | Description |
| :--- | :--- | :--- |
| `customer_tweet_id` | String | Unique identifier of customer tweet |
| `customer_author_id` | String | Anonymized Twitter author ID |
| `customer_text` | String | Cleaned customer tweet text |
| `intent_annotated` | String | Target intent category (one of 9 taxonomy classes) |
| `expected_decision` | String | `AUTO_HANDLE` or `ESCALATE` |
| `escalation_reason_type` | String | Triggered policy rule (e.g., `ACCOUNT_SECURITY`, `FINANCIAL_REFUND`, `PHYSICAL_HARDWARE`, `ABUSIVE_HOSTILE`, `NONE`) |
| `verified_by_human` | Boolean | `True` if validated by a human annotator; `False` if generated via rubric bootstrap |
| `notes` | String | Contextual rationale for boundary decisions |

---

## Human Verification Protocol & Workflow

> [!IMPORTANT]
> **Transparency Disclosure**:
> In compliance with strict academic and engineering integrity standards, all rows in `golden_set.csv` are initially generated using rule-stratified heuristics and are explicitly marked with **`verified_by_human: False`**. No claims of manual labeling are made unless human verification is performed.

### Verification Instructions for Annotators:
1. Open `golden_set.csv` in a spreadsheet editor (Excel, Google Sheets, or VS Code CSV Editor).
2. For each row:
   - **Review `customer_text`**: Read the customer's stated issue.
   - **Validate `intent_annotated`**: Verify if the assigned intent matches the primary customer complaint. If multiple issues are present, prioritize the blocking problem (e.g., account lockout over UI bug).
   - **Validate `expected_decision`**:
     - Assign `ESCALATE` if the query involves:
       - Account passwords, Apple ID lockouts, or 2FA credentials (`ACCOUNT_SECURITY`).
       - Monetary refund demands or credit card disputes (`FINANCIAL_REFUND`).
       - Physical device damage, cracked screen, or hardware repair (`PHYSICAL_HARDWARE`).
       - Explicit profanity, legal threats, or hostile rants (`ABUSIVE_HOSTILE`).
       - Truncated or unintelligible complaints (`UNSUPPORTED_DOMAIN`).
     - Assign `AUTO_HANDLE` if the issue is a standard software/troubleshooting concern (rebooting, adjusting settings, updating iOS).
   - **Set `verified_by_human` to `True`**: Update the boolean flag once reviewed.
   - **Update `notes`**: Add any clarification for borderline decisions.
3. Save the validated file back to `data/golden_set.csv`.
