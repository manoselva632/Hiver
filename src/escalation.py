"""
Conservative, explainable escalation engine.
Enforces strict policy rules for security, billing, hardware damage, abuse, and retrieval uncertainty.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# Core escalation reason types
ESCALATION_REASONS = {
    "ACCOUNT_SECURITY": "Sensitive account security inquiry requiring human identity verification (Apple ID, passcode, lockouts).",
    "FINANCIAL_REFUND": "Financial dispute, billing inquiry, subscription cancellation, or refund demand.",
    "PHYSICAL_HARDWARE": "Physical hardware damage, battery swelling, or repair requiring in-person Genius Bar assessment.",
    "ABUSIVE_HOSTILE": "Hostile, abusive language or legal threats requiring human agent de-escalation.",
    "LOW_CONFIDENCE": "Insufficient historical evidence or low retrieval similarity to reliably auto-handle.",
    "UNSUPPORTED_DOMAIN": "Query falls outside supported Apple technical troubleshooting domain or is incoherent."
}


class EscalationPolicy:
    """
    Evaluates customer tweet, predicted intent, and retrieved evidence to decide:
    AUTO_HANDLE vs ESCALATE.
    Conservative by design: when in doubt, safely escalate to human specialists.
    """
    def __init__(self, similarity_threshold: float = 0.55, intent_confidence_threshold: float = 0.50):
        self.similarity_threshold = similarity_threshold
        self.intent_confidence_threshold = intent_confidence_threshold

    def evaluate(
        self,
        customer_text: str,
        predicted_intent: str,
        intent_confidence: float,
        retrieved_evidence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate inquiry against deterministic safety policies.
        Returns:
            decision: "AUTO_HANDLE" | "ESCALATE"
            escalation_reason: Specific explanation and rule triggered
            escalation_type: Category identifier or "NONE"
            trigger_evidence: Cited keywords, thresholds, or retrieval scores
        """
        text_lower = customer_text.lower()
        
        # 1. Abusive, hostile, or legal threats
        abuse_patterns = [
            r"\b(fuck|fucking|bitch|bastard|asshole|scam(mers)?|thieves|lawsuit|lawyer|sue|suing|police)\b"
        ]
        for pattern in abuse_patterns:
            match = re.search(pattern, text_lower)
            if match:
                return {
                    "decision": "ESCALATE",
                    "escalation_type": "ABUSIVE_HOSTILE",
                    "escalation_reason": f"High hostility or legal risk detected ('{match.group(0)}'). Requires human specialist de-escalation.",
                    "confidence_score": 0.99
                }
                
        # 2. Sensitive Account Security
        security_patterns = [
            r"\b(apple\s*id|password|passcode|locked\s*out|hacked|two[- ]factor|2fa|verification\s*code|recover\s*account)\b"
        ]
        if predicted_intent == "account_apple_id" or any(re.search(p, text_lower) for p in security_patterns):
            return {
                "decision": "ESCALATE",
                "escalation_type": "ACCOUNT_SECURITY",
                "escalation_reason": "Account credentials and security locks require verified human authentication to prevent unauthorized takeover.",
                "confidence_score": 0.95
            }
            
        # 3. Financial, Billing, and Refund Inquiries
        billing_patterns = [
            r"\b(refund|overcharged|unauthorized\s*charge|dispute|money\s*back|credit\s*card|billed\s*twice)\b"
        ]
        if predicted_intent == "app_store_billing" or any(re.search(p, text_lower) for p in billing_patterns):
            return {
                "decision": "ESCALATE",
                "escalation_type": "FINANCIAL_REFUND",
                "escalation_reason": "Financial transaction or refund request requires direct human review and transaction lookup.",
                "confidence_score": 0.95
            }
            
        # 4. Physical Hardware Damage & Repairs
        hardware_patterns = [
            r"\b(cracked|shattered|broken\s*screen|water\s*damage|dropped\s*in|genius\s*bar|swollen\s*battery|repair\s*cost)\b"
        ]
        if predicted_intent == "hardware_physical" or any(re.search(p, text_lower) for p in hardware_patterns):
            return {
                "decision": "ESCALATE",
                "escalation_type": "PHYSICAL_HARDWARE",
                "escalation_reason": "Physical damage or hardware repair requires physical Genius Bar diagnostics or authorized service dispatch.",
                "confidence_score": 0.92
            }
            
        # 5. Out of Domain or Incoherent Queries
        if len(customer_text.strip().split()) < 3 or re.match(r"^[^a-zA-Z0-9]+$", customer_text.strip()):
            return {
                "decision": "ESCALATE",
                "escalation_type": "UNSUPPORTED_DOMAIN",
                "escalation_reason": "Inquiry is too brief or syntactically malformed to reliably interpret automated intent.",
                "confidence_score": 0.85
            }
            
        # 6. Retrieval Grounding & Evidence Sufficiency Check
        if not retrieved_evidence:
            return {
                "decision": "ESCALATE",
                "escalation_type": "LOW_CONFIDENCE",
                "escalation_reason": "Zero historical precedents retrieved from reference knowledge base. Escalating to avoid hallucinated response.",
                "confidence_score": 0.90
            }
            
        top_similarity = max((e.get("similarity", 0.0) for e in retrieved_evidence), default=0.0)
        if top_similarity < self.similarity_threshold:
            return {
                "decision": "ESCALATE",
                "escalation_type": "LOW_CONFIDENCE",
                "escalation_reason": (
                    f"Top retrieved historical precedent has similarity {top_similarity:.3f}, "
                    f"below minimum safety threshold {self.similarity_threshold:.2f}. "
                    "Cannot guarantee grounded resolution."
                ),
                "confidence_score": round(1.0 - top_similarity, 3)
            }
            
        if intent_confidence < self.intent_confidence_threshold:
            return {
                "decision": "ESCALATE",
                "escalation_type": "LOW_CONFIDENCE",
                "escalation_reason": (
                    f"Predicted intent confidence ({intent_confidence:.2f}) is below minimum reliable threshold "
                    f"({self.intent_confidence_threshold:.2f})."
                ),
                "confidence_score": round(1.0 - intent_confidence, 3)
            }
            
        # 7. Safe for automated handling
        return {
            "decision": "AUTO_HANDLE",
            "escalation_type": "NONE",
            "escalation_reason": (
                f"Sufficient historical evidence found (top similarity {top_similarity:.3f}). "
                f"Standard troubleshooting intent ({predicted_intent}) confirmed without security or financial risks."
            ),
            "confidence_score": round(top_similarity, 3)
        }
