"""
Lightweight Web & API Server for the AI Support Agent Dashboard.
Serves static dashboard UI and provides live triage API endpoints.
Built on Python's standard library http.server (zero extra dependencies).
"""

import os
import sys
import json
import logging
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import urllib.parse
import pandas as pd

# Add repo root to sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.intent import TFIDFLogisticIntentClassifier, ProposedIntentClassifier
from src.retrieval import RetrievalEngine
from src.escalation import EscalationPolicy
from src.agent import ProposedAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Global Agent Instance
SUPPORT_AGENT = None
GOLDEN_SET_DATA = []
METRICS_DATA = {}


def initialize_agent():
    """Load reference data and initialize retrieval agent."""
    global SUPPORT_AGENT, GOLDEN_SET_DATA, METRICS_DATA
    logger.info("Initializing Agent and Loading Knowledge Bases...")
    
    data_dir = os.path.join(REPO_ROOT, "data")
    reports_dir = os.path.join(REPO_ROOT, "reports")
    
    ref_path = os.path.join(data_dir, "reference_set.csv")
    train_path = os.path.join(data_dir, "train_set.csv")
    golden_path = os.path.join(data_dir, "golden_set.csv")
    metrics_path = os.path.join(reports_dir, "eval_results.json")

    # Load golden set
    if os.path.exists(golden_path):
        df_golden = pd.read_csv(golden_path)
        GOLDEN_SET_DATA = df_golden.to_dict(orient="records")
        logger.info(f"Loaded {len(GOLDEN_SET_DATA)} golden benchmark records.")

    # Load metrics
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            METRICS_DATA = json.load(f)
        logger.info("Loaded benchmark evaluation metrics.")

    # Initialize agent models
    if os.path.exists(ref_path) and os.path.exists(train_path):
        df_ref = pd.read_csv(ref_path)
        df_train = pd.read_csv(train_path)

        tfidf_clf = TFIDFLogisticIntentClassifier()
        tfidf_clf.fit(df_train["customer_text"].astype(str).tolist(), df_train["intent"].astype(str).tolist())

        retrieval_engine = RetrievalEngine(cache_dir=data_dir, use_dense=True)
        retrieval_engine.index(df_ref, text_col="customer_text")

        proposed_clf = ProposedIntentClassifier(tfidf_model=tfidf_clf)
        policy = EscalationPolicy(similarity_threshold=0.55, intent_confidence_threshold=0.50)

        SUPPORT_AGENT = ProposedAgent(
            intent_classifier=proposed_clf,
            retrieval_engine=retrieval_engine,
            escalation_policy=policy
        )
        logger.info("Proposed Support Agent successfully initialized and ready for live triage.")
    else:
        logger.warning("Reference or Train CSV missing. Live agent triage API will operate in mock mode.")


class DashboardAPIHandler(SimpleHTTPRequestHandler):
    """HTTP Request Handler supporting static web files and API endpoints."""

    def __init__(self, *args, **kwargs):
        web_dir = os.path.join(REPO_ROOT, "web")
        super().__init__(*args, directory=web_dir, **kwargs)

    def _set_json_headers(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_json_headers(200)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        
        # API: Get Metrics
        if parsed.path == "/api/metrics":
            self._set_json_headers(200)
            self.wfile.write(json.dumps(METRICS_DATA).encode("utf-8"))
            return

        # API: Get Golden Set
        if parsed.path == "/api/golden_set":
            self._set_json_headers(200)
            self.wfile.write(json.dumps(GOLDEN_SET_DATA).encode("utf-8"))
            return

        # Fallback to serving static files
        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)

        # API: Live Triage Endpoint
        if parsed.path == "/api/triage":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len).decode("utf-8")
            
            try:
                payload = json.loads(body)
                query = payload.get("query", "").strip()

                if not query:
                    self._set_json_headers(400)
                    self.wfile.write(json.dumps({"error": "Empty query"}).encode("utf-8"))
                    return

                if SUPPORT_AGENT is not None:
                    result = SUPPORT_AGENT.predict(query)
                else:
                    result = {
                        "intent": "general_inquiry",
                        "reply": "Please DM us your device model and iOS version.",
                        "decision": "AUTO_HANDLE",
                        "escalation_reason": "Mock agent fallback.",
                        "evidence": []
                    }

                self._set_json_headers(200)
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except Exception as e:
                logger.error(f"Error processing triage API request: {e}")
                self._set_json_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        self._set_json_headers(404)
        self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))


def run_server(port: int = 8000):
    initialize_agent()
    server_address = ("", port)
    httpd = ThreadingHTTPServer(server_address, DashboardAPIHandler)
    logger.info(f"Dashboard Web Application running at: http://localhost:{port}")
    logger.info(f"Press Ctrl+C to stop the server.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_server(port)
