# src/anomaly.py
from src.pipeline import TeleRAGPipeline
 
ANOMALY_INDICATORS = [
    'high interference', 'handover failure', 'RLF', 'radio link failure',
    'call drop', 'throughput degradation', 'high BLER', 'ping pong',
    'CQI drop', 'alarm', 'fault', 'outage', 'degradation',
    'packet loss', 'latency spike', 'cell down', 'coverage hole',
]
 
class AnomalyDetector:
    def __init__(self, pipeline: TeleRAGPipeline):
        self.pipeline = pipeline
 
    def analyze(self, log_text: str) -> dict:
        # Step 1 — keyword pre-screen
        found = [kw for kw in ANOMALY_INDICATORS if kw.lower() in log_text.lower()]

        enriched = (
            f"Analyze the RAN log for anomalies"
            f"Identify the fault, affected components and recommend fix:\n"
            f"{log_text}"
        )
 
        # Step 2 — RAG-based analysis using RCA query
        result = self.pipeline.query(enriched)
 
        return {
            'log':              log_text,
            'flagged_keywords': found,
            'is_anomaly':       len(found) > 0,
            'rag_analysis':     result['answer'],
            'sources':          result['sources'],
        }
 