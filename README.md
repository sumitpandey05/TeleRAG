# TeleRAG - RAG based future ready Telecom RAN assistant

A Retrieval-Augmented Generation (RAG) system tailored for Telecom Radio Access Network (RAN) tasks, built for the **RAG-Based Future-Ready Telecom RAN Assistant Hackathon**.

TeleRAG automates and accelerates decision-making for telecom engineers by combining domain-specific knowledge retrieval with LLM-based reasoning — covering 3GPP spec understanding, O-RAN architecture, root cause analysis, and anomaly detection.

## Problem Statement

Telecom RANs are growing in complexity and scale, making tasks like root cause analysis, anomaly detection, and network optimization increasingly difficult to manage with traditional tools. Manual intervention by Subject Matter Experts (SMEs) leads to delays, inefficiencies, and scalability issues.

TeleRAG addresses this by providing an intelligent assistant that:
- Answers 3GPP specification and O-RAN architecture questions
- Performs automated multi-step root cause analysis
- Detects anomalies in RAN logs and performance metrics
- Cites exact sources (spec name + page number) for every answer

---




---
### Two-Stage Retrieval
1. **Stage 1 — FAISS Semantic Search**: Retrieves top-30 candidate chunks using `all-MiniLM-L6-v2` embeddings
2. **Stage 2 — CrossEncoder Reranking**: Reranks all 30 candidates using `ms-marco-MiniLM-L-6-v2` for precise relevance scoring
3. **Stage 3 — Diversity Filter**: Caps TeleQnA results at 2 per query to ensure 3GPP/O-RAN spec chunks are always represented

### Multi-Step RCA
Root cause analysis uses a **4-step reasoning chain**:
1. Retrieve context on the reported symptom
2. Ask LLM to identify root cause from context
3. Re-retrieve using the identified root cause
4. Synthesize final structured answer (SYMPTOM → ROOT CAUSE → AFFECTED COMPONENTS → RECOMMENDED FIX → SOURCES)

## Features

| Feature | Description |
|---|---|
| **3GPP QnA** | Answers questions grounded in 3GPP Release 16/18/19 specifications |
| **O-RAN Understanding** | Answers questions from O-RAN SC working group specifications |
| **Root Cause Analysis** | Multi-step reasoning chain for RAN fault diagnosis |
| **Anomaly Detection** | Keyword pre-screening + RAG-based analysis of RAN logs |
| **Source Citation** | Every answer cites exact document name and page number |
| **PII Sanitization** | Redacts IMEI, IMSI, IP addresses, MAC addresses from all queries |
| **Response Timing** | Tracks retrieval and generation time separately |
| **Explainability** | Answers indicate whether they come from context or general knowledge |

## Datasets

| Dataset | Size | Usage |
|---|---|---|
| **TeleQnA** | 10,000 Q&A pairs | Domain-specific QnA, evaluation ground truth |
| **O-RAN SC Specs** | 5 working group PDFs | O-RAN architecture, E2AP, E2SM-RC, E2SM-KPM, OAD, SMO |
| **3GPP TS 38.211** | Physical layer spec | PDCCH, physical channels |
| **3GPP TS 38.213** | Physical layer procedures | Control channel procedures |
| **3GPP TS 38.214** | Physical layer procedures | Data channel procedures |
| **3GPP TS 38.300** | NR overall description | Overall 5G NR architecture |
| **3GPP TS 38.321** | MAC layer | MAC procedures |
| **3GPP TS 38.331** | RRC | Radio Resource Control |
| **3GPP TS 38.401** | NG-RAN architecture | Network architecture |
(still increasing the data set specs)
**Total indexed vectors: 18,931**


## Acknowledgements

- [TeleQnA Dataset](https://github.com/netop-team/TeleQnA) — netop-team
- [Tele-LLMs](https://github.com/Ali-maatouk/Tele-LLMs) — Ali Maatouk
- [O-RAN SC Specifications](https://oransc.org/specifications) — O-RAN Software Community
- [3GPP Specifications](https://www.3gpp.org) — 3rd Generation Partnership Project
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — Georgi Gerganov