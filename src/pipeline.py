# src/pipeline.py
import re, time, hashlib, json
from pathlib import Path
from langchain_ollama import OllamaLLM as Ollama
from src.retriever import TeleRAGRetriever

LLM_MODEL = 'tele-llm'
CACHE_DIR  = Path('cache')


class TeleRAGPipeline:
    def __init__(self):
        self.retriever   = TeleRAGRetriever()
        self.vectorstore = self.retriever.vectorstore  # expose for evaluator
        self.llm         = Ollama(model=LLM_MODEL, temperature=0, timeout=300)

        with open('prompts/system.txt', 'r') as f:
            self.system_prompt = f.read()

        # CHANGE 4: create cache directory on startup
        CACHE_DIR.mkdir(exist_ok=True)

        print('Pipeline Ready')

    # ── CHANGE 4: query cache ──────────────────────────────────
    def _cache_key(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def _load_cache(self, query: str):
        path = CACHE_DIR / f'{self._cache_key(query)}.json'
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    def _save_cache(self, query: str, result: dict):
        path      = CACHE_DIR / f'{self._cache_key(query)}.json'
        # Don't cache 'retrieved' — Document objects aren't JSON serializable
        cacheable = {k: v for k, v in result.items() if k != 'retrieved'}
        with open(path, 'w') as f:
            json.dump(cacheable, f)

    # ── PII sanitizer — unchanged ──────────────────────────────
    def _sanitize(self, text: str) -> str:
        text = re.sub(r'\b\d{15}\b',
                      '[IMEI_REDACTED]', text)
        text = re.sub(r'\b\d{14,15}\b',
                      '[IMSI_REDACTED]', text)
        text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
                      '[IP_REDACTED]', text)
        text = re.sub(r'\b[0-9A-Fa-f]{2}(:[0-9A-Fa-f]{2}){5}\b',
                      '[MAC_REDACTED]', text)
        return text

    # ── Format context — unchanged ─────────────────────────────
    def _format_context(self, docs):
        parts = []
        for i, (doc, score) in enumerate(docs):
            src  = doc.metadata.get('source', 'unknown')
            page = doc.metadata.get('page', '?')
            parts.append(
                f'[Context {i+1}] Source: {src}, Page: {page}\n'
                f'{doc.page_content}'
            )
        return '\n\n---\n\n'.join(parts)

    # ── CHANGE 5: confidence scoring ──────────────────────────
    def _compute_confidence(self, retrieved: list) -> str:
        """
        Rate confidence based on top CrossEncoder reranking score.
        Scores are sigmoid-normalized to 0-1 range.
        HIGH >= 0.85, MEDIUM >= 0.60, LOW < 0.60
        """
        if not retrieved:
            return 'LOW'
        top_score = retrieved[0][1]
        if top_score >= 0.85:
            return 'HIGH'
        if top_score >= 0.60:
            return 'MEDIUM'
        return 'LOW'

    # ── Format sources ─────────────────────────────────────────
    def _format_sources(self, docs: list) -> list:
        return [
            {
                'source': d.metadata.get('source', '?'),
                'page':   d.metadata.get('page',   '?'),
                'type':   d.metadata.get('type',   '?'),
                'score':  round(float(s), 3),
            }
            for d, s in docs
        ]

    # ── Standard QnA query ────────────────────────────────────
    def query(self, question: str, verbose: bool = False,
              use_cache: bool = True) -> dict:

        question = self._sanitize(question)

        # CHANGE 4: check cache before doing any work
        if use_cache:
            cached = self._load_cache(question)
            if cached:
                print('  [cache hit]')
                return cached

        t0        = time.time()
        retrieved = self.retriever.retrieve(question)
        t_ret     = time.time() - t0

        if verbose:
            for doc, score in retrieved:
                print(f'  [{score:.3f}] {doc.metadata["source"]}')

        context = self._format_context(retrieved)
        prompt  = (
            f'{self.system_prompt}\n\n'
            f'CONTEXT PASSAGES:\n{context}\n\n'
            f'QUESTION: {question}'
        )

        t1      = time.time()
        answer  = self.llm.invoke(prompt)
        t_gen   = time.time() - t1

        result = {
            'question':          question,
            'answer':            answer,
            'sources':           self._format_sources(retrieved),
            'retrieved':         retrieved,
            'confidence':        self._compute_confidence(retrieved),  # CHANGE 5
            'retrieval_time_s':  round(t_ret, 3),
            'generation_time_s': round(t_gen, 3),
            'total_time_s':      round(t_ret + t_gen, 3),
        }

        # CHANGE 4: save result to cache
        if use_cache:
            self._save_cache(question, result)

        return result

    # ── Multi-step RCA query ───────────────────────────────────
    def rca_query(self, issue: str) -> dict:
        issue = self._sanitize(issue)

        # Step 1 — retrieve on the symptom
        step1 = self.retriever.retrieve(issue)
        ctx1  = self._format_context(step1)

        # Step 2 — identify root cause from context
        cause_prompt = (
            f'{self.system_prompt}\n\n'
            f'CONTEXT:\n{ctx1}\n\n'
            f'TASK: Based ONLY on the context, identify the most likely '
            f'root cause of this issue in one sentence starting with '
            f'"ROOT CAUSE:"\n\n'
            f'ISSUE: {issue}'
        )
        root_cause = self.llm.invoke(cause_prompt).strip()
        if not root_cause:
            root_cause = 'ROOT CAUSE: Unable to determine from available context.'

        # Step 3 — retrieve again using identified root cause
        step2 = self.retriever.retrieve(root_cause)
        ctx2  = self._format_context(step2)

        # Step 4 — synthesize final structured answer
        final_prompt = (
            f'{self.system_prompt}\n\n'
            f'CONTEXT:\n{ctx1}\n\n---\n\n{ctx2}\n\n'
            f'ISSUE: {issue}\n'
            f'IDENTIFIED CAUSE: {root_cause}\n\n'
            f'Provide a complete answer in EXACTLY this format:\n'
            f'SYMPTOM: ...\n'
            f'ROOT CAUSE: ...\n'
            f'AFFECTED COMPONENTS: ...\n'
            f'RECOMMENDED FIX: ...\n'
            f'SOURCES: ...'
        )
        final_answer = self.llm.invoke(final_prompt).strip()

        # Fallback if model returns empty
        if not final_answer:
            fallback = (
                f'{self.system_prompt}\n\n'
                f'CONTEXT:\n{ctx1}\n\n'
                f'Analyze this issue and give SYMPTOM, ROOT CAUSE, '
                f'AFFECTED COMPONENTS, RECOMMENDED FIX:\n\n{issue}'
            )
            final_answer = self.llm.invoke(fallback).strip()

        # Deduplicate retrieved docs from both steps
        seen, all_docs = set(), []
        for doc, score in step1 + step2:
            key = (doc.metadata.get('source'), doc.metadata.get('page'))
            if key not in seen:
                seen.add(key)
                all_docs.append((doc, score))

        return {
            'question':        issue,
            'root_cause_step': root_cause,
            'answer':          final_answer,
            'retrieved':       all_docs,
            'confidence':      self._compute_confidence(all_docs),  # CHANGE 5
            'sources':         self._format_sources(all_docs),
        }


if __name__ == '__main__':
    p = TeleRAGPipeline()
    for q in ['What is PDCCH in 5G NR?', 'What does gNB stand for?']:
        print('\n' + '='*60)
        r = p.query(q)
        print(f'Q: {q}')
        print(f'A: {r["answer"]}')
        print(f'Confidence: {r["confidence"]}')
        print(f'Time: {r["total_time_s"]}s')