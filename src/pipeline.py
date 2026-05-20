import re
import time
from langchain_ollama import OllamaLLM as Ollama
from src.retriever import TeleRAGRetriever
 
LLM_MODEL = 'tele-llm'
class TeleRAGPipeline:
    def __init__(self):
        self.retriever    = TeleRAGRetriever()
        self.vectorstore  = self.retriever.vectorstore   # expose for evaluator
        self.llm          = Ollama(model=LLM_MODEL, temperature=0, timeout = 300)
        with open('prompts/system.txt', 'r') as f:
            self.system_prompt = f.read()
        print('Pipeline Ready')
 
    # ── Sanitize sensitive identifiers ────────────────────────
    def _sanitize(self, text: str) -> str:
        text = re.sub(r'\b\d{15}\b',                    '[IMEI_REDACTED]', text)
        text = re.sub(r'\b\d{14,15}\b',                 '[IMSI_REDACTED]', text)
        text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b',  '[IP_REDACTED]',   text)
        text = re.sub(r'\b[0-9A-Fa-f]{2}(:[0-9A-Fa-f]{2}){5}\b', '[MAC_REDACTED]', text)
        return text
 
    # ── Format retrieved docs into context string ──────────────
    def _format_context(self, docs):
        parts = []
        for i, (doc, score) in enumerate(docs):
            src  = doc.metadata.get('source', 'unknown')
            page = doc.metadata.get('page', '?')
            parts.append(f'[Context {i+1}] Source: {src}, Page: {page}\n{doc.page_content}')
        return '\n\n---\n\n'.join(parts)
 
    # ── Standard single-step QnA query ────────────────────────
    def query(self, question, verbose=False):
        question = self._sanitize(question)
 
        t0        = time.time()
        retrieved = self.retriever.retrieve(question)
        t_ret     = time.time() - t0
 
        if verbose:
            for doc, score in retrieved:
                print(f'  [{score:.2f}] {doc.metadata["source"]}')
 
        context = self._format_context(retrieved)
        prompt  = f'{self.system_prompt}\n\nCONTEXT PASSAGES:\n{context}\n\nQUESTION: {question}'
 
        t1      = time.time()
        answer  = self.llm.invoke(prompt)
        t_gen   = time.time() - t1
 
        sources = [
            {
                'source': d.metadata.get('source', '?'),
                'page':   d.metadata.get('page', '?'),
                'type':   d.metadata.get('type', '?'),
                'score':  round(float(s), 3),
            }
            for d, s in retrieved
        ]
 
        return {
            'question':          question,
            'answer':            answer,
            'sources':           sources,
            'retrieved':         retrieved,
            'retrieval_time_s':  round(t_ret, 3),
            'generation_time_s': round(t_gen, 3),
            'total_time_s':      round(t_ret + t_gen, 3),
        }
 
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
            f'TASK: Based ONLY on the context, identify the most likely root cause '
            f'of this issue in one sentence starting with "ROOT CAUSE:"\n\n'
            f'ISSUE: {issue}'
        )
        root_cause = self.llm.invoke(cause_prompt)
 
        if not root_cause:
            root_cause = f'ROOT CAUSE: Unable to determine from context for issue: {issue}'


        # Step 3 — retrieve again using the identified cause
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
        final_answer = self.llm.invoke(final_prompt)

        if not final_answer:
            fallback_prompt = (
                f'{self.system_prompt}\n\n'
                f'CONTEXT:\n{ctx1}\n\n'
                f'Analyze this issue and give SYMPTOM, ROOT CAUSE, '
                f'AFFECTED COMPONENTS, RECOMMENDED FIX:\n\n{issue}'
            )
            final_answer = self.llm.invoke(fallback_prompt).strip()
 
        # Merge retrieved docs (deduplicated)
        seen     = set()
        all_docs = []
        for doc, score in step1 + step2:
            key = (doc.metadata.get('source'), doc.metadata.get('page'))
            if key not in seen:
                seen.add(key)
                all_docs.append((doc, score))
 
        return {
            'question':             issue,
            'root_cause_step':      root_cause,
            'answer':               final_answer,
            'retrieved':            all_docs,
            'sources': [
                {
                    'source': d.metadata.get('source', '?'),
                    'page':   d.metadata.get('page', '?'),
                    'type':   d.metadata.get('type', '?'),
                    'score':  round(float(s), 3),
                }
                for d, s in all_docs
            ],
        }
 
 
if __name__ == '__main__':
    p = TeleRAGPipeline()
    for q in ['What is handover in 5G?', 'What does gNB stand for?']:
        print('\n' + '='*60)
        r = p.query(q)
        print(f'Q: {q}\nA: {r["answer"]}')
        print(f'Time: {r["total_time_s"]}s')