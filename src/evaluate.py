# src/evaluate.py
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
# from datasets import Dataset
from sentence_transformers import SentenceTransformer, util
from langchain_ollama import OllamaLLM as Ollama
from src.pipeline import TeleRAGPipeline

TELEQNA_PATH  = 'data/teleqna/TeleQnA.json'
NUM_QUESTIONS = 10        # raise to 50 for final run
EMBED_MODEL   = 'sentence-transformers/all-MiniLM-L6-v2'
JUDGE_MODEL = 'llama3.1:8b'

DOC_THRESHOLD = 0.75
SENT_THRESHOLD = 0.40      # cosine similarity threshold

KPI_TARGETS = {
    'mrr':               0.75,
    'topk_accuracy':     0.85,
    'accuracy':          0.80,
    'faithfulness':      0.90,
    'context_recall':    0.85,
    'context_precision': 0.75,
}

# ─────────────────────────────────────────────────────────────
# Load held-out test questions
# ─────────────────────────────────────────────────────────────
def load_test_questions(path, n):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_items  = list(data.items())
    test_start = int(len(all_items) * 0.8)
    subset     = all_items[test_start : test_start + n]

    questions, ground_truths, gt_chunks = [], [], []

    for key, item in subset:
        q           = item.get('question', '').strip()
        answer_raw  = item.get('answer', '')
        explanation = item.get('explanation', '')
        category    = item.get('category', '')

        answer_text = (
            answer_raw.split(':', 1)[1].strip()
            if ':' in answer_raw
            else answer_raw.strip()
        )

        if q and answer_text:
            questions.append(q)
            ground_truths.append(answer_text)
            gt_chunks.append(
                f"Question: {q}\n"
                f"Answer: {answer_text}\n"
                f"Explanation: {explanation}\n"
                f"Category: {category}"
            )

    return questions, ground_truths, gt_chunks


# ─────────────────────────────────────────────────────────────
# SBERT singleton
# ─────────────────────────────────────────────────────────────
_sbert = None

def _get_sbert():
    global _sbert
    if _sbert is None:
        print('  Loading SBERT...')
        _sbert = SentenceTransformer(EMBED_MODEL)
    return _sbert

_llm = None
def _get_llm():
    # llama3.1:8b used ONLY for faithfulness YES/NO calls
    # Not tele-llm — that model is too slow for 50 sequential calls
    global _llm
    if _llm is None:
        print(f'  Loading LLM judge ({JUDGE_MODEL})...')
        _llm = Ollama(model=JUDGE_MODEL, temperature=0)
    return _llm

# ─────────────────────────────────────────────────────────────
# All 6 KPI metrics — fully embedding-based, no LLM judge needed
# ─────────────────────────────────────────────────────────────

def compute_mrr(questions, vectorstore, gt_chunks, k=10):
    sbert = _get_sbert()
    ranks = []

    for q, gt in zip(questions, gt_chunks):
        retrieved  = vectorstore.similarity_search(q, k=k)
        gt_emb     = sbert.encode(gt, convert_to_tensor=True)
        found_rank = None

        for rank, doc in enumerate(retrieved, start=1):
            doc_emb = sbert.encode(doc.page_content, convert_to_tensor=True)
            if float(util.cos_sim(gt_emb, doc_emb)) >= DOC_THRESHOLD:
                found_rank = rank
                break
        ranks.append(1.0 / found_rank if found_rank else 0.0)
    return float(np.mean(ranks))


def compute_topk_accuracy(questions, vectorstore, gt_chunks, k=5):
    sbert = _get_sbert()
    hits = 0
    for q, gt in zip(questions, gt_chunks):
        retrieved = vectorstore.similarity_search(q, k=k)
        gt_emb    = sbert.encode(gt, convert_to_tensor=True)
        for doc in retrieved:
            doc_emb = sbert.encode(doc.page_content, convert_to_tensor=True)
            if float(util.cos_sim(gt_emb, doc_emb)) >=  DOC_THRESHOLD:
                hits += 1
                break
    return hits / len(questions)


def compute_accuracy(generated_answers, ground_truths,questions):
    
    sbert  = _get_sbert()
    scores = []

    for answer, gt, q in zip(generated_answers, ground_truths, questions):
        if gt.lower().strip() in answer.lower():
            scores.append(1.0)
            continue
 
        # Check 2 — SBERT best-sentence similarity (handles verbose answers)
        sentences = [
            s.strip() for s in answer.replace('\n', '. ').split('.')
            if len(s.strip()) > 10
        ]
        best_sbert = 0.0

        if sentences:
            gt_emb     = sbert.encode(gt, convert_to_tensor=True)
            sent_embs  = sbert.encode(sentences, convert_to_tensor=True)
            best_sbert = max(float(util.cos_sim(se, gt_emb)) for se in sent_embs)

        # If SBERT is already high enough, use it directly
        if best_sbert >= 0.70:
            scores.append(best_sbert)
            continue

        # Check 3 — LLM judge for rephrased answers
        # Catches: "minimum of nt and nr" vs ground truth "min(nt, nr)"
        prompt = (
            f"QUESTION: {q}\n"
            f"CORRECT ANSWER: {gt}\n"
            f"STUDENT RESPONSE: {answer[:500]}\n\n"
            f"Does the student response correctly convey the answer '{gt}'?\n"
            f"Reply with YES or NO only. Do not explain."
        )
        if _ask_yes_no(prompt):
            scores.append(1.0)
        else:
            scores.append(best_sbert)   # use SBERT score as partial credit

    return float(np.mean(scores))

def _ask_yes_no(prompt: str) -> bool:
    llm      = _get_llm()
    response = llm.invoke(prompt).strip().lower()
    return 'yes' in response[:20]


def compute_faithfulness(answers, contexts_list):
    
    print('  Scoring Faithfulness with LLM judge...')
    scores = []
 
    for i, (answer, contexts) in enumerate(zip(answers, contexts_list)):
        print(f'    Q{i+1}/{len(answers)}', end=' ', flush=True)
 
        # Top 3 context chunks only — keeps prompt short for reliable response
        context_text = '\n---\n'.join(contexts[:3])
 
        sentences = [
            s.strip() for s in answer.replace('\n', '. ').split('.')
            if len(s.strip()) > 15
        ]
        if not sentences:
            scores.append(0.0)
            print('(no sentences found)')
            continue
 
        yes_count   = 0
        check_sents = sentences[:5]   # check up to 5 sentences per answer
 
        for sent in check_sents:
            prompt = (
                f"CONTEXT:\n{context_text}\n\n"
                f"CLAIM: {sent}\n\n"
                f"Is this claim supported by the context above?\n"
                f"Reply with YES or NO only. Do not explain."
            )
            if _ask_yes_no(prompt):
                yes_count += 1
 
        score = yes_count / len(check_sents)
        scores.append(score)
        print(f'({yes_count}/{len(check_sents)} sentences supported)')
 
    return float(np.mean(scores))


def compute_context_recall(questions, vectorstore, gt_chunks, k=10):
    sbert  = _get_sbert()
    scores = []
 
    for q, gt in zip(questions, gt_chunks):
        # Raw vectorstore — bypasses diversity filter
        retrieved = vectorstore.similarity_search(q, k=k)
        gt_emb    = sbert.encode(gt, convert_to_tensor=True)
 
        found = any(
            float(util.cos_sim(
                gt_emb,
                sbert.encode(doc.page_content, convert_to_tensor=True)
            )) >= DOC_THRESHOLD
            for doc in retrieved
        )
        scores.append(1.0 if found else 0.0)
 
    return float(np.mean(scores))


def compute_context_precision(questions, gt_chunks, contexts_list):
    sbert  = _get_sbert()
    scores = []
 
    for q, gt, contexts in zip(questions, gt_chunks, contexts_list):
        if not contexts:
            scores.append(0.0)
            continue
 
        q_emb  = sbert.encode(q,  convert_to_tensor=True)
        gt_emb = sbert.encode(gt, convert_to_tensor=True)
 
        relevant = 0
        for ctx in contexts:
            ctx_emb = sbert.encode(ctx, convert_to_tensor=True)
            sim_q   = float(util.cos_sim(ctx_emb, q_emb))
            sim_gt  = float(util.cos_sim(ctx_emb, gt_emb))
            if max(sim_q, sim_gt) >= SENT_THRESHOLD:
                relevant += 1
 
        scores.append(relevant / len(contexts))
 
    return float(np.mean(scores))


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def run_eval():
    print('=' * 50)
    print('TeleRAG Evaluation')
    print('=' * 50)
 
    print('\nLoading TeleRAG pipeline...')
    pipeline = TeleRAGPipeline()
 
    print(f'\nLoading {NUM_QUESTIONS} held-out test questions...')
    questions, ground_truths, gt_chunks = load_test_questions(
        TELEQNA_PATH, NUM_QUESTIONS
    )
 
    # ── Run all questions through the pipeline ────────────────
    print(f'\nRunning {len(questions)} questions through pipeline...')
    answers, contexts = [], []
    for i, q in enumerate(questions):
        print(f'  {i+1}/{len(questions)}: {q[:65]}...')
        r = pipeline.query(q)
        answers.append(r['answer'])
        contexts.append([doc.page_content for doc, _ in r['retrieved']])
 
    vectorstore = pipeline.vectorstore
 
    # ── Compute all 6 metrics ─────────────────────────────────
    print('\nComputing metrics...')
 
    print('  [1/6] MRR...')
    mrr = compute_mrr(questions, vectorstore, gt_chunks, k=10)
 
    print('  [2/6] Top-k Accuracy...')
    topk_acc = compute_topk_accuracy(questions, vectorstore, gt_chunks, k=5)
 
    print('  [3/6] Accuracy...')
    accuracy = compute_accuracy(answers, ground_truths, questions)
 
    print('  [4/6] Context Recall...')
    recall = compute_context_recall(questions, vectorstore, gt_chunks, k=10)
 
    print('  [5/6] Context Precision...')
    precision = compute_context_precision(questions, gt_chunks, contexts)
 
    print('  [6/6] Faithfulness (LLM judge — this takes ~20-30 min on CPU)...')
    faith = compute_faithfulness(answers, contexts)
 
    # ── Print results ─────────────────────────────────────────
    final_scores = {
        'mrr':               mrr,
        'topk_accuracy':     topk_acc,
        'accuracy':          accuracy,
        'faithfulness':      faith,
        'context_recall':    recall,
        'context_precision': precision,
    }
 
    print('\n========== TELERAG EVALUATION RESULTS ==========')
    for metric, value in final_scores.items():
        target = KPI_TARGETS[metric]
        status = '✓ PASS' if value >= target else '✗ FAIL'
        print(f'  {metric:<22} {value:.3f}   (target > {target})  {status}')
    print('=================================================')
 
    pd.DataFrame([final_scores]).to_csv('evaluation_results.csv', index=False)
    print('Saved to evaluation_results.csv')
 
 
if __name__ == '__main__':
    run_eval()