# src/evaluate.py
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer, util
from langchain_ollama import OllamaLLM as Ollama
from src.pipeline import TeleRAGPipeline

TELEQNA_PATH  = 'data/teleqna/TeleQnA.json'
NUM_QUESTIONS = 10        # raise to 50 for final run
EMBED_MODEL   = 'sentence-transformers/all-MiniLM-L6-v2'
OLLAMA_MODEL = 'tele-llm'
SIM_THRESHOLD = 0.75      # cosine similarity threshold


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


# ─────────────────────────────────────────────────────────────
# All 6 KPI metrics — fully embedding-based, no LLM judge needed
# ─────────────────────────────────────────────────────────────

def compute_mrr(questions, vectorstore, gt_chunks, k=10):
    """Rank of first retrieved chunk that matches the ground-truth document."""
    sbert = _get_sbert()
    reciprocal_ranks = []
    for q, gt in zip(questions, gt_chunks):
        retrieved  = vectorstore.similarity_search(q, k=k)
        gt_emb     = sbert.encode(gt, convert_to_tensor=True)
        found_rank = None
        for rank, doc in enumerate(retrieved, start=1):
            doc_emb = sbert.encode(doc.page_content, convert_to_tensor=True)
            if float(util.cos_sim(gt_emb, doc_emb)) >= SIM_THRESHOLD:
                found_rank = rank
                break
        reciprocal_ranks.append(1.0 / found_rank if found_rank else 0.0)
    return float(np.mean(reciprocal_ranks))


def compute_topk_accuracy(questions, vectorstore, gt_chunks, k=5):
    """Is the ground-truth document present anywhere in the top-k results?"""
    sbert = _get_sbert()
    hits = 0
    for q, gt in zip(questions, gt_chunks):
        retrieved = vectorstore.similarity_search(q, k=k)
        gt_emb    = sbert.encode(gt, convert_to_tensor=True)
        for doc in retrieved:
            doc_emb = sbert.encode(doc.page_content, convert_to_tensor=True)
            if float(util.cos_sim(gt_emb, doc_emb)) >= SIM_THRESHOLD:
                hits += 1
                break
    return hits / len(questions)


def compute_accuracy(generated_answers, ground_truths):
    """
    Best-sentence accuracy: find the sentence in the generated answer
    that best matches the ground truth, rather than comparing the full
    verbose answer. Fixes the long-answer vs short-ground-truth penalty.
    """
    sbert  = _get_sbert()
    scores = []
    gt_embs = sbert.encode(ground_truths, convert_to_tensor=True)

    for answer, gt_emb in zip(generated_answers, gt_embs):
        # Split answer into sentences, score each against ground truth
        sentences = [s.strip() for s in answer.replace('\n', '. ').split('.') if len(s.strip()) > 10]
        if not sentences:
            scores.append(0.0)
            continue
        sent_embs  = sbert.encode(sentences, convert_to_tensor=True)
        # Take the best matching sentence
        best_score = max(float(util.cos_sim(se, gt_emb)) for se in sent_embs)
        scores.append(best_score)

    return float(np.mean(scores))


def compute_faithfulness(answers, contexts_list):
    """
    Embedding-based faithfulness: fraction of answer sentences that are
    semantically covered by the retrieved context.
    Replaces RAGAS LLM judge (which returns nan with 8B models on CPU).
    """
    sbert  = _get_sbert()
    scores = []
    for answer, contexts in zip(answers, contexts_list):
        sentences = [s.strip() for s in answer.replace('\n', '. ').split('.') if len(s.strip()) > 10]
        if not sentences:
            scores.append(0.0)
            continue
        # Encode all context as one block
        context_text = ' '.join(contexts)
        ctx_emb      = sbert.encode(context_text, convert_to_tensor=True)
        covered = sum(
            1 for s in sentences
            if float(util.cos_sim(sbert.encode(s, convert_to_tensor=True), ctx_emb)) >= SIM_THRESHOLD
        )
        scores.append(covered / len(sentences))
    return float(np.mean(scores))


def compute_context_recall(ground_truths, contexts_list):
    """
    Embedding-based context recall: fraction of ground-truth sentences
    that are semantically present in the retrieved context.
    """
    sbert  = _get_sbert()
    scores = []
    for gt, contexts in zip(ground_truths, contexts_list):
        gt_sents = [s.strip() for s in gt.replace('\n', '. ').split('.') if len(s.strip()) > 10]
        if not gt_sents:
            scores.append(1.0)
            continue
        context_text = ' '.join(contexts)
        ctx_emb      = sbert.encode(context_text, convert_to_tensor=True)
        found = sum(
            1 for s in gt_sents
            if float(util.cos_sim(sbert.encode(s, convert_to_tensor=True), ctx_emb)) >= SIM_THRESHOLD
        )
        scores.append(found / len(gt_sents))
    return float(np.mean(scores))


def compute_context_precision(questions, contexts_list):
    """
    Embedding-based context precision: fraction of retrieved chunks
    that are semantically relevant to the question.
    """
    sbert  = _get_sbert()
    scores = []
    for q, contexts in zip(questions, contexts_list):
        if not contexts:
            scores.append(0.0)
            continue
        q_emb = sbert.encode(q, convert_to_tensor=True)
        relevant = sum(
            1 for ctx in contexts
            if float(util.cos_sim(sbert.encode(ctx, convert_to_tensor=True), q_emb)) >= SIM_THRESHOLD
        )
        scores.append(relevant / len(contexts))
    return float(np.mean(scores))


# ─────────────────────────────────────────────────────────────
# KPI targets
# ─────────────────────────────────────────────────────────────
KPI_TARGETS = {
    'mrr':               0.75,
    'topk_accuracy':     0.85,
    'accuracy':          0.80,
    'faithfulness':      0.90,
    'context_recall':    0.85,
    'context_precision': 0.75,
}


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def run_eval():
    print('Loading TeleRAG pipeline...')
    pipeline = TeleRAGPipeline()

    print(f'Loading {NUM_QUESTIONS} test questions...')
    questions, ground_truths, gt_chunks = load_test_questions(TELEQNA_PATH, NUM_QUESTIONS)

    print(f'\nRunning {len(questions)} questions through pipeline...')
    answers, contexts = [], []
    for i, q in enumerate(questions):
        print(f'  {i+1}/{len(questions)}: {q[:60]}...')
        r = pipeline.query(q)
        answers.append(r['answer'])
        contexts.append([doc.page_content for doc, _ in r['retrieved']])

    print('\nComputing all 6 KPI metrics (embedding-based, no LLM judge)...')
    vectorstore = pipeline.vectorstore

    final_scores = {
        'mrr':               compute_mrr(questions, vectorstore, gt_chunks, k=10),
        'topk_accuracy':     compute_topk_accuracy(questions, vectorstore, gt_chunks, k=5),
        'accuracy':          compute_accuracy(answers, ground_truths),
        'faithfulness':      compute_faithfulness(answers, contexts),
        'context_recall':    compute_context_recall(ground_truths, contexts),
        'context_precision': compute_context_precision(questions, contexts),
    }

    print('\n========== TELERAG EVALUATION RESULTS ==========')
    for metric, value in final_scores.items():
        target = KPI_TARGETS[metric]
        status = '✓ PASS' if value >= target else '✗ FAIL'
        print(f'  {metric:<22} {value:.3f}   (target > {target})  {status}')
    print('=================================================')

    summary_df = pd.DataFrame([final_scores])
    summary_df.to_csv('evaluation_results.csv', index=False)
    print('Saved to evaluation_results.csv')


if __name__ == '__main__':
    run_eval()