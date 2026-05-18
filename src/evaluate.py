import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, ContextRecall, ContextPrecision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer, util
from src.pipeline import TeleRAGPipeline

TELEQNA_PATH  = 'data/teleqna/TeleQnA.json'
NUM_QUESTIONS = 10        # keep at 10 while testing; raise to 50 for final run
OLLAMA_MODEL  = 'llama3.1:8b'
EMBED_MODEL   = 'sentence-transformers/all-MiniLM-L6-v2'
SIM_THRESHOLD = 0.75           # Cosine similarity threshold for MRR / Top-K

# ── Load held-out test questions ──────────────────────────────
def load_test_questions(path, n):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_items  = list(data.values())
    test_start = int(len(all_items) * 0.8)
    subset     = all_items[test_start : test_start + n]
    questions, ground_truths = [], []
    for item in subset:
        q          = item.get('question', '').strip()
        answer_raw = item.get('answer', '')
        answer_text = (
            answer_raw.split(':', 1)[1].strip()
            if ':' in answer_raw
            else answer_raw.strip()
        )
        if q and answer_text:
            questions.append(q)
            ground_truths.append(answer_text)
    return questions, ground_truths

_sbert = None

def _get_sbert():
    global _sbert
    if _sbert is None:
        print('loading sbert for custom metrics')
        _sbert = SentenceTransformer(EMBED_MODEL)
    return _sbert

def compute_mrr (questions, vectorstore, ground_truth_chunks, k = 10):
    """Mean Reciprocal Rank — at what rank does the relevant chunk appear?"""
    sbert = _get_sbert()
    reciprocal_ranks = []
    for q, gt_chunk in zip(questions, ground_truth_chunks):
        retrieved = vectorstore.similarity_search(q, k=k)
        gt_emb = sbert.encode(gt_chunk, convert_to_tensor=True)
        found_rank = None
        for rank, doc in enumerate(retrieved, start=1):
            doc_emb = sbert.encode(doc.page_content, convert_to_tensor=True)
            if float(util.cos_sim(gt_emb, doc_emb)) >= SIM_THRESHOLD:
                found_rank = rank
                break

        reciprocal_ranks.append(1.0/found_rank if found_rank else 0.0)
    return float(np.mean(reciprocal_ranks))

def compute_topk_accuracy(questions,vectorstore, ground_truth_chunks, k = 10):
    """Top-k Accuracy — is the relevant chunk anywhere in the top-k results?"""
    sbert = _get_sbert()
    hits = 0
    for q, gt_chunk in zip(questions, ground_truth_chunks):
        retrieved = vectorstore.similarity_search(q, k=k)
        gt_emb = sbert.encode(gt_chunk, convert_to_tensor=True)
        for doc in retrieved:
            doc_emb = sbert.encode(doc.page_content, convert_to_tensor=True)
            if float(util.cos_sim(gt_emb, doc_emb)) >= SIM_THRESHOLD:
                hits += 1
                break
    return hits/len(questions)

def compute_accuracy(generated_answers, ground_truth_chunks):
    """Semantic accuracy — cosine similarity between generated and reference answers."""
    sbert = _get_sbert()
    gen_embs = sbert.encode(generated_answers, convert_to_tensor=True)
    gt_embs = sbert.encode(ground_truth_chunks, convert_to_tensor = True)
    scores = [float(util.cos_sim(g,r)) for g,r in zip(gen_embs, gt_embs)]

    return float(np.mean(scores))

KPI_TARGETS =  {
    'mrr': 0.75,
    'topk_accuracy': 0.85,
    'accuracy': 0.80,
    'context_recall':    0.85,
    'faithfulness':      0.90,
    'context_precision':0.75,
}

# ── Main ──────────────────────────────────────────────────────
def run_eval():
    print('Setting up Ollama for RAGAs (no OpenAI needed)...')

    # Longer timeout stops Ollama from timing out on slow CPU
    ragas_llm = LangchainLLMWrapper(
        ChatOllama(
            model=OLLAMA_MODEL, 
            temperature=0, 
            timeout=180, 
            format = 'json',
        )
    )


    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    )
    # 3 metrics only — AnswerRelevancy removed (slowest, needs embeddings)

    metrics = [
        Faithfulness(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
        ContextPrecision(llm=ragas_llm),
    ]

    print('Loading TeleRAG pipeline...')
    pipeline = TeleRAGPipeline()

    print(f'Loading {NUM_QUESTIONS} test questions...')
    questions, ground_truths = load_test_questions(TELEQNA_PATH, NUM_QUESTIONS)

    print(f'\nRunning {len(questions)} questions through pipeline...')
    answers, contexts, gt_chunks = [], [], []

    for i, (q,gt) in enumerate(zip(questions, ground_truths)):
        print(f'  {i+1}/{len(questions)}: {q[:60]}...')
        r = pipeline.query(q)
        answers.append(r['answer'])
        ctx_texts = [doc.page_content for doc, _ in r['retrieved']]

        contexts.append(ctx_texts)

        gt_chunks.append(gt)

    ds = Dataset.from_dict({
        'question':     questions,
        'answer':       answers,
        'contexts':     contexts,
        'ground_truth': ground_truths
    })

    print('\nScoring with RAGAs — this takes a few minutes per question...')
    results = evaluate(
        dataset=ds, 
        metrics=metrics,
        embeddings = ragas_embeddings,
        raise_exceptions = False,
    )

    # ragas 0.2.x returns lists — take the mean of each
    scores = results.to_pandas()
    faith     = float(scores['faithfulness'].mean())
    recall    = float(scores['context_recall'].mean())
    precision = float(scores['context_precision'].mean())

    print('\nComputing custom KPI metrics (MRR, Top-k Accuracy, Accuracy)...')
    vectorstore = pipeline.vectorstore

    mrr = compute_mrr(questions, vectorstore, gt_chunks, k=10)
    topk_acc = compute_topk_accuracy(questions, vectorstore, gt_chunks, k=5)
    accuracy = compute_accuracy(answers, gt_chunks)

    final_scores = {
        'mrr': mrr,
        'topk_accuracy': topk_acc,
        'accuracy': accuracy,
        'context_recall': recall,
        'faithfulness': faith,
        'context_precision': precision,
    }

    print('\n========== TELERAG EVALUATION RESULTS ==========')
    for metric, value in final_scores.items():
        target = KPI_TARGETS[metric]
        status = '✓ PASS' if value >= target else '✗ FAIL'
        print(f'  {metric:<22} {value:.3f}   (target > {target})  {status}')
    print('=================================================')


    scores['mrr_score'] = mrr
    scores['topk_accuracy'] = topk_acc
    scores['accuracy'] = accuracy

    scores.to_csv('evaluation_results.csv', index=False)

    summary_df = pd.DataFrame([final_scores])
    summary_df.to_csv('evaluation_summary.csv', index=False)

    print('Saved to evaluation_results.csv and evaluation_summary.csv')

if __name__ == '__main__':
    run_eval()