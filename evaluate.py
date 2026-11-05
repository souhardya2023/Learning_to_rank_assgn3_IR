"""
evaluate.py - Evaluation Framework and Baseline Models for Cranfield Retrieval.

Computes:
- Precision@K (K=5, 10, 20)
- Recall@K (K=5, 10, 20)
- F1@K (K=5, 10, 20)
- Mean Average Precision (MAP)
- Mean Reciprocal Rank (MRR)
- NDCG@K (K=10, 20)

Includes Baselines:
- Unweighted Zone Scoring (equal weights across title, author, bib, body)
- Standard Vector Space Model (TF-IDF Cosine)
- Okapi BM25
- Method A: Boolean Weighted Zone Scoring
- Method B: Linear SVM Learning to Rank
"""

import math
import numpy as np
from collections import defaultdict

def precision_at_k(ranked_docs, relevant_docs, k):
    if k <= 0 or not ranked_docs:
        return 0.0
    top_k = [did for did, _ in ranked_docs[:k]]
    hits = sum(1 for did in top_k if did in relevant_docs)
    return hits / float(k)

def recall_at_k(ranked_docs, relevant_docs, k):
    if not relevant_docs:
        return 0.0
    top_k = [did for did, _ in ranked_docs[:k]]
    hits = sum(1 for did in top_k if did in relevant_docs)
    return hits / float(len(relevant_docs))

def f1_at_k(ranked_docs, relevant_docs, k):
    p = precision_at_k(ranked_docs, relevant_docs, k)
    r = recall_at_k(ranked_docs, relevant_docs, k)
    if p + r == 0:
        return 0.0
    return (2.0 * p * r) / (p + r)

def average_precision(ranked_docs, relevant_docs):
    if not relevant_docs:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for rank, (did, _) in enumerate(ranked_docs, start=1):
        if did in relevant_docs:
            hits += 1
            sum_prec += hits / float(rank)
    return sum_prec / float(len(relevant_docs))

def reciprocal_rank(ranked_docs, relevant_docs):
    if not relevant_docs:
        return 0.0
    for rank, (did, _) in enumerate(ranked_docs, start=1):
        if did in relevant_docs:
            return 1.0 / float(rank)
    return 0.0

def ndcg_at_k(ranked_docs, relevant_docs, k):
    if not relevant_docs or k <= 0:
        return 0.0
    dcg = 0.0
    for rank, (did, _) in enumerate(ranked_docs[:k], start=1):
        rel = 1.0 if did in relevant_docs else 0.0
        dcg += rel / math.log2(rank + 1)
        
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant_docs)) + 1))
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_system(ranker_fn, corpus, eval_qids, cutoffs=(5, 10, 20)):
    """
    Evaluates a ranking function over eval_qids.
    ranker_fn(qid) -> list of (doc_id, score) sorted descending.
    Returns dictionary of averaged metrics.
    """
    metrics = defaultdict(list)
    
    for qid in eval_qids:
        rel_docs = corpus.get_relevant_docs(qid)
        if not rel_docs:
            continue
            
        ranked = ranker_fn(qid)
        
        for k in cutoffs:
            metrics[f'P@{k}'].append(precision_at_k(ranked, rel_docs, k))
            metrics[f'R@{k}'].append(recall_at_k(ranked, rel_docs, k))
            metrics[f'F1@{k}'].append(f1_at_k(ranked, rel_docs, k))
            metrics[f'NDCG@{k}'].append(ndcg_at_k(ranked, rel_docs, k))
            
        metrics['MAP'].append(average_precision(ranked, rel_docs))
        metrics['MRR'].append(reciprocal_rank(ranked, rel_docs))
        
    summary = {}
    for metric_name, values in metrics.items():
        summary[metric_name] = float(np.mean(values))
    return summary

class BM25Baseline:
    def __init__(self, corpus, extractor):
        self.corpus = corpus
        self.extractor = extractor
        
    def rank(self, qid):
        all_dids = sorted(self.corpus.docs.keys())
        X, d_ids = self.extractor.extract_features_for_query(qid, doc_ids=all_dids)
        scores = X[:, 21]  # Index 21 is BM25
        order = np.argsort(-scores)
        return [(d_ids[i], float(scores[i])) for i in order]

class VSMBaseline:
    def __init__(self, corpus, extractor):
        self.corpus = corpus
        self.extractor = extractor
        
    def rank(self, qid):
        all_dids = sorted(self.corpus.docs.keys())
        X, d_ids = self.extractor.extract_features_for_query(qid, doc_ids=all_dids)
        scores = X[:, 20]  # Index 20 is VSM Cosine
        order = np.argsort(-scores)
        return [(d_ids[i], float(scores[i])) for i in order]

class UnweightedZoneBaseline:
    def __init__(self, zone_scorer):
        self.zone_scorer = zone_scorer
        self.unweighted = (0.25, 0.25, 0.25, 0.25)
        
    def rank(self, qid):
        return self.zone_scorer.score_docs(qid, weights=self.unweighted)
