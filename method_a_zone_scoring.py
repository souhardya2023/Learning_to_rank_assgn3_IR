"""
method_a_zone_scoring.py - Method A: Boolean Weighted Zone Scoring with Grid Search Optimization.

Features:
- Extracts Boolean matching scores for Title, Author, Bibliography, and Text Body:
  s_field(q, d) in {0, 1}.
- Weighted score: S(q, d) = w_title * s_title + w_author * s_author + w_bib * s_bib + w_body * s_body
  with sum(w_i) = 1, w_i >= 0.
- Fast vectorized Grid search optimization over weight combinations to maximize ranking performance (MAP/F1)
  and minimize Mean Squared Error (MSE) against relevance judgments.
- Document ranking and evaluation interface.
"""

import numpy as np
from collections import defaultdict

class BooleanWeightedZoneScorer:
    def __init__(self, corpus, fields=('title', 'author', 'bib', 'body')):
        self.corpus = corpus
        self.fields = list(fields)
        self.num_fields = len(self.fields)
        self.weights = np.ones(self.num_fields, dtype=np.float32) / self.num_fields
        
        # Precompute zone sets for all documents
        self.doc_field_sets = defaultdict(dict)
        for did, fdict in corpus.doc_tokens.items():
            for f in self.fields:
                self.doc_field_sets[did][f] = set(fdict[f])
                
        # Cache boolean match vectors: (qid, did) -> np.array([s_1, s_2, s_3, s_4])
        self.zone_cache = {}

    def compute_zone_vector(self, query_tokens, doc_id):
        """
        Computes binary vector [s_title, s_author, s_bib, s_body] in {0, 1}^4.
        s_field = 1 if any query term is in field, else 0.
        """
        q_set = set(query_tokens)
        vec = np.zeros(self.num_fields, dtype=np.float32)
        d_sets = self.doc_field_sets[doc_id]
        for idx, f in enumerate(self.fields):
            if bool(q_set & d_sets[f]):
                vec[idx] = 1.0
        return vec

    def get_query_zone_matrix(self, qid, doc_ids=None):
        """
        Returns an (M, 4) matrix of boolean matching features for query qid across specified doc_ids (default all 1400).
        """
        if doc_ids is None:
            doc_ids = sorted(self.corpus.docs.keys())
        q_toks = self.corpus.query_tokens[qid]
        
        mat = np.zeros((len(doc_ids), self.num_fields), dtype=np.float32)
        for i, did in enumerate(doc_ids):
            cache_key = (qid, did)
            if cache_key not in self.zone_cache:
                self.zone_cache[cache_key] = self.compute_zone_vector(q_toks, did)
            mat[i] = self.zone_cache[cache_key]
        return mat, doc_ids

    def score_docs(self, qid, weights=None, doc_ids=None):
        """
        Computes S(q, d) = sum(w_i * s_i) for all documents.
        Returns list of (doc_id, score) sorted by score descending.
        """
        if weights is None:
            weights = self.weights
        weights = np.array(weights, dtype=np.float32)
        
        mat, d_ids = self.get_query_zone_matrix(qid, doc_ids=doc_ids)
        scores = np.dot(mat, weights)
        
        # Tie-breaker using term overlap with full text
        q_set = set(self.corpus.query_tokens[qid])
        ranked = []
        for i, did in enumerate(d_ids):
            s = float(scores[i])
            if s > 0:
                overlap = len(q_set & set(self.corpus.doc_tokens[did]['all']))
                tie_break = s + 1e-4 * overlap
            else:
                tie_break = 0.0
            ranked.append((did, s, tie_break))
            
        ranked.sort(key=lambda x: (x[2], x[1]), reverse=True)
        return [(did, score) for did, score, _ in ranked]

    def grid_search_weights(self, train_qids, step=0.05, metric='MAP'):
        """
        Vectorized simplex search over:
        w_1 + w_2 + w_3 + w_4 = 1.0, w_i >= 0.
        Optimizes for MAP/F1 and tracks MSE.
        """
        steps = int(round(1.0 / step))
        candidates = []
        for w1 in range(steps + 1):
            for w2 in range(steps + 1 - w1):
                for w3 in range(steps + 1 - w1 - w2):
                    w4 = steps - w1 - w2 - w3
                    candidates.append((w1 * step, w2 * step, w3 * step, w4 * step))
                    
        W_cand = np.array(candidates, dtype=np.float32)  # (C, 4)
        
        # Precompute zone matrices and ground truths for train_qids
        train_data = []
        all_dids = sorted(self.corpus.docs.keys())
        N_docs = len(all_dids)
        rank_denoms = np.arange(1, N_docs + 1, dtype=np.float32)

        # Sufficient statistics for quadratic MSE:
        # sum ||mat @ w - y||^2 = w^T (sum mat^T mat) w - 2 w^T (sum mat^T y) + sum ||y||^2
        XtX_sum = np.zeros((4, 4), dtype=np.float64)
        Xty_sum = np.zeros(4, dtype=np.float64)
        yy_sum = 0.0
        total_pairs = len(train_qids) * N_docs

        for qid in train_qids:
            mat, _ = self.get_query_zone_matrix(qid, doc_ids=all_dids)
            rel_set = self.corpus.get_relevant_docs(qid)
            rel_binary = np.array([1.0 if did in rel_set else 0.0 for did in all_dids], dtype=np.float32)
            num_rel = float(len(rel_set))
            
            if num_rel > 0:
                train_data.append((mat, rel_binary, num_rel))
                
            XtX_sum += mat.T @ mat
            Xty_sum += mat.T @ rel_binary
            yy_sum += float(np.sum(rel_binary ** 2))

        # Vectorized MSE for all candidate weights
        # MSE = (diag(W @ XtX_sum @ W.T) - 2 * (W @ Xty_sum) + yy_sum) / total_pairs
        mse_terms = np.sum((W_cand @ XtX_sum) * W_cand, axis=1) - 2.0 * (W_cand @ Xty_sum) + yy_sum
        mses = mse_terms / float(total_pairs)
        best_mse_idx = int(np.argmin(mses))
        best_mse = float(mses[best_mse_idx])
        best_weights_mse = tuple(candidates[best_mse_idx])

        # Vectorized MAP computation over candidate weights
        num_train = len(train_data)
        metric_sums = np.zeros(len(candidates), dtype=np.float32)
        
        for mat, rel_binary, num_rel in train_data:
            # preds: (N_docs, C) = (1400, 4) @ (4, C)
            all_preds = mat @ W_cand.T
            
            # Rank for each candidate
            for c_idx in range(len(candidates)):
                preds = all_preds[:, c_idx]
                ranks = np.argsort(-preds)
                hits = rel_binary[ranks]
                cum_hits = np.cumsum(hits)
                ap = np.sum((cum_hits / rank_denoms) * hits) / num_rel
                metric_sums[c_idx] += ap

        avg_metrics = metric_sums / float(num_train)
        best_metric_idx = int(np.argmax(avg_metrics))
        best_metric_score = float(avg_metrics[best_metric_idx])
        best_weights_ranking = tuple(candidates[best_metric_idx])

        self.weights = np.array(best_weights_ranking, dtype=np.float32)
        
        history = [(candidates[i], float(avg_metrics[i]), float(mses[i])) for i in range(len(candidates))]

        return {
            'best_weights_ranking': best_weights_ranking,
            'best_metric_score': best_metric_score,
            'best_weights_mse': best_weights_mse,
            'best_mse_score': best_mse,
            'history': history
        }
