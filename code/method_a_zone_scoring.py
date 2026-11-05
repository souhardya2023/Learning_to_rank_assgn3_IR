"""
method_a_zone_scoring.py - Method A: Boolean Weighted Zone Scoring with Grid Search.

For every (query, document) pair and every zone z in {title, author, bib, body}
the *Boolean* match of each query term against the zone is computed
(term in zone -> 1, else 0).  The zone input is the mean of these Boolean
matches over the distinct query terms:

    s_z(q, d) = (1/|q|) * sum_{t in q} [t in d_z]          (mode='term')

The ranking score is the weighted zone score  S(q,d) = sum_z w_z * s_z(q,d)
with w_z >= 0, sum w_z = 1.  The weights are learned by exhaustive simplex
grid search maximising MAP on the training queries.

mode='any' reproduces the coarse variant  s_z = [q ∩ d_z != {}]  and is only kept
as an ablation.

Training and inference use the very same scoring/ordering function (stable sort,
ties broken by ascending doc id), so the objective that is optimised is the
ranking that is deployed.
"""

import numpy as np

FIELDS = ('title', 'author', 'bib', 'body')


class BooleanWeightedZoneScorer:
    def __init__(self, corpus, fields=FIELDS, mode='term'):
        self.corpus = corpus
        self.fields = list(fields)
        self.num_fields = len(self.fields)
        self.mode = mode
        self.weights = np.ones(self.num_fields, dtype=np.float32) / self.num_fields
        self.all_dids = sorted(corpus.docs.keys())
        self.doc_field_sets = {
            did: {f: set(fd[f]) for f in self.fields}
            for did, fd in corpus.doc_tokens.items()
        }
        self._cache = {}

    def get_query_zone_matrix(self, qid, doc_ids=None):
        """(N, 4) matrix of zone inputs s_z(q, d) for all (or the given) docs."""
        if qid not in self._cache:
            q_set = set(self.corpus.query_tokens[qid])
            n_q = max(len(q_set), 1)
            mat = np.zeros((len(self.all_dids), self.num_fields), dtype=np.float32)
            for i, did in enumerate(self.all_dids):
                sets = self.doc_field_sets[did]
                for j, f in enumerate(self.fields):
                    m = len(q_set & sets[f])
                    if self.mode == 'any':
                        mat[i, j] = 1.0 if m > 0 else 0.0
                    else:
                        mat[i, j] = m / n_q
            self._cache[qid] = mat
        mat = self._cache[qid]
        if doc_ids is None:
            return mat, self.all_dids
        idx = [self.all_dids.index(d) for d in doc_ids]
        return mat[idx], list(doc_ids)

    def score_docs(self, qid, weights=None, doc_ids=None):
        """Ranked list of (doc_id, score), score descending (stable, doc-id ties)."""
        w = np.asarray(self.weights if weights is None else weights, dtype=np.float32)
        mat, d_ids = self.get_query_zone_matrix(qid, doc_ids)
        scores = mat @ w
        order = np.argsort(-scores, kind='stable')
        return [(d_ids[i], float(scores[i])) for i in order]

    def grid_search_weights(self, train_qids, step=0.05):
        """Exhaustive simplex grid search maximising mean AP over train_qids."""
        steps = int(round(1.0 / step))
        cands = [(a, b, c, steps - a - b - c)
                 for a in range(steps + 1)
                 for b in range(steps + 1 - a)
                 for c in range(steps + 1 - a - b)]
        W = np.array(cands, dtype=np.float32) / steps          # (C, 4)
        n_docs = len(self.all_dids)
        denom = np.arange(1, n_docs + 1, dtype=np.float32)[:, None]
        ap_sum = np.zeros(len(cands), dtype=np.float64)
        used = 0
        for qid in train_qids:
            rel = self.corpus.get_relevant_docs(qid)
            if not rel:
                continue
            y = np.array([d in rel for d in self.all_dids], dtype=np.float32)
            mat, _ = self.get_query_zone_matrix(qid)
            preds = mat @ W.T                                   # (N, C)
            order = np.argsort(-preds, axis=0, kind='stable')
            hits = y[order]
            ap_sum += ((np.cumsum(hits, axis=0) / denom) * hits).sum(axis=0) / y.sum()
            used += 1
        avg = ap_sum / max(used, 1)
        best = int(np.argmax(avg))
        self.weights = W[best].copy()
        return {
            'best_weights_ranking': tuple(float(x) for x in W[best]),
            'best_metric_score': float(avg[best]),
            'history': [(tuple(float(x) for x in W[i]), float(avg[i])) for i in range(len(cands))],
        }
