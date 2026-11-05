"""
method_b_svm.py - Method B: Learning real-valued weights with a linear SVM.

The 22 features of a (query, document) pair are scaled with a RobustScaler
(median / IQR) and a linear SVM learns one real-valued weight per feature.
Ranking score:  S(q, d) = w . phi(q, d) (+ b).

Two training formulations (both linear SVMs, both learn w over the same 22 features):
  * mode='point' : pointwise  - relevant vs. non-relevant (sampled) documents,
                   class_weight='balanced'.
  * mode='pair'  : pairwise   - RankSVM: for each relevant doc d+ and sampled
                   non-relevant d-, the difference phi(d+) - phi(d-) is labelled +1
                   (and the mirrored difference -1); no bias term.
Mode, C and the negative-sampling ratio are chosen by inner k-fold cross-validation
on the *training* queries only (maximising MAP).
"""

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import RobustScaler
from feature_extractor import FEATURE_NAMES


def _ap(scores, y):
    order = np.argsort(-scores, kind='stable')
    h = y[order]
    return float((np.cumsum(h) / np.arange(1, len(h) + 1) * h).sum() / y.sum())


class SVMLearningToRank:
    def __init__(self, corpus, extractor, C=0.005, mode='point', neg_ratio=3, random_state=42):
        self.corpus = corpus
        self.extractor = extractor
        self.C = C
        self.mode = mode
        self.neg_ratio = neg_ratio
        self.random_state = random_state
        self.all_dids = sorted(corpus.docs.keys())
        self.scaler = RobustScaler(quantile_range=(25.0, 75.0))
        self.model = None
        self.is_trained = False
        self.feature_weights = None
        self.intercept = 0.0
        self._ycache = {}

    # ------------------------------------------------------------------ data
    def _xy(self, qid):
        X, _ = self.extractor.extract_features_for_query(qid)
        if qid not in self._ycache:
            rel = self.corpus.get_relevant_docs(qid)
            self._ycache[qid] = np.array([d in rel for d in self.all_dids], dtype=np.float32)
        return X, self._ycache[qid]

    def _fit_scaler(self, qids):
        rows = []
        for q in qids:
            X, y = self._xy(q)
            rows.append(X[(X[:, 1] > 0) | (y > 0)])
        self.scaler.fit(np.vstack(rows))

    def _build(self, qids, mode, neg_ratio):
        rng = np.random.RandomState(self.random_state)
        Xs, ys = [], []
        for q in qids:
            X, y = self._xy(q)
            if y.sum() == 0:
                continue
            Z = self.scaler.transform(X)
            pos = np.where(y > 0)[0]
            negs = np.where((y == 0) & (X[:, 1] > 0))[0]          # candidates sharing >=1 term
            if mode == 'point':
                k = min(len(negs), neg_ratio * len(pos))
                ch = rng.choice(negs, k, replace=False) if k else negs[:0]
                Xs.append(Z[np.r_[pos, ch]])
                ys.append(np.r_[np.ones(len(pos)), np.zeros(len(ch))])
            else:
                k = min(len(negs), neg_ratio)
                if k == 0:
                    continue
                ch = np.stack([rng.choice(negs, k, replace=False) for _ in pos])   # (P, k)
                D = (Z[pos][:, None, :] - Z[ch]).reshape(-1, Z.shape[1])
                Xs.append(np.vstack([D, -D]))
                ys.append(np.r_[np.ones(len(D)), np.zeros(len(D))])
        return np.vstack(Xs), np.concatenate(ys)

    def _make_model(self, C, mode):
        if mode == 'point':
            return LinearSVC(C=C, loss='squared_hinge', dual=False, class_weight='balanced',
                             max_iter=5000, tol=1e-4, random_state=self.random_state)
        return LinearSVC(C=C, loss='squared_hinge', dual=False, fit_intercept=False,
                         max_iter=5000, tol=1e-4, random_state=self.random_state)

    def _fit(self, qids, C, mode, neg_ratio):
        self._fit_scaler(qids)
        Xt, yt = self._build(qids, mode, neg_ratio)
        m = self._make_model(C, mode)
        m.fit(Xt, yt)
        return m, Xt, yt

    # -------------------------------------------------------------- training
    def tune(self, qids, modes=('point', 'pair'), Cs=(0.0005, 0.001, 0.005, 0.01, 0.05, 0.1),
             neg_ratios=(3, 10), folds=5):
        """Inner k-fold CV over (mode, C, neg_ratio) maximising MAP on `qids` only."""
        qids = list(qids)
        splits = [qids[i::folds] for i in range(folds)]
        results = {}
        for mode in modes:
            for nr in neg_ratios:
                for C in Cs:
                    aps = []
                    for f in splits:
                        tr = [q for q in qids if q not in f]
                        m, _, _ = self._fit(tr, C, mode, nr)
                        w = m.coef_[0]
                        b = float(m.intercept_[0]) if mode == 'point' else 0.0
                        for q in f:
                            X, y = self._xy(q)
                            if y.sum() == 0:
                                continue
                            aps.append(_ap(self.scaler.transform(X) @ w + b, y))
                    results[(mode, nr, C)] = float(np.mean(aps))
        best = max(results, key=results.get)
        self.mode, self.neg_ratio, self.C = best
        return {'best': best, 'cv_map': results[best],
                'grid': {f"{k[0]}|neg{k[1]}|C{k[2]}": v for k, v in results.items()}}

    def train(self, train_qids, C=None, mode=None, neg_ratio=None):
        if C is not None: self.C = C
        if mode is not None: self.mode = mode
        if neg_ratio is not None: self.neg_ratio = neg_ratio
        self.model, Xt, yt = self._fit(list(train_qids), self.C, self.mode, self.neg_ratio)
        self.feature_weights = self.model.coef_[0]
        self.intercept = float(self.model.intercept_[0]) if self.mode == 'point' else 0.0
        self.is_trained = True
        return {'C': self.C, 'mode': self.mode, 'neg_ratio': self.neg_ratio,
                'weights': self.feature_weights, 'intercept': self.intercept,
                'train_samples': len(yt), 'positives': int(np.sum(yt == 1)),
                'negatives': int(np.sum(yt == 0))}

    # ------------------------------------------------------------- inference
    def score_docs(self, qid, doc_ids=None):
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet!")
        X, d_ids = self.extractor.extract_features_for_query(qid, doc_ids=doc_ids)
        s = self.scaler.transform(X) @ self.feature_weights + self.intercept
        order = np.argsort(-s, kind='stable')
        return [(d_ids[i], float(s[i])) for i in order]

    def get_feature_importances(self):
        if not self.is_trained:
            return {}
        return {n: float(w) for n, w in zip(FEATURE_NAMES, self.feature_weights)}
