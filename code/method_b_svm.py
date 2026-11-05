"""
method_b_svm.py - Method B: Learning Real-Valued Weights using Linear Support Vector Machines.

Features:
- Constructs training instances (X, y) from (query, document) pairs using the 22 features.
- Balances class distribution with balanced weighting and informative negative sampling.
- Robust feature scaling (RobustScaler) using median and IQR to resist heavy-tailed term distributions.
- Trains Linear SVM (LinearSVC) with fast primal optimization (dual=False).
- Ranks candidate documents using the real-valued decision score: S(q, d) = w^T x + b.
- Inspects and exports learned feature weights.
"""

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import RobustScaler
from feature_extractor import FEATURE_NAMES

class SVMLearningToRank:
    def __init__(self, corpus, extractor, C=0.005, random_state=42):
        self.corpus = corpus
        self.extractor = extractor
        self.C = C
        self.random_state = random_state
        self.scaler = RobustScaler(quantile_range=(25.0, 75.0))
        self.model = LinearSVC(
            C=self.C,
            loss='squared_hinge',
            dual=False,
            class_weight='balanced',
            max_iter=3000,
            tol=1e-3,
            random_state=self.random_state
        )
        self.is_trained = False
        self.feature_weights = None
        self.intercept = 0.0

    def build_dataset(self, qids, neg_ratio=3, only_matching=True):
        """
        Builds (X, y, query_doc_meta) for the given query IDs.
        For each query:
        - Includes all relevant documents as positive examples (y=1).
        - Samples negative examples from candidate non-relevant documents.
        """
        all_dids = sorted(self.corpus.docs.keys())
        X_list = []
        y_list = []
        meta_list = []

        for qid in qids:
            rel_docs = self.corpus.get_relevant_docs(qid)
            if not rel_docs:
                continue
            
            q_tokens = set(self.corpus.query_tokens[qid])
            pos_count = 0
            
            # Positives
            for did in rel_docs:
                feat = self.extractor.extract_features(qid, did)
                X_list.append(feat)
                y_list.append(1)
                meta_list.append((qid, did))
                pos_count += 1
                
            # Negatives: candidate non-relevant docs
            non_rel_candidates = []
            for did in all_dids:
                if did in rel_docs:
                    continue
                if only_matching:
                    d_tokens = set(self.corpus.doc_tokens[did]['all'])
                    if bool(q_tokens & d_tokens):
                        non_rel_candidates.append(did)
                else:
                    non_rel_candidates.append(did)
                    
            target_neg = min(len(non_rel_candidates), pos_count * neg_ratio)
            rng = np.random.RandomState(self.random_state + qid)
            if len(non_rel_candidates) > target_neg:
                selected_negs = rng.choice(non_rel_candidates, size=target_neg, replace=False)
            else:
                selected_negs = non_rel_candidates
                
            for did in selected_negs:
                feat = self.extractor.extract_features(qid, did)
                X_list.append(feat)
                y_list.append(0)
                meta_list.append((qid, did))
                
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)
        return X, y, meta_list

    def train(self, train_qids, neg_ratio=3, tune_C=False, val_qids=None):
        """
        Trains the Linear SVM on the training queries.
        If tune_C is True and val_qids is provided, performs fast validation search over C.
        """
        X_train, y_train, _ = self.build_dataset(train_qids, neg_ratio=neg_ratio)
        
        # Fit scaler
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if tune_C and val_qids:
            # Pre-scale validation matrices
            all_dids = sorted(self.corpus.docs.keys())
            val_matrices = {}
            val_rels = {}
            for qid in val_qids:
                rels = self.corpus.get_relevant_docs(qid)
                if rels:
                    X_raw, _ = self.extractor.extract_features_for_query(qid, doc_ids=all_dids)
                    val_matrices[qid] = self.scaler.transform(X_raw)
                    val_rels[qid] = rels
                    
            best_c = self.C
            best_map = -1.0
            candidate_c = [0.001, 0.005, 0.01, 0.05, 0.1]
            
            for c_val in candidate_c:
                temp_model = LinearSVC(
                    C=c_val,
                    loss='squared_hinge',
                    dual=False,
                    class_weight='balanced',
                    max_iter=3000,
                    tol=1e-3,
                    random_state=self.random_state
                )
                temp_model.fit(X_train_scaled, y_train)
                w = temp_model.coef_[0]
                b = float(temp_model.intercept_[0])
                
                maps = []
                for qid, X_val in val_matrices.items():
                    rel_docs = val_rels[qid]
                    scores = X_val @ w + b
                    ranks = np.argsort(-scores)
                    hits = 0
                    ap = 0.0
                    for rank, idx in enumerate(ranks, start=1):
                        did = all_dids[idx]
                        if did in rel_docs:
                            hits += 1
                            ap += hits / rank
                    maps.append(ap / len(rel_docs))
                avg_map = float(np.mean(maps)) if maps else 0.0
                if avg_map > best_map:
                    best_map = avg_map
                    best_c = c_val
                    
            self.C = best_c
            
        self.model = LinearSVC(
            C=self.C,
            loss='squared_hinge',
            dual=False,
            class_weight='balanced',
            max_iter=3000,
            tol=1e-3,
            random_state=self.random_state
        )
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        self.feature_weights = self.model.coef_[0]
        self.intercept = float(self.model.intercept_[0])
        
        return {
            'C': self.C,
            'weights': self.feature_weights,
            'intercept': self.intercept,
            'train_samples': len(y_train),
            'positives': int(np.sum(y_train == 1)),
            'negatives': int(np.sum(y_train == 0))
        }

    def score_docs(self, qid, doc_ids=None):
        """
        Computes SVM decision function score: S(q, d) = w^T * scaled_x + b
        for all candidate docs, returning list of (doc_id, score) sorted descending.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet!")
            
        if doc_ids is None:
            doc_ids = sorted(self.corpus.docs.keys())
            
        X, d_ids = self.extractor.extract_features_for_query(qid, doc_ids=doc_ids)
        X_scaled = self.scaler.transform(X)
        
        scores = self.model.decision_function(X_scaled)
        
        # Rank by decision score descending
        ranked = list(zip(d_ids, [float(s) for s in scores]))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def get_feature_importances(self):
        """
        Returns dictionary of feature names mapped to learned weights.
        """
        if not self.is_trained:
            return {}
        return {name: float(w) for name, w in zip(FEATURE_NAMES, self.feature_weights)}
