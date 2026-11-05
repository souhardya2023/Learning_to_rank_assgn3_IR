"""
feature_extractor.py - Feature Extraction for Method B (Linear SVM Learning to Rank).

Computes the exact 22 features specified in the assignment PDF:
1.  query term number
2.  query term ratio
3.  stream length
4.  idf
5.  sum of term frequency
6.  min of term frequency
7.  max of term frequency
8.  mean of term frequency
9.  variance of term frequency
10. sum of stream length normalized term frequency
11. min of stream length normalized term frequency
12. max of stream length normalized term frequency
13. mean of stream length normalized term frequency
14. variance of stream length normalized term frequency
15. sum of tf*idf
16. min of tf*idf
17. max of tf*idf
18. mean of tf*idf
19. variance of tf*idf
20. boolean model
21. vector space model
22. BM25
"""

import math
import numpy as np
from collections import Counter

FEATURE_NAMES = [
    "query_term_number",
    "query_term_ratio",
    "stream_length",
    "idf",
    "sum_of_term_frequency",
    "min_of_term_frequency",
    "max_of_term_frequency",
    "mean_of_term_frequency",
    "variance_of_term_frequency",
    "sum_of_stream_length_normalized_term_frequency",
    "min_of_stream_length_normalized_term_frequency",
    "max_of_stream_length_normalized_term_frequency",
    "mean_of_stream_length_normalized_term_frequency",
    "variance_of_stream_length_normalized_term_frequency",
    "sum_of_tf_idf",
    "min_of_tf_idf",
    "max_of_tf_idf",
    "mean_of_tf_idf",
    "variance_of_tf_idf",
    "boolean_model",
    "vector_space_model",
    "BM25"
]

class FeatureExtractor:
    def __init__(self, corpus, k1=1.2, b=0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        
        # Precompute document term frequency vectors & lengths
        self.doc_tfs = {}        # did -> Counter({term: tf})
        self.doc_norm_l2 = {}    # did -> L2 norm of doc tf-idf vector
        self._precompute_doc_stats()
        
        # Matrix cache: qid -> (1400, 22) numpy array
        self.query_matrix_cache = {}
        
    def _precompute_doc_stats(self):
        for did, fdict in self.corpus.doc_tokens.items():
            toks = fdict['all']
            tf_counts = Counter(toks)
            self.doc_tfs[did] = tf_counts
            
            # Precompute document L2 norm for VSM
            sq_sum = 0.0
            for term, count in tf_counts.items():
                if term in self.corpus.idf:
                    w = (1.0 + math.log(count)) * self.corpus.idf[term]
                    sq_sum += w * w
            self.doc_norm_l2[did] = math.sqrt(sq_sum) if sq_sum > 0 else 1.0

    def extract_features(self, qid, did):
        """
        Extracts 22 real-valued features for query qid and document did.
        """
        q_tokens = self.corpus.query_tokens[qid]
        num_q_terms = len(q_tokens)
        if num_q_terms == 0:
            return np.zeros(22, dtype=np.float32)
            
        doc_tf = self.doc_tfs.get(did, Counter())
        stream_len = float(self.corpus.doc_lengths.get(did, 0))
        denom_len = stream_len if stream_len > 0 else 1.0
        
        # Query TF
        q_tf = Counter(q_tokens)
        unique_q_terms = list(q_tf.keys())
        
        # Term frequencies for each term in query
        tfs = []
        matching_tfs = []
        idfs = []
        tf_idfs = []
        matching_tf_idfs = []
        
        bm25_score = 0.0
        vsm_dot = 0.0
        q_norm_sq = 0.0
        
        for term in unique_q_terms:
            tf = doc_tf.get(term, 0)
            tfs.append(tf)
            
            idf_val = self.corpus.idf.get(term, 1.0)
            idfs.append(idf_val)
            
            w_d = (1.0 + math.log(tf)) * idf_val if tf > 0 else 0.0
            w_q = (1.0 + math.log(q_tf[term])) * idf_val
            q_norm_sq += w_q * w_q
            
            if tf > 0:
                matching_tfs.append(tf)
                tf_idf = tf * idf_val
                tf_idfs.append(tf_idf)
                matching_tf_idfs.append(tf_idf)
                vsm_dot += w_d * w_q
                
                # BM25 term score
                bm25_idf = self.corpus.bm25_idf.get(term, 0.0)
                num = tf * (self.k1 + 1.0)
                den = tf + self.k1 * (1.0 - self.b + self.b * (stream_len / self.corpus.avg_doc_len))
                bm25_score += bm25_idf * (num / den)
            else:
                tf_idfs.append(0.0)

        # 1. query term number
        f1_query_term_number = float(num_q_terms)
        
        # 2. query term ratio
        matching_unique = len(matching_tfs)
        f2_query_term_ratio = float(matching_unique) / float(len(unique_q_terms))
        
        # 3. stream length
        f3_stream_length = stream_len
        
        # 4. idf (sum of idf for matching query terms)
        f4_idf = sum(self.corpus.idf.get(t, 0.0) for t in unique_q_terms if doc_tf.get(t, 0) > 0)
        
        # 5 - 9: Term frequency stats
        tf_arr = np.array(tfs, dtype=np.float32)
        f5_sum_tf = float(np.sum(tf_arr))
        f6_min_tf = float(np.min(matching_tfs)) if matching_tfs else 0.0
        f7_max_tf = float(np.max(tf_arr)) if len(tf_arr) > 0 else 0.0
        f8_mean_tf = float(np.mean(tf_arr)) if len(tf_arr) > 0 else 0.0
        f9_var_tf = float(np.var(tf_arr)) if len(tf_arr) > 0 else 0.0
        
        # 10 - 14: Stream length normalized term frequency stats
        norm_tf_arr = tf_arr / denom_len
        matching_norm_tfs = [tf / denom_len for tf in matching_tfs]
        f10_sum_norm_tf = float(np.sum(norm_tf_arr))
        f11_min_norm_tf = float(np.min(matching_norm_tfs)) if matching_norm_tfs else 0.0
        f12_max_norm_tf = float(np.max(norm_tf_arr)) if len(norm_tf_arr) > 0 else 0.0
        f13_mean_norm_tf = float(np.mean(norm_tf_arr)) if len(norm_tf_arr) > 0 else 0.0
        f14_var_norm_tf = float(np.var(norm_tf_arr)) if len(norm_tf_arr) > 0 else 0.0
        
        # 15 - 19: TF*IDF stats
        tfidf_arr = np.array(tf_idfs, dtype=np.float32)
        f15_sum_tfidf = float(np.sum(tfidf_arr))
        f16_min_tfidf = float(np.min(matching_tf_idfs)) if matching_tf_idfs else 0.0
        f17_max_tfidf = float(np.max(tfidf_arr)) if len(tfidf_arr) > 0 else 0.0
        f18_mean_tfidf = float(np.mean(tfidf_arr)) if len(tfidf_arr) > 0 else 0.0
        f19_var_tfidf = float(np.var(tfidf_arr)) if len(tfidf_arr) > 0 else 0.0
        
        # 20. boolean model (1.0 if matching terms > 0, else 0.0)
        f20_boolean_model = 1.0 if matching_unique > 0 else 0.0
        
        # 21. vector space model (cosine similarity)
        q_norm = math.sqrt(q_norm_sq) if q_norm_sq > 0 else 1.0
        d_norm = self.doc_norm_l2.get(did, 1.0)
        f21_vsm = float(vsm_dot / (q_norm * d_norm)) if (q_norm * d_norm) > 0 else 0.0
        
        # 22. BM25 score
        f22_bm25 = float(bm25_score)
        
        return np.array([
            f1_query_term_number,
            f2_query_term_ratio,
            f3_stream_length,
            f4_idf,
            f5_sum_tf,
            f6_min_tf,
            f7_max_tf,
            f8_mean_tf,
            f9_var_tf,
            f10_sum_norm_tf,
            f11_min_norm_tf,
            f12_max_norm_tf,
            f13_mean_norm_tf,
            f14_var_norm_tf,
            f15_sum_tfidf,
            f16_min_tfidf,
            f17_max_tfidf,
            f18_mean_tfidf,
            f19_var_tfidf,
            f20_boolean_model,
            f21_vsm,
            f22_bm25
        ], dtype=np.float32)

    def extract_features_for_query(self, qid, doc_ids=None):
        """
        Extracts feature vectors for candidate doc_ids for a single query.
        Uses cached full matrix when all docs requested.
        """
        all_dids = sorted(self.corpus.docs.keys())
        if doc_ids is None or doc_ids == all_dids:
            if qid not in self.query_matrix_cache:
                X = np.zeros((len(all_dids), 22), dtype=np.float32)
                for i, did in enumerate(all_dids):
                    X[i] = self.extract_features(qid, did)
                self.query_matrix_cache[qid] = X
            return self.query_matrix_cache[qid], all_dids
            
        X = np.zeros((len(doc_ids), 22), dtype=np.float32)
        for i, did in enumerate(doc_ids):
            X[i] = self.extract_features(qid, did)
        return X, doc_ids
