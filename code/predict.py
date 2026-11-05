"""
predict.py - Standalone Prediction Script for Evaluating on Unknown Test Queries.

Usage:
  python predict.py --queries <path_to_queries_file> --output <output_file> --method <svm|zone|bm25>

Arguments:
  --queries : Path to queries file (either TSV 'qid\tquery_text' or Cranfield .qry format)
  --output  : Path to output text file (default: predictions.txt)
  --method  : svm (Method B) | zone (Method A) | bm25
  --top_k   : Number of top documents to retrieve per query (default: 100)
"""

import os
import sys
import argparse
import json
import numpy as np
from dataset_parser import CranfieldCorpus, tokenize
from feature_extractor import FeatureExtractor
from method_a_zone_scoring import BooleanWeightedZoneScorer
from method_b_svm import SVMLearningToRank
from evaluate import BM25Baseline

def load_external_queries(query_path):
    """
    Parses either tab-separated 'qid\tquery' or Cranfield .qry style.
    """
    queries = {}
    with open(query_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
    if '.I ' in content:
        # Cranfield format
        entries = content.split('.I ')
        for idx, entry in enumerate(entries[1:], start=1):
            lines = entry.split('\n')
            qtext = ' '.join(l for l in lines[1:] if not l.startswith('.W')).strip()
            queries[idx] = qtext
    else:
        # TSV or line-by-line format
        for line_no, line in enumerate(content.split('\n'), start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                try:
                    qid = int(parts[0])
                    qtext = parts[1].strip()
                except ValueError:
                    qid = line_no
                    qtext = line
            else:
                qid = line_no
                qtext = line
            queries[qid] = qtext
    return queries

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    default_results_dir = os.path.join(root_dir, 'results')
    default_weights = os.path.join(default_results_dir, 'experiment_results.json')
    if not os.path.exists(default_weights):
        default_weights = 'experiment_results.json'

    parser = argparse.ArgumentParser(description="Evaluate on unknown test queries")
    parser.add_argument('--queries', type=str, default=os.path.join(root_dir, 'data', 'TEST', 'query.txt'), help='Path to queries file')
    parser.add_argument('--output', type=str, default=os.path.join(default_results_dir, 'predictions.txt'), help='Path to output file')
    parser.add_argument('--method', type=str, choices=['svm', 'zone', 'bm25'], default='svm', help='Method to use')
    parser.add_argument('--top_k', type=int, default=100, help='Number of documents per query')
    parser.add_argument('--cutoff', type=int, default=None, help='K for the set-based list (default: F1-optimal K from training)')
    parser.add_argument('--weights_file', type=str, default=default_weights, help='Model weights file')
    args = parser.parse_args()

    # Resolve query path if relative
    if not os.path.exists(args.queries):
        alt_q = os.path.join(root_dir, args.queries)
        if os.path.exists(alt_q):
            args.queries = alt_q

    print(f"Loading corpus and models for method: {args.method}...")
    corpus = CranfieldCorpus()
    extractor = FeatureExtractor(corpus)

    # Load tuned hyper-parameters of the final models (trained on all 225 judged queries)
    final = {}
    weights_path = args.weights_file
    if not os.path.exists(weights_path):
        weights_path = os.path.join(default_results_dir, 'experiment_results.json')
    if os.path.exists(weights_path):
        with open(weights_path, 'r') as f:
            final = json.load(f).get('final', {})
    weights_a = final.get('method_a_weights', [0.35, 0.25, 0.15, 0.25])
    cutoffs = final.get('cutoffs', {})

    zone_scorer = BooleanWeightedZoneScorer(corpus)
    if args.method == 'zone':
        rank_fn = lambda qid: zone_scorer.score_docs(qid, weights=weights_a)
        cut_key = "Method A (Weighted Zone)"
    elif args.method == 'bm25':
        rank_fn = BM25Baseline(corpus, extractor).rank
        cut_key = "Okapi BM25"
    else:
        svm_ranker = SVMLearningToRank(corpus, extractor,
                                       C=final.get('method_b_C', 0.005),
                                       mode=final.get('method_b_mode', 'point'),
                                       neg_ratio=final.get('method_b_neg_ratio', 3))
        svm_ranker.train(sorted(corpus.queries.keys()))
        rank_fn = svm_ranker.score_docs
        cut_key = "Method B (Linear SVM L2R)"
    cutoff_k = args.cutoff or cutoffs.get(cut_key, 10)

    # Training is finished: drop every per-query cache (external query ids may collide
    # with the ids of the judged queries) before the new queries are registered.
    extractor.query_matrix_cache.clear()
    zone_scorer._cache.clear()

    # Load target queries
    external_queries = load_external_queries(args.queries)
    print(f"Loaded {len(external_queries)} queries from {args.queries}")

    # Temporary overwrite corpus queries if evaluating external queries
    for qid, qtext in external_queries.items():
        corpus.queries[qid] = qtext
        corpus.query_tokens[qid] = tokenize(qtext)

    # Generate predictions: full ranking (top_k) + set-based list (top-K* by F1)
    cut_path = os.path.splitext(args.output)[0] + '_cutoff' + os.path.splitext(args.output)[1]
    print(f"Writing top {args.top_k} rankings to {args.output} and top {cutoff_k} to {cut_path}...")
    with open(args.output, 'w') as out, open(cut_path, 'w') as out_c:
        for qid in sorted(external_queries.keys()):
            ranked = rank_fn(qid)[:args.top_k]
            for rank, (did, score) in enumerate(ranked, start=1):
                line = f"{qid}\t{did}\t{rank}\t{score:.6f}\n"
                out.write(line)
                if rank <= cutoff_k:
                    out_c.write(line)

    print(f"Inference completed successfully! Results written to {args.output}")

if __name__ == '__main__':
    main()
