"""
predict.py - Standalone Prediction Script for Evaluating on Unknown Test Queries.

Usage:
  python predict.py --queries <path_to_queries_file> --output <output_file> --method <svm|zone|bm25>

Arguments:
  --queries : Path to queries file (either TSV 'qid\tquery_text' or Cranfield .qry format)
  --output  : Path to output text file (default: predictions.txt)
  --method  : Ranking method: 'svm' (Method B), 'zone' (Method A), or 'bm25' (default: svm)
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
    parser = argparse.ArgumentParser(description="Evaluate on unknown test queries")
    parser.add_argument('--queries', type=str, default='data/TEST/query.txt', help='Path to queries file')
    parser.add_argument('--output', type=str, default='predictions.txt', help='Path to output file')
    parser.add_argument('--method', type=str, choices=['svm', 'zone', 'bm25'], default='svm', help='Method to use')
    parser.add_argument('--top_k', type=int, default=100, help='Number of documents per query')
    parser.add_argument('--weights_file', type=str, default='experiment_results.json', help='Model weights file')
    args = parser.parse_args()

    print(f"Loading corpus and models for method: {args.method}...")
    corpus = CranfieldCorpus()
    extractor = FeatureExtractor(corpus)

    # Load trained parameters if available
    weights_a = [0.15, 0.0, 0.05, 0.80]
    best_c = 0.1
    if os.path.exists(args.weights_file):
        with open(args.weights_file, 'r') as f:
            meta = json.load(f)
            weights_a = meta.get('method_a_weights', weights_a)
            best_c = meta.get('method_b_C', best_c)

    # Setup ranker
    if args.method == 'zone':
        zone_scorer = BooleanWeightedZoneScorer(corpus)
        rank_fn = lambda qid: zone_scorer.score_docs(qid, weights=weights_a)
    elif args.method == 'bm25':
        bm25_model = BM25Baseline(corpus, extractor)
        rank_fn = bm25_model.rank
    else:
        svm_ranker = SVMLearningToRank(corpus, extractor, C=best_c)
        all_train_qids = sorted(corpus.queries.keys())
        svm_ranker.train(all_train_qids, neg_ratio=8, tune_C=False)
        rank_fn = svm_ranker.score_docs

    # Load target queries
    external_queries = load_external_queries(args.queries)
    print(f"Loaded {len(external_queries)} queries from {args.queries}")

    # Temporary overwrite corpus queries if evaluating external queries
    for qid, qtext in external_queries.items():
        corpus.queries[qid] = qtext
        corpus.query_tokens[qid] = tokenize(qtext)

    # Generate predictions
    print(f"Writing top {args.top_k} rankings to {args.output}...")
    with open(args.output, 'w') as out:
        for qid in sorted(external_queries.keys()):
            ranked = rank_fn(qid)[:args.top_k]
            for rank, (did, score) in enumerate(ranked, start=1):
                out.write(f"{qid}\t{did}\t{rank}\t{score:.6f}\n")

    print(f"Inference completed successfully! Results written to {args.output}")

if __name__ == '__main__':
    main()
