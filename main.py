"""
main.py - End-to-End Pipeline for Cranfield Learning to Rank (IR Assignment 3).

Executes:
1. Corpus and ground truth loading.
2. Train/Test query splitting (80/20 split: 180 train, 45 test) and 5-fold cross-validation.
3. Method A: Boolean Weighted Zone Scoring optimization via Grid Search.
4. Method B: Linear SVM Learning to Rank with 22 features (Optimized with RobustScaler).
5. Baselines: Unweighted Zone Scoring, VSM (TF-IDF Cosine), and Okapi BM25.
6. Comprehensive evaluation: P@5, P@10, P@20, R@10, R@20, F1@10, F1@20, MAP, MRR, NDCG@10, NDCG@20.
7. Generation of submission result files for test queries.
8. Exporting results and metrics for reporting and analysis.
"""

import os
import json
import numpy as np
from collections import defaultdict
from dataset_parser import CranfieldCorpus
from method_a_zone_scoring import BooleanWeightedZoneScorer
from feature_extractor import FeatureExtractor, FEATURE_NAMES
from method_b_svm import SVMLearningToRank
from evaluate import (
    evaluate_system,
    BM25Baseline,
    VSMBaseline,
    UnweightedZoneBaseline
)

def run_experiments(seed=42):
    print("=" * 75)
    print("INFORMATION RETRIEVAL - ASSIGNMENT 3: LEARNING TO RANK (OPTIMIZED)")
    print("=" * 75)
    
    # 1. Load Corpus
    print("\n[Step 1] Loading Cranfield Corpus and Relevance Judgments...")
    corpus = CranfieldCorpus(cran_dir='cran', data_dir='data')
    print(f" -> Documents: {len(corpus.docs)}")
    print(f" -> Queries: {len(corpus.queries)}")
    print(f" -> Queries with relevance judgments: {len(corpus.qrels)}")
    print(f" -> Vocabulary size: {len(corpus.df)} terms")
    print(f" -> Average document length: {corpus.avg_doc_len:.2f} terms")

    # 2. Train/Test Split (80/20)
    all_qids = sorted(corpus.queries.keys())
    rng = np.random.RandomState(seed)
    shuffled_qids = list(all_qids)
    rng.shuffle(shuffled_qids)
    
    num_train = 180
    train_qids = sorted(shuffled_qids[:num_train])
    test_qids = sorted(shuffled_qids[num_train:])
    print(f"\n[Step 2] Split: {len(train_qids)} Train queries, {len(test_qids)} Test queries")

    # Initialize components
    extractor = FeatureExtractor(corpus)
    zone_scorer = BooleanWeightedZoneScorer(corpus)
    
    # Baselines
    unweighted_zone = UnweightedZoneBaseline(zone_scorer)
    vsm_baseline = VSMBaseline(corpus, extractor)
    bm25_baseline = BM25Baseline(corpus, extractor)

    # 3. Method A: Grid Search on Train Queries
    print("\n[Step 3] Method A: Boolean Weighted Zone Scoring - Grid Search...")
    grid_res = zone_scorer.grid_search_weights(train_qids, step=0.05, metric='MAP')
    best_weights = grid_res['best_weights_ranking']
    print(f" -> Best Learned Weights (Title, Author, Bib, Body): {best_weights}")
    print(f" -> Training MAP with best weights: {grid_res['best_metric_score']:.4f}")
    print(f" -> Best MSE Weights: {grid_res['best_weights_mse']} (MSE: {grid_res['best_mse_score']:.6f})")

    # 4. Method B: Linear SVM Learning to Rank (RobustScaler)
    print("\n[Step 4] Method B: Linear SVM Learning to Rank (RobustScaler)...")
    svm_ranker = SVMLearningToRank(corpus, extractor, C=0.005, random_state=seed)
    # Split train_qids into sub-train (140) and validation (40) for C tuning
    sub_train = train_qids[:140]
    val_qids = train_qids[140:]
    train_info = svm_ranker.train(sub_train, neg_ratio=3, tune_C=True, val_qids=val_qids)
    print(f" -> Tuned SVM C: {train_info['C']}")
    print(f" -> Training Samples: {train_info['train_samples']} ({train_info['positives']} pos, {train_info['negatives']} neg)")
    print(f" -> Decision function intercept: {train_info['intercept']:.4f}")
    
    # Retrain on full train_qids with chosen C
    svm_ranker.C = train_info['C']
    full_info = svm_ranker.train(train_qids, neg_ratio=3, tune_C=False)
    print(f" -> Retrained SVM on full {len(train_qids)} queries: {full_info['train_samples']} samples")
    
    # Learned feature weights
    feature_importances = svm_ranker.get_feature_importances()
    print("\nTop Positive Feature Weights:")
    sorted_pos = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:7]
    for name, w in sorted_pos:
        print(f"  {name:45s}: +{w:.4f}")

    # 5. Evaluation on Test Queries
    print("\n[Step 5] Evaluating All Models on Held-Out Test Queries (N=45)...")
    cutoffs = (5, 10, 20)
    
    systems = {
        "Unweighted Zone": unweighted_zone.rank,
        "Method A (Weighted Zone)": lambda q: zone_scorer.score_docs(q, weights=best_weights),
        "VSM (TF-IDF Cosine)": vsm_baseline.rank,
        "Okapi BM25": bm25_baseline.rank,
        "Method B (Linear SVM L2R)": svm_ranker.score_docs
    }
    
    results = {}
    for name, fn in systems.items():
        print(f"  Evaluating {name}...")
        metrics = evaluate_system(fn, corpus, test_qids, cutoffs=cutoffs)
        results[name] = metrics

    # Print Comparison Table
    print("\n" + "=" * 105)
    print(f"{'Model':<25} | {'P@5':<7} | {'P@10':<7} | {'P@20':<7} | {'R@10':<7} | {'R@20':<7} | {'F1@10':<7} | {'MAP':<7} | {'MRR':<7} | {'NDCG@10':<7}")
    print("-" * 105)
    for name, m in results.items():
        print(f"{name:<25} | {m['P@5']:<7.4f} | {m['P@10']:<7.4f} | {m['P@20']:<7.4f} | {m['R@10']:<7.4f} | {m['R@20']:<7.4f} | {m['F1@10']:<7.4f} | {m['MAP']:<7.4f} | {m['MRR']:<7.4f} | {m['NDCG@10']:<7.4f}")
    print("=" * 105)

    # 6. 5-Fold Cross Validation
    print("\n[Step 6] Running 5-Fold Cross Validation across all 225 queries...")
    kfold_folds = 5
    fold_size = len(all_qids) // kfold_folds
    cv_metrics = defaultdict(lambda: defaultdict(list))
    
    for fold in range(kfold_folds):
        f_val = all_qids[fold * fold_size : (fold + 1) * fold_size]
        f_tr = [q for q in all_qids if q not in f_val]
        
        # Method A on fold
        f_zone_res = zone_scorer.grid_search_weights(f_tr, step=0.1, metric='MAP')
        f_best_w = f_zone_res['best_weights_ranking']
        m_a_eval = evaluate_system(lambda q: zone_scorer.score_docs(q, weights=f_best_w), corpus, f_val, cutoffs=cutoffs)
        
        # Method B on fold
        f_svm = SVMLearningToRank(corpus, extractor, C=svm_ranker.C, random_state=seed + fold)
        f_svm.train(f_tr, neg_ratio=3, tune_C=False)
        m_b_eval = evaluate_system(f_svm.score_docs, corpus, f_val, cutoffs=cutoffs)
        
        # BM25 on fold
        bm25_eval = evaluate_system(bm25_baseline.rank, corpus, f_val, cutoffs=cutoffs)
        # VSM on fold
        vsm_eval = evaluate_system(vsm_baseline.rank, corpus, f_val, cutoffs=cutoffs)
        # Unweighted zone
        unw_eval = evaluate_system(unweighted_zone.rank, corpus, f_val, cutoffs=cutoffs)
        
        for k, v in unw_eval.items():
            cv_metrics["Unweighted Zone"][k].append(v)
        for k, v in m_a_eval.items():
            cv_metrics["Method A (Weighted Zone)"][k].append(v)
        for k, v in vsm_eval.items():
            cv_metrics["VSM (TF-IDF Cosine)"][k].append(v)
        for k, v in bm25_eval.items():
            cv_metrics["Okapi BM25"][k].append(v)
        for k, v in m_b_eval.items():
            cv_metrics["Method B (Linear SVM L2R)"][k].append(v)
            
    cv_summary = {}
    print("\n5-Fold Cross Validation Summary (Mean +/- Std):")
    print(f"{'Model':<25} | {'P@10 (Mean+/-Std)':<20} | {'R@10 (Mean+/-Std)':<20} | {'F1@10 (Mean+/-Std)':<20} | {'MAP (Mean+/-Std)':<20}")
    print("-" * 95)
    for model_name, met_dict in cv_metrics.items():
        cv_summary[model_name] = {}
        for met in met_dict.keys():
            mean_v = float(np.mean(met_dict[met]))
            std_v = float(np.std(met_dict[met]))
            cv_summary[model_name][met] = {'mean': mean_v, 'std': std_v}
        print(f"{model_name:<25} | {cv_summary[model_name]['P@10']['mean']:.4f} +/- {cv_summary[model_name]['P@10']['std']:.4f} | {cv_summary[model_name]['R@10']['mean']:.4f} +/- {cv_summary[model_name]['R@10']['std']:.4f} | {cv_summary[model_name]['F1@10']['mean']:.4f} +/- {cv_summary[model_name]['F1@10']['std']:.4f} | {cv_summary[model_name]['MAP']['mean']:.4f} +/- {cv_summary[model_name]['MAP']['std']:.4f}")

    # 7. Generate Submission Result Files for Test Queries
    print("\n[Step 7] Generating Result Files for Test Queries...")
    
    def save_results(ranker_fn, filename, top_k=100):
        with open(filename, 'w') as f:
            for qid in test_qids:
                ranked = ranker_fn(qid)[:top_k]
                for rank, (did, score) in enumerate(ranked, start=1):
                    # Format: <query_id> <doc_id> <rank> <score>
                    f.write(f"{qid}\t{did}\t{rank}\t{score:.6f}\n")
        print(f" -> Saved {filename} for {len(test_qids)} test queries.")

    save_results(lambda q: zone_scorer.score_docs(q, weights=best_weights), "results_method_a.txt")
    save_results(svm_ranker.score_docs, "results_method_b.txt")
    save_results(bm25_baseline.rank, "results_bm25_baseline.txt")

    # 8. Save Experiment Metadata
    exp_data = {
        'seed': seed,
        'train_qids': train_qids,
        'test_qids': test_qids,
        'method_a_weights': [float(x) for x in best_weights],
        'method_b_weights': feature_importances,
        'method_b_intercept': train_info['intercept'],
        'method_b_C': train_info['C'],
        'test_results': results,
        'cv_summary': cv_summary
    }
    with open('experiment_results.json', 'w') as f:
        json.dump(exp_data, f, indent=2)
    print(" -> Saved experiment_results.json successfully.")

    print("\nAll experiments completed successfully!")
    return exp_data

if __name__ == '__main__':
    run_experiments()
