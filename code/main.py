"""
main.py - End-to-end pipeline for Cranfield Learning to Rank (IR Assignment 3).

1. Load corpus / judgments, 180/45 train/test query split (seed 42).
2. Method A: Boolean weighted zone scoring, simplex grid search (MAP) on train queries.
3. Method B: linear SVM on the 22 features; mode (pointwise / pairwise RankSVM), C and
   negative ratio chosen by inner 5-fold CV on the train queries only.
4. Baselines: unweighted zone, coarse 'any-term' zone (old variant), VSM, BM25.
5. Held-out evaluation (rank metrics + set-based P/R/F1 at an F1-optimal cutoff K
   selected on the training queries) and nested 5-fold CV over all 225 queries.
6. Final models trained on all 225 queries (used by predict.py) and result files.
"""

import os
import json
import numpy as np
from collections import defaultdict
from dataset_parser import CranfieldCorpus
from method_a_zone_scoring import BooleanWeightedZoneScorer
from feature_extractor import FeatureExtractor
from method_b_svm import SVMLearningToRank
from evaluate import (evaluate_system, evaluate_at_cutoff, choose_cutoff,
                      BM25Baseline, VSMBaseline, UnweightedZoneBaseline)

CUTOFFS = (5, 10, 20)


def run_experiments(seed=42):
    print("=" * 80)
    print("INFORMATION RETRIEVAL - ASSIGNMENT 3: LEARNING TO RANK")
    print("=" * 80)
    corpus = CranfieldCorpus()
    print(f"Docs={len(corpus.docs)} Queries={len(corpus.queries)} "
          f"Judged={len(corpus.qrels)} Vocab={len(corpus.df)} AvgLen={corpus.avg_doc_len:.1f}")

    all_qids = sorted(corpus.queries.keys())
    rng = np.random.RandomState(seed)
    shuf = list(all_qids)
    rng.shuffle(shuf)
    train_qids, test_qids = sorted(shuf[:180]), sorted(shuf[180:])
    print(f"Split: {len(train_qids)} train / {len(test_qids)} test")

    extractor = FeatureExtractor(corpus)
    zone = BooleanWeightedZoneScorer(corpus, mode='term')
    zone_any = BooleanWeightedZoneScorer(corpus, mode='any')
    unweighted = UnweightedZoneBaseline(zone)
    vsm, bm25 = VSMBaseline(corpus, extractor), BM25Baseline(corpus, extractor)

    # ---- Method A
    print("\n[Method A] simplex grid search (step 0.05, 1771 candidates)")
    ga = zone.grid_search_weights(train_qids, step=0.05)
    wA = ga['best_weights_ranking']
    print(f" term-level Boolean zones : w(T,A,B,W)={wA}  train MAP={ga['best_metric_score']:.4f}")
    ga_any = zone_any.grid_search_weights(train_qids, step=0.05)
    wA_any = ga_any['best_weights_ranking']
    print(f" coarse any-term (ablation): w(T,A,B,W)={wA_any}  train MAP={ga_any['best_metric_score']:.4f}")

    # ---- Method B
    print("\n[Method B] inner 5-fold CV over (mode, neg ratio, C) on train queries")
    svm = SVMLearningToRank(corpus, extractor, random_state=seed)
    tune = svm.tune(train_qids)
    print(f" best={tune['best']}  inner-CV MAP={tune['cv_map']:.4f}")
    info = svm.train(train_qids)
    print(f" trained: {info['train_samples']} samples ({info['positives']} pos / {info['negatives']} neg)")
    fi = svm.get_feature_importances()
    for n, w in sorted(fi.items(), key=lambda x: -abs(x[1]))[:8]:
        print(f"   {n:55s}{w:+.4f}")

    systems = {
        "Unweighted Zone": unweighted.rank,
        "Method A (Weighted Zone)": lambda q: zone.score_docs(q, weights=wA),
        "Method A (coarse any-term)": lambda q: zone_any.score_docs(q, weights=wA_any),
        "VSM (TF-IDF Cosine)": vsm.rank,
        "Okapi BM25": bm25.rank,
        "Method B (Linear SVM L2R)": svm.score_docs,
    }

    # ---- held-out evaluation
    print("\n[Held-out test, N=45]")
    results, cutoff_results, best_k = {}, {}, {}
    for name, fn in systems.items():
        results[name] = evaluate_system(fn, corpus, test_qids, cutoffs=CUTOFFS)
        best_k[name] = choose_cutoff(fn, corpus, train_qids)
        cutoff_results[name] = evaluate_at_cutoff(fn, corpus, test_qids, best_k[name])
    hdr = f"{'Model':<28}|" + "|".join(f"{h:^8}" for h in
          ["P@5", "P@10", "R@10", "F1@10", "MAP", "MRR", "NDCG@10"]) + "| K*  P/R/F1 @K*"
    print(hdr)
    for name, m in results.items():
        c = cutoff_results[name]
        print(f"{name:<28}|" + "|".join(f"{m[k]:^8.4f}" for k in
              ["P@5", "P@10", "R@10", "F1@10", "MAP", "MRR", "NDCG@10"])
              + f"| {c['K']:>2}  {c['P']:.3f}/{c['R']:.3f}/{c['F1']:.3f}")

    # ---- nested 5-fold CV
    print("\n[5-fold CV over all 225 queries - A grid and B hyper-parameters re-tuned inside each fold]")
    folds = [all_qids[i::5] for i in range(5)]
    cv = defaultdict(lambda: defaultdict(list))
    cv_cfg = []
    for fi_, val in enumerate(folds):
        tr = [q for q in all_qids if q not in val]
        zf = BooleanWeightedZoneScorer(corpus, mode='term')
        wf = zf.grid_search_weights(tr)['best_weights_ranking']
        zfa = BooleanWeightedZoneScorer(corpus, mode='any')
        wfa = zfa.grid_search_weights(tr)['best_weights_ranking']
        sf = SVMLearningToRank(corpus, extractor, random_state=seed + fi_)
        t = sf.tune(tr, Cs=(0.001, 0.005, 0.01, 0.05), neg_ratios=(3,), folds=3)
        sf.train(tr)
        cv_cfg.append({'fold': fi_, 'zone_w': wf, 'svm': list(t['best'])})
        print(f" fold {fi_}: A w={wf}  B cfg={t['best']}")
        fns = {
            "Unweighted Zone": UnweightedZoneBaseline(zf).rank,
            "Method A (Weighted Zone)": lambda q, zf=zf, wf=wf: zf.score_docs(q, weights=wf),
            "Method A (coarse any-term)": lambda q, z=zfa, w=wfa: z.score_docs(q, weights=w),
            "VSM (TF-IDF Cosine)": vsm.rank,
            "Okapi BM25": bm25.rank,
            "Method B (Linear SVM L2R)": sf.score_docs,
        }
        for name, fn in fns.items():
            for k, v in evaluate_system(fn, corpus, val, cutoffs=CUTOFFS).items():
                cv[name][k].append(v)
    cv_summary = {n: {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in d.items()}
                  for n, d in cv.items()}
    print(f"{'Model':<28}| P@10          | R@10          | F1@10         | MAP")
    for n, d in cv_summary.items():
        print(f"{n:<28}| " + " | ".join(f"{d[k]['mean']:.4f}±{d[k]['std']:.4f}" for k in ["P@10", "R@10", "F1@10", "MAP"]))

    # ---- result files (held-out test queries)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rdir = os.path.join(root, 'results')
    os.makedirs(rdir, exist_ok=True)

    def save(fn, fname, top_k=100):
        with open(os.path.join(rdir, fname), 'w') as f:
            for q in test_qids:
                for r, (d, s) in enumerate(fn(q)[:top_k], 1):
                    f.write(f"{q}\t{d}\t{r}\t{s:.6f}\n")
    save(systems["Method A (Weighted Zone)"], "results_method_a.txt")
    save(systems["Method B (Linear SVM L2R)"], "results_method_b.txt")
    save(systems["Okapi BM25"], "results_bm25_baseline.txt")

    # ---- final models on all 225 queries (consumed by predict.py)
    print("\n[Final models on all 225 queries]")
    zfin = BooleanWeightedZoneScorer(corpus, mode='term')
    wA_final = zfin.grid_search_weights(all_qids)['best_weights_ranking']
    sfin = SVMLearningToRank(corpus, extractor, random_state=seed)
    tfin = sfin.tune(all_qids)
    finfo = sfin.train(all_qids)
    fcut = {"Method A (Weighted Zone)": choose_cutoff(lambda q: zfin.score_docs(q, weights=wA_final), corpus, all_qids),
            "Method B (Linear SVM L2R)": choose_cutoff(sfin.score_docs, corpus, all_qids),
            "Okapi BM25": choose_cutoff(bm25.rank, corpus, all_qids)}
    print(f" A weights={wA_final}  B cfg={tfin['best']}  cutoffs={fcut}")

    out = {
        'seed': seed, 'train_qids': train_qids, 'test_qids': test_qids,
        'method_a_weights': [float(x) for x in wA],
        'method_a_any_weights': [float(x) for x in wA_any],
        'method_a_train_map': ga['best_metric_score'],
        'method_b_weights': fi, 'method_b_intercept': info['intercept'],
        'method_b_C': info['C'], 'method_b_mode': info['mode'], 'method_b_neg_ratio': info['neg_ratio'],
        'method_b_tuning': tune,
        'test_results': results, 'cutoff_results': cutoff_results, 'best_k': best_k,
        'cv_summary': cv_summary, 'cv_configs': cv_cfg,
        'final': {'method_a_weights': [float(x) for x in wA_final],
                  'method_b_mode': finfo['mode'], 'method_b_C': finfo['C'],
                  'method_b_neg_ratio': finfo['neg_ratio'],
                  'method_b_weights': sfin.get_feature_importances(),
                  'cutoffs': fcut},
    }
    with open(os.path.join(rdir, 'experiment_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(" saved results/experiment_results.json")
    return out


if __name__ == '__main__':
    run_experiments()
