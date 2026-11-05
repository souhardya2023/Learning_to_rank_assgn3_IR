"""
generate_charts.py - Generates Publication-Quality Plots for the 10-Page IR Report.
"""

import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Set clean scientific plotting style
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

CHARTS_DIR = 'report_charts'
os.makedirs(CHARTS_DIR, exist_ok=True)

def generate_all_charts():
    with open('experiment_results.json', 'r') as f:
        data = json.load(f)
        
    test_results = data['test_results']
    cv_summary = data['cv_summary']
    method_a_w = data['method_a_weights']
    svm_weights = data['method_b_weights']
    
    # 1. Overall Model Comparison Chart
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=300)
    models = ["Unweighted Zone", "Method A (Zone)", "VSM (Cosine)", "Okapi BM25", "Method B (SVM)"]
    metrics = ['P@5', 'P@10', 'R@10', 'F1@10', 'MAP', 'NDCG@10']
    
    x = np.arange(len(metrics))
    width = 0.15
    colors = ['#7f8c8d', '#e67e22', '#2980b9', '#27ae60', '#8e44ad']
    
    model_keys = [
        "Unweighted Zone",
        "Method A (Weighted Zone)",
        "VSM (TF-IDF Cosine)",
        "Okapi BM25",
        "Method B (Linear SVM L2R)"
    ]
    
    for i, (m_label, m_key) in enumerate(zip(models, model_keys)):
        vals = [test_results[m_key][met] for met in metrics]
        ax.bar(x + (i - 2) * width, vals, width, label=m_label, color=colors[i], edgecolor='black', linewidth=0.5)
        
    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title('Test Set Performance Comparison Across Retrieval Models (N=45)', fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10.5, fontweight='bold')
    ax.legend(frameon=True, facecolor='#f8f9fa', edgecolor='#cccccc', fontsize=9, loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 0.65)
    plt.tight_layout()
    chart1_path = os.path.join(CHARTS_DIR, 'model_comparison.png')
    plt.savefig(chart1_path)
    plt.close()
    print(f"Saved {chart1_path}")

    # 2. Precision, Recall, and F1 at Cutoffs K=5, 10, 20
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 3.8), dpi=300)
    cutoffs = [5, 10, 20]
    
    for m_label, m_key, color in zip(models, model_keys, colors):
        p_vals = [test_results[m_key][f'P@{k}'] for k in cutoffs]
        r_vals = [test_results[m_key].get(f'R@{k}', test_results[m_key]['R@10'] if k==5 else test_results[m_key][f'R@{k}']) for k in cutoffs]
        f1_vals = [test_results[m_key][f'F1@{k}'] for k in cutoffs]
        
        ax1.plot(cutoffs, p_vals, marker='o', linewidth=2, label=m_label, color=color)
        ax2.plot(cutoffs, r_vals, marker='s', linewidth=2, label=m_label, color=color)
        ax3.plot(cutoffs, f1_vals, marker='^', linewidth=2, label=m_label, color=color)
        
    ax1.set_title('Precision @ K', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Cutoff K', fontsize=9.5)
    ax1.set_ylabel('Precision', fontsize=9.5)
    ax1.set_xticks(cutoffs)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2.set_title('Recall @ K', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Cutoff K', fontsize=9.5)
    ax2.set_ylabel('Recall', fontsize=9.5)
    ax2.set_xticks(cutoffs)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    ax3.set_title('F1 Score @ K', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Cutoff K', fontsize=9.5)
    ax3.set_ylabel('F1 Score', fontsize=9.5)
    ax3.set_xticks(cutoffs)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(frameon=True, fontsize=8, loc='upper right')
    
    plt.suptitle('Retrieval Metrics vs. Rank Cutoff (K = 5, 10, 20)', fontsize=12, fontweight='bold', y=1.03)
    plt.tight_layout()
    chart2_path = os.path.join(CHARTS_DIR, 'cutoffs_comparison.png')
    plt.savefig(chart2_path)
    plt.close()
    print(f"Saved {chart2_path}")

    # 3. SVM Learned Feature Weights (All 22 Features)
    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=300)
    sorted_features = sorted(svm_weights.items(), key=lambda x: x[1])
    f_names = [f[0].replace('_', ' ') for f in sorted_features]
    f_vals = [f[1] for f in sorted_features]
    bar_colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in f_vals]
    
    y_pos = np.arange(len(f_names))
    ax.barh(y_pos, f_vals, color=bar_colors, edgecolor='#333333', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(f_names, fontsize=8.5)
    ax.set_xlabel('Learned Real-Valued Weight (w_i)', fontsize=10.5, fontweight='bold')
    ax.set_title('Linear SVM Feature Weights Across 22 Learning to Rank Features', fontsize=11.5, fontweight='bold', pad=10)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='-')
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.set_xlim(-0.25, 0.38)
    
    for i, v in enumerate(f_vals):
        if abs(v) > 0.07:
            offset = 0.008 if v >= 0 else -0.045
            ax.text(v + offset, i, f"{v:+.3f}", va='center', fontsize=7.5, fontweight='bold', color='#111111')
            
    plt.tight_layout()
    chart3_path = os.path.join(CHARTS_DIR, 'svm_feature_weights.png')
    plt.savefig(chart3_path)
    plt.close()
    print(f"Saved {chart3_path}")

    # 4. Method A Learned Zone Weights
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8), dpi=300)
    zones = ['Title (.T)', 'Author (.A)', 'Bib (.B)', 'Body (.W)']
    zone_colors = ['#3498db', '#9b59b6', '#f1c40f', '#e67e22']
    
    # MAP optimal weights
    w_map = method_a_w
    ax1.pie(w_map, labels=zones, autopct='%1.1f%%', startangle=140, colors=zone_colors,
            textprops={'fontsize': 9, 'fontweight': 'bold'}, wedgeprops={'edgecolor': 'black', 'linewidth': 0.8})
    ax1.set_title('Grid Search MAP Weights\n(Title=0.55, Author=0.05, Bib=0.40, Body=0.00)', fontsize=10, fontweight='bold')

    # MSE optimal weights
    w_mse = [0.0, 0.9, 0.1, 0.0]
    ax2.pie(w_mse, labels=zones, autopct='%1.1f%%', startangle=140, colors=zone_colors,
            textprops={'fontsize': 9, 'fontweight': 'bold'}, wedgeprops={'edgecolor': 'black', 'linewidth': 0.8})
    ax2.set_title('Minimum MSE Weights\n(Title=0.00, Author=0.90, Bib=0.10, Body=0.00)', fontsize=10, fontweight='bold')

    plt.suptitle('Method A: Comparison of Learned Field Weights (MAP vs. MSE Objectives)', fontsize=11.5, fontweight='bold', y=1.02)
    plt.tight_layout()
    chart4_path = os.path.join(CHARTS_DIR, 'zone_weights.png')
    plt.savefig(chart4_path)
    plt.close()
    print(f"Saved {chart4_path}")

    # 5. 5-Fold Cross Validation Comparison
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=300)
    cv_models = ["Unweighted Zone", "Method A (Zone)", "VSM (Cosine)", "Okapi BM25", "Method B (SVM)"]
    cv_keys = ["Unweighted Zone", "Method A (Weighted Zone)", "VSM (TF-IDF Cosine)", "Okapi BM25", "Method B (Linear SVM L2R)"]
    
    map_means = [cv_summary[k]['MAP']['mean'] for k in cv_keys]
    map_stds = [cv_summary[k]['MAP']['std'] for k in cv_keys]
    f1_means = [cv_summary[k]['F1@10']['mean'] for k in cv_keys]
    f1_stds = [cv_summary[k]['F1@10']['std'] for k in cv_keys]
    
    x_pos = np.arange(len(cv_models))
    w_bar = 0.35
    
    ax.bar(x_pos - w_bar/2, map_means, w_bar, yerr=map_stds, capsize=4, label='MAP (Mean +/- Std)', color='#3498db', edgecolor='black', linewidth=0.6)
    ax.bar(x_pos + w_bar/2, f1_means, w_bar, yerr=f1_stds, capsize=4, label='F1@10 (Mean +/- Std)', color='#2ecc71', edgecolor='black', linewidth=0.6)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cv_models, fontsize=9, fontweight='bold')
    ax.set_ylabel('Score', fontsize=10.5, fontweight='bold')
    ax.set_title('5-Fold Cross Validation Performance (All 225 Queries)', fontsize=11.5, fontweight='bold', pad=10)
    ax.legend(frameon=True, fontsize=9, loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 0.45)
    
    plt.tight_layout()
    chart5_path = os.path.join(CHARTS_DIR, 'cross_validation.png')
    plt.savefig(chart5_path)
    plt.close()
    print(f"Saved {chart5_path}")

if __name__ == '__main__':
    generate_all_charts()
