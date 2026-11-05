# Learning to Rank (L2R) on the Cranfield Collection

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Course: Information Retrieval](https://img.shields.io/badge/Course-Information%20Retrieval%20(CS60092)-darkgreen.svg)](#)

> **Information Retrieval (CS60092) — Programming Assignment III**  
> **Group Name:** `search-o-SAURS`  
> **Student Name:** Souhardya Dandapat  
> **Roll Number:** 23CS30054  
> **Department:** Computer Science & Engineering, IIT Kharagpur  
> **Semester:** 7th Semester (Autumn 2026)

---

A comprehensive implementation, empirical benchmark, and evaluation of **Learning to Rank (L2R)** algorithms evaluated on the standard **Cranfield 1,400** aeronautics collection. This project implements:
1. **Method A: Boolean Weighted Zone Scoring** learned via fast vectorized simplex grid search optimizing Mean Average Precision (MAP).
2. **Method B: Learning Real-Valued Weights using Linear Support Vector Machines (SVM)** trained on the exact 22 query-document ranking features.
3. **Competitive Baselines:** Unweighted Zone Scoring, Vector Space Model (TF-IDF Cosine Similarity), and Okapi BM25 ($k_1=1.5, b=0.75$).
4. **Statistical Rigor:** Evaluated on an 80/20 train-test split (180 training, 45 test queries) and 5-fold cross-validation across all 225 queries.
5. **Hierarchical Architecture:** Structured cleanly into `code/`, `data/`, and `results/` folders with a standalone prediction script (`code/predict.py`) for evaluating unseen test queries.

---

## Table of Contents
- [Repository Structure](#repository-structure)
- [Dataset Architecture](#dataset-architecture)
- [Methodology](#methodology)
  - [Method A: Boolean Weighted Zone Scoring](#method-a-boolean-weighted-zone-scoring)
  - [Method B: Linear SVM Learning to Rank](#method-b-linear-svm-learning-to-rank)
  - [The 22 Ranking Features](#the-22-ranking-features)
  - [Addressing IR Feature Skew with RobustScaler](#addressing-ir-feature-skew-with-robustscaler)
- [Empirical Results](#empirical-results)
  - [Held-Out Test Set Performance](#held-out-test-set-performance)
  - [5-Fold Cross-Validation](#5-fold-cross-validation)
  - [Learned Linear SVM Feature Weights](#learned-linear-svm-feature-weights)
- [Installation and Environment Setup](#installation-and-environment-setup)
- [Usage Guide](#usage-guide)
  - [Running the Complete Benchmark](#1-running-the-complete-benchmark)
  - [Evaluating Unseen Test Queries](#2-evaluating-unseen-test-queries)
  - [Regenerating Visualizations](#3-regenerating-visualizations)
  - [Recompiling the Academic Report](#4-recompiling-the-academic-report)
- [Submission Deliverables](#submission-deliverables)

---

## Repository Structure

The project is organized in a clear hierarchical folder structure:

```text
.
├── code/                           # Modular Python program files
│   ├── dataset_parser.py           # Robust parser, tokenizer, Porter stemmer, and indexer
│   ├── feature_extractor.py        # Extraction engine for all 22 ranking features
│   ├── method_a_zone_scoring.py    # Vectorized simplex grid search for zone weights
│   ├── method_b_svm.py             # Linear SVM L2R with RobustScaler and candidate sampling
│   ├── evaluate.py                 # Evaluation suite (P@K, R@K, F1@K, MAP, MRR, NDCG@10)
│   ├── main.py                     # End-to-end benchmark (80/20 split & 5-fold CV)
│   ├── predict.py                  # Standalone CLI for unseen test query ranking
│   ├── generate_charts.py          # Script generating all publication figures
│   └── generate_report.py          # ReportLab compiler generating report.pdf
├── data/                           # Data directory
│   ├── cran/                       # Authoritative Cranfield collection
│   │   ├── cran.all.1400           # 1,400 documents with 4 zones (.T, .A, .B, .W)
│   │   ├── cran.qry                # 225 natural language queries (.I, .W)
│   │   ├── cranqrel                # Human relevance judgments (queries 1 to 225)
│   │   └── cranqrel.readme         # Cranfield relevance judgment format guide
│   ├── Cranfield/                  # 1,400 individual text documents
│   └── TEST/                       # Queries (query.txt) and relevance files (RES/)
├── results/                        # Generated evaluation outputs and predictions
│   ├── results_method_a.txt        # Method A test ranking results
│   ├── results_method_b.txt        # Method B (Linear SVM) test ranking results
│   ├── results_bm25_baseline.txt   # Okapi BM25 baseline test ranking results
│   └── experiment_results.json     # Complete structured evaluation outputs
├── report_charts/                  # High-resolution (300 DPI) publication figures
│   ├── zone_weights.png            # Method A zone weight distributions
│   ├── svm_feature_weights.png     # Method B learned 22 feature coefficients
│   ├── model_comparison.png        # Bar chart comparing P@10, R@10, F1@10, MAP, MRR
│   ├── cutoffs_comparison.png      # Precision & Recall curves across K = 5, 10, 20
│   └── cross_validation.png        # 5-fold cross-validation performance with error bars
├── requirements.txt                # Pinned Python package dependencies
├── .gitignore                      # Git exclusion rules
├── PA3.pdf                         # Official assignment problem description
└── README.md                       # Comprehensive repository documentation
```

> **Note on Report:** The 10-page academic report (`report.pdf`) is compiled locally using `python code/generate_report.py` and submitted directly via Moodle as required by the assignment guidelines (excluded from the public Git repository).

---

## Dataset Architecture

The Cranfield collection represents a foundational benchmark in information retrieval:
* **Documents ($N = 1,400$):** Sourced from aeronautical engineering literature. The authoritative dataset is `data/cran/cran.all.1400`, which contains four explicit document zones:
  - `.T`: Document Title
  - `.A`: Author(s)
  - `.B`: Bibliography / Affiliation
  - `.W`: Main Body / Abstract
* **Queries ($M = 225$):** Natural language engineering information requests.
* **Relevance Assessments:** Mapped from `data/cran/cranqrel` (scale 1 to 4 treated as relevant, matching the assignment criteria).

---

## Methodology

### Method A: Boolean Weighted Zone Scoring
For a query $q$ and document $d$, each zone $z \in \mathcal{Z} = \{\text{title}, \text{author}, \text{bibliography}, \text{body}\}$ receives a binary match indicator:
$$s_z(q, d) = \begin{cases} 1 & \text{if } q \cap d_z \neq \emptyset \\ 0 & \text{otherwise} \end{cases}$$

The overall document zone score is:
$$S_A(q, d) = \sum_{z \in \mathcal{Z}} w_z \cdot s_z(q, d), \quad \text{subject to } \sum_{z \in \mathcal{Z}} w_z = 1, \; w_z \ge 0$$

An exhaustive **Simplex Grid Search** with step size $\delta = 0.05$ evaluates all 1,771 valid weight quadruplets $(w_T, w_A, w_B, w_W)$. To ensure computational efficiency across 180 training queries ($180 \times 1400 = 252,000$ evaluations per candidate), candidate scoring is fully vectorized via BLAS matrix multiplication:
$$\mathbf{P} = \mathbf{X}_q \mathbf{W}^T \in \mathbb{R}^{1400 \times 1771}$$

**Learned Weights:** The MAP-optimal grid search converged to:
$$(w_{\text{title}} = 0.55, \quad w_{\text{author}} = 0.05, \quad w_{\text{bibliography}} = 0.40, \quad w_{\text{body}} = 0.00)$$

*Insight:* In specialized technical abstracts, title keywords and bibliographic citations carry high discriminative precision, whereas body keywords match non-specifically across the collection.

---

### Method B: Linear SVM Learning to Rank

Under Method B, ranking is formulated as learning a real-valued scoring function:
$$f(q, d) = \mathbf{w}^T \phi(q, d) + b$$
where $\phi(q, d) \in \mathbb{R}^{22}$ is the 22-dimensional feature vector.

#### The 22 Ranking Features
The system computes the exact 22 features required by the specification:

| # | Feature Name | Mathematical Description |
|---|---|---|
| $f_1$ | `query_term_number` | Number of distinct query terms: $\|q\|$ |
| $f_2$ | `query_term_ratio` | Ratio of matched query terms in document: $\frac{\|q \cap d\|}{\|q\|}$ |
| $f_3$ | `stream_length` | Total token length of document: $\|d\|$ |
| $f_4$ | `idf` | Sum of Robertson-Spärck Jones IDF for matched query terms |
| $f_5$ | `sum_of_term_frequency` | Sum of raw term frequencies: $\sum_{t \in q} \text{tf}(t, d)$ |
| $f_6$ | `min_of_term_frequency` | Minimum raw term frequency among matched terms |
| $f_7$ | `max_of_term_frequency` | Maximum raw term frequency among matched terms |
| $f_8$ | `mean_of_term_frequency` | Mean raw term frequency: $\frac{1}{\|q\|} \sum_{t \in q} \text{tf}(t, d)$ |
| $f_9$ | `variance_of_term_frequency` | Variance of raw term frequencies across query terms |
| $f_{10}$ | `sum_of_stream_length_norm_tf` | Sum of length-normalized frequencies: $\sum_{t \in q} \frac{\text{tf}(t, d)}{\|d\|}$ |
| $f_{11}$ | `min_of_stream_length_norm_tf` | Minimum length-normalized term frequency |
| $f_{12}$ | `max_of_stream_length_norm_tf` | Maximum length-normalized term frequency |
| $f_{13}$ | `mean_of_stream_length_norm_tf` | Mean length-normalized term frequency |
| $f_{14}$ | `variance_of_stream_length_norm_tf`| Variance of length-normalized term frequencies |
| $f_{15}$ | `sum_of_tf_idf` | Sum of TF-IDF scores: $\sum_{t \in q} (1 + \ln \text{tf}(t, d)) \cdot \text{idf}(t)$ |
| $f_{16}$ | `min_of_tf_idf` | Minimum TF-IDF among matched query terms |
| $f_{17}$ | `max_of_tf_idf` | Maximum TF-IDF among matched query terms |
| $f_{18}$ | `mean_of_tf_idf` | Mean TF-IDF across all query terms |
| $f_{19}$ | `variance_of_tf_idf` | Variance of TF-IDF scores across query terms |
| $f_{20}$ | `boolean_model` | Exact Boolean conjunct match: $\mathbb{I}[\forall t \in q, t \in d]$ |
| $f_{21}$ | `vector_space_model` | Cosine similarity under ltc.lnc weighting: $\cos(\mathbf{v}_q, \mathbf{v}_d)$ |
| $f_{22}$ | `BM25` | Okapi BM25 score with $k_1 = 1.5, b = 0.75$ |

#### Addressing IR Feature Skew with RobustScaler
In information retrieval, lexical features exhibit heavy-tailed, power-law distributions. When standardized via conventional Z-score normalization ($\mu, \sigma$), extreme document lengths and raw term frequencies compress over 95% of documents near zero, crippling linear separation.

We solved this by integrating `RobustScaler`, which centers and scales features using the **median and Interquartile Range (IQR)**:
$$\hat{\phi}_j = \frac{\phi_j - \text{median}(\phi_j)}{Q_3(\phi_j) - Q_1(\phi_j)}$$
This preserves discrimination among competitive top candidates without allowing extreme document lengths to distort the ranking boundary.

#### Optimization Details
* **Objective:** L2-regularized Squared Hinge Loss with coordinate descent (`loss='squared_hinge'`, `dual=False`).
* **Imbalance Handling:** Query-candidate negative sampling ($1:3$ positive-to-negative ratio) combined with `class_weight='balanced'`.
* **Regularization:** Tuned to $C = 0.005$, preventing overfitting to dense low-level TF statistics while anchoring on high-level retrieval signals.

---

## Empirical Results

### Held-Out Test Set Performance
Evaluated on the independent test partition (45 queries, 63,000 query-document candidate pairs):

| Model | P@5 | P@10 | P@20 | R@10 | R@20 | F1@10 | F1@20 | MAP | MRR | NDCG@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unweighted Zone Baseline** | 0.0889 | 0.0711 | 0.0556 | 0.1189 | 0.1844 | 0.0827 | 0.0792 | 0.0934 | 0.2015 | 0.1147 |
| **Method A (Zone Scoring)** | 0.1556 | 0.1267 | 0.0967 | 0.2144 | 0.3168 | 0.1472 | 0.1384 | 0.1706 | 0.3702 | 0.2035 |
| **VSM Baseline (TF-IDF)** | 0.3378 | 0.2511 | 0.1733 | 0.3953 | 0.5122 | 0.2796 | 0.2370 | **0.3276** | 0.5574 | 0.3929 |
| **Okapi BM25 Baseline** | 0.3244 | 0.2511 | 0.1733 | 0.3804 | 0.5122 | 0.2743 | 0.2370 | 0.3226 | 0.5621 | 0.3888 |
| **Method B (Linear SVM)** | **0.3422** | **0.2556** | **0.1800** | **0.4038** | **0.5326** | **0.2824** | **0.2452** | 0.3195 | **0.5761** | **0.4041** |

> **Key Takeaway:** Method B outperforms Okapi BM25 and VSM across **P@5, P@10, P@20, R@10, R@20, F1@10, F1@20, MRR, and NDCG@10**.

---

### 5-Fold Cross-Validation
Across all 225 queries ($5 \times 45$ validation queries):

| Metric | Method A (Tuned) | VSM Baseline | Okapi BM25 Baseline | Method B (Linear SVM) |
| :--- | :---: | :---: | :---: | :---: |
| **Precision@10** | $0.1360 \pm 0.0163$ | $0.2453 \pm 0.0305$ | $0.2382 \pm 0.0249$ | **$0.2462 \pm 0.0297$** |
| **Recall@10** | $0.2388 \pm 0.0315$ | $0.4190 \pm 0.0526$ | $0.3998 \pm 0.0416$ | **$0.4149 \pm 0.0461$** |
| **F1-Score@10** | $0.1580 \pm 0.0165$ | $0.2779 \pm 0.0270$ | $0.2697 \pm 0.0205$ | **$0.2785 \pm 0.0232$** |
| **MAP** | $0.1632 \pm 0.0142$ | **$0.3340 \pm 0.0287$** | $0.3217 \pm 0.0253$ | $0.3155 \pm 0.0226$ |

---

### Learned Linear SVM Feature Weights

The learned feature coefficients provide clear interpretability into the ranking mechanism:

| Rank | Feature | SVM Weight ($w_j$) | Role in Ranking |
| :---: | :--- | :---: | :--- |
| 1 | `vector_space_model` ($f_{21}$) | **+0.3049** | Primary angle/cosine alignment |
| 2 | `BM25` ($f_{22}$) | **+0.2278** | Primary non-linear term saturation |
| 3 | `max_of_tf_idf` ($f_{17}$) | **+0.1836** | Peak salience across query terms |
| 4 | `mean_of_tf_idf` ($f_{18}$) | **+0.1445** | Average thematic relevance |
| 5 | `sum_of_tf_idf` ($f_{15}$) | **+0.1018** | Cumulative query match weight |
| 6 | `idf` ($f_4$) | **+0.0768** | Rare term coverage |
| 7 | `query_term_ratio` ($f_2$) | **+0.0543** | Breadth of query overlap |
| 8 | `boolean_model` ($f_{20}$) | **+0.0478** | Strict conjunct bonus |
| ... | ... | ... | ... |
| 21 | `variance_of_term_freq` ($f_9$) | **-0.0886** | Penalizes unbalanced term repetitions |
| 22 | `stream_length` ($f_3$) | **-0.1245** | Corrects document verbosity bias |

---

## Installation and Environment Setup

### Prerequisites
* Python 3.10, 3.11, or 3.12
* Git and standard Unix build tools

### Setup Instructions

```bash
# 1. Clone the repository
git clone git@github.com:souhardya2023/Learning_to_rank_assgn3_IR.git
cd Learning_to_rank_assgn3_IR

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage Guide

### 1. Running the Complete Benchmark
Execute the end-to-end evaluation, which runs the 80/20 train-test split, 5-fold cross-validation, and produces all test prediction files in `results/`:

```bash
python code/main.py
```

*Output Files Generated in `results/`:*
* `results/results_method_a.txt`: Test predictions under Method A (4,500 lines, top-100 per query).
* `results/results_method_b.txt`: Test predictions under Method B Linear SVM (4,500 lines).
* `results/results_bm25_baseline.txt`: Test predictions under Okapi BM25 (4,500 lines).
* `results/experiment_results.json`: Complete JSON summary of metrics and cross-validation folds.

---

### 2. Evaluating Unseen Test Queries
Use `code/predict.py` to evaluate arbitrary new query files formatted according to the Cranfield specification:

```bash
# Rank using the top-performing Linear SVM (Method B)
python code/predict.py \
    --queries data/cran/cran.qry \
    --output results/predictions_svm.txt \
    --method svm \
    --top_k 100

# Rank using Method A (Boolean Weighted Zone Scoring)
python code/predict.py \
    --queries data/cran/cran.qry \
    --output results/predictions_zone.txt \
    --method zone \
    --top_k 100

# Rank using Okapi BM25 baseline
python code/predict.py \
    --queries data/cran/cran.qry \
    --output results/predictions_bm25.txt \
    --method bm25 \
    --top_k 100
```

#### Output File Format
Prediction files adhere strictly to standard TREC/IR ranking formatting:
```text
<query_id> <rank> <doc_id> <score>
1 1 184 1.849201
1 2 29 1.621043
1 3 31 1.487192
...
```

---

### 3. Regenerating Visualizations
To regenerate the 300-DPI publication charts in `report_charts/`:

```bash
python code/generate_charts.py
```

---

### 4. Recompiling the Academic Report
To recompile the full academic PDF report (`report.pdf`):

```bash
python code/generate_report.py
```

The resulting `report.pdf` strictly conforms to the 6 required sections:
1. **Group Details** (`search-o-SAURS`, Souhardya Dandapat, 23CS30054)
2. **Problem Statement**
3. **Implementation Details**
4. **Plan of Experiments**
5. **Results**
6. **Discussion of Results**

---

## Submission Deliverables

- [x] **Program Files:** Structured under `code/` (`dataset_parser.py`, `method_a_zone_scoring.py`, `feature_extractor.py`, `method_b_svm.py`, `evaluate.py`, `main.py`, `predict.py`).
- [x] **Data Files:** Structured under `data/` (`data/cran/`, `data/Cranfield/`, `data/TEST/`).
- [x] **Result Files:** Structured under `results/` (`results_method_a.txt`, `results_method_b.txt`, `results_bm25_baseline.txt`, `experiment_results.json`).
- [x] **Academic Report:** `report.pdf` (compiled locally, 10 pages).
- [x] **Evaluation Script:** `code/predict.py` with full CLI argument support.
- [x] **Configuration & Setup:** `requirements.txt`, `.gitignore`, and comprehensive `README.md`.

---

## License
This repository is open-sourced under the MIT License. Academic evaluations and datasets are governed by the respective Cranfield Collection terms.
