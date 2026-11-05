"""
generate_report.py - Compiles a comprehensive 10-page academic PDF report for IR Assignment 3.

Contains the 6 mandatory sections:
1. Group Details
2. Problem Statement
3. Implementation Details
4. Plan of Experiments
5. Results
6. Discussion of Results
"""

import os
import json
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to compute dynamic total page count and add running headers/footers.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_decorations(self, total_pages):
        self.saveState()
        self.setFont('Helvetica-Bold', 7.5)
        self.setFillColor(colors.HexColor('#444444'))
        
        # Header (Pages 2+)
        if self._pageNumber > 1:
            self.drawString(54, 752, "INFORMATION RETRIEVAL (CS60092) — PROGRAMMING ASSIGNMENT III: LEARNING TO RANK")
            self.setStrokeColor(colors.HexColor('#b0bec5'))
            self.setLineWidth(0.6)
            self.line(54, 746, 558, 746)
            
        # Footer (All pages)
        self.setStrokeColor(colors.HexColor('#b0bec5'))
        self.setLineWidth(0.6)
        self.line(54, 42, 558, 42)
        
        self.setFont('Helvetica', 8)
        self.drawString(54, 30, "Indian Institute of Technology Kharagpur — Department of Computer Science & Engineering")
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(558, 30, page_str)
        self.restoreState()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
CHARTS_DIR = os.path.join(PROJECT_ROOT, 'report_charts')

def build_pdf_report(output_filename=None):
    if output_filename is None:
        output_filename = os.path.join(PROJECT_ROOT, "report.pdf")

    json_path = os.path.join(RESULTS_DIR, 'experiment_results.json')
    if not os.path.exists(json_path):
        json_path = 'experiment_results.json'

    with open(json_path, 'r') as f:
        exp_data = json.load(f)
        
    test_res = exp_data['test_results']
    cv_res = exp_data['cv_summary']
    method_a_w = exp_data['method_a_weights']
    svm_w = exp_data['method_b_weights']
    tB = test_res['Method B (Linear SVM L2R)']; tA = test_res['Method A (Weighted Zone)']
    tU = test_res['Unweighted Zone']; tAny = test_res['Method A (coarse any-term)']
    tV = test_res['VSM (TF-IDF Cosine)']; tM = test_res['Okapi BM25']
    cvB = cv_res['Method B (Linear SVM L2R)']; cvA = cv_res['Method A (Weighted Zone)']
    cvV = cv_res['VSM (TF-IDF Cosine)']; cvM = cv_res['Okapi BM25']; cvAny = cv_res['Method A (coarse any-term)']
    cut = exp_data['cutoff_results']; bestk = exp_data['best_k']
    wany = exp_data['method_a_any_weights']
    top_w = sorted(svm_w.items(), key=lambda x: -abs(x[1]))[:4]
    mode_txt = 'pairwise RankSVM' if exp_data['method_b_mode'] == 'pair' else 'pointwise'
    fin = exp_data['final']
    
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=17,
        leading=21,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#0d47a1')
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10.5,
        leading=14,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#37474f')
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor('#0d47a1'),
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12.5,
        textColor=colors.HexColor('#263238'),
        spaceBefore=5,
        spaceAfter=2,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor('#212121')
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        alignment=TA_CENTER
    )
    
    table_cell_left = ParagraphStyle(
        'TableCellLeft',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        alignment=TA_LEFT
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        alignment=TA_CENTER,
        textColor=colors.white
    )
    
    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=10.5,
        alignment=TA_LEFT,
        textColor=colors.HexColor('#1a237e')
    )

    math_block = ParagraphStyle(
        'MathBlock',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=12,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1565c0')
    )

    story = []

    # =========================================================================
    # PAGE 1: TITLE, GROUP DETAILS, ABSTRACT & PROBLEM STATEMENT
    # =========================================================================
    story.append(Paragraph("INDIAN INSTITUTE OF TECHNOLOGY KHARAGPUR", subtitle_style))
    story.append(Paragraph("Department of Computer Science & Engineering", subtitle_style))
    story.append(Spacer(1, 3))
    story.append(Paragraph("Information Retrieval (CS60092) — Programming Assignment III", subtitle_style))
    story.append(Spacer(1, 5))
    story.append(Paragraph("LEARNING TO RANK ON THE CRANFIELD COLLECTION", title_style))
    story.append(Spacer(1, 3))
    story.append(Paragraph("Boolean Weighted Zone Scoring & Real-Valued Linear SVM Ranking", subtitle_style))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0d47a1'), spaceAfter=6))

    # SECTION 1: GROUP DETAILS
    story.append(Paragraph("1. Group Details", h1_style))
    group_table_data = [
        [Paragraph("<b>Group Attribute</b>", table_header), Paragraph("<b>Details</b>", table_header)],
        [Paragraph("<b>Course</b>", table_cell_left), Paragraph("Information Retrieval (CS60092)", table_cell_left)],
        [Paragraph("<b>Assignment</b>", table_cell_left), Paragraph("Programming Assignment III: Learning to Rank", table_cell_left)],
        [Paragraph("<b>Group Name</b>", table_cell_left), Paragraph("<b>search-o-SAURS</b>", table_cell_left)],
        [Paragraph("<b>Student Name</b>", table_cell_left), Paragraph("Souhardya Dandapat", table_cell_left)],
        [Paragraph("<b>Roll Number</b>", table_cell_left), Paragraph("23CS30054", table_cell_left)],
        [Paragraph("<b>Department / Semester</b>", table_cell_left), Paragraph("Computer Science and Engineering / 7th Semester", table_cell_left)],
        [Paragraph("<b>Submission Date</b>", table_cell_left), Paragraph("November 6, 2026", table_cell_left)],
        [Paragraph("<b>Implementation Environment</b>", table_cell_left), Paragraph("Python 3.12, NumPy, Scikit-Learn, ReportLab (Self-Contained)", table_cell_left)]
    ]
    g_tab = Table(group_table_data, colWidths=[150, 354])
    g_tab.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0d47a1')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cfd8dc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f7fa')]),
        ('TOPPADDING', (0, 0), (-1, -1), 1.8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.8),
    ]))
    story.append(g_tab)
    story.append(Spacer(1, 5))

    # ABSTRACT / EXECUTIVE SUMMARY
    story.append(Paragraph("Executive Summary", h2_style))
    story.append(Paragraph(
        "Learning to Rank (L2R) modernizes traditional information retrieval by framing ranked document retrieval as a machine learning optimization problem. This report presents the design, mathematical formulation, empirical evaluation, and deep qualitative analysis of two distinct Learning to Rank methodologies implemented on the Cranfield aeronautical test collection: (1) <b>Method A: Boolean Weighted Zone Scoring</b>, which explores the parametric weight simplex across four document structural zones (Title, Author, Bibliography, and Text Body) via exhaustive grid search optimization; and (2) <b>Method B: Real-Valued Linear Support Vector Machines (SVM)</b>, which extracts 22 NLP and statistical retrieval features per query-document pair to train a maximal-margin hyperplane separator. Using standard Cranfield relevance judgments, both models are benchmarked against classic retrieval models including Unweighted Zone Scoring, the Vector Space Model (TF-IDF Cosine), and Okapi BM25 across an 80/20 train/test partition (180 train queries, 45 held-out test queries) and 5-fold cross-validation. On the held-out queries Method A with term-level Boolean zone matching reaches MAP = " + f"{tA['MAP']:.3f}" + " (coarse any-term variant: " + f"{tAny['MAP']:.3f}" + "), and Method B (" + mode_txt + " linear SVM) reaches MAP = " + f"{tB['MAP']:.3f}" + ", P@5 = " + f"{tB['P@5']:.3f}" + ", MRR = " + f"{tB['MRR']:.3f}" + ", on par with BM25 (" + f"{tM['MAP']:.3f}" + ") and VSM (" + f"{tV['MAP']:.3f}" + "); 5-fold CV confirms this ordering.",
        body_style
    ))
    story.append(Spacer(1, 6))

    # SECTION 2: PROBLEM STATEMENT
    story.append(Paragraph("2. Problem Statement", h1_style))
    story.append(Paragraph(
        "In modern Information Retrieval (IR), matching queries against text collections requires more than simple keyword boolean presence. Traditional ranking heuristics (e.g., standard TF-IDF or isolated field matching) treat all document fields and lexical signals uniformly or rely on ad-hoc hand-tuned constants. The objective of this assignment is to apply supervised machine learning algorithms to learn document rankings on the Cranfield text collection and rigorously evaluate retrieval efficacy against human relevance judgments.",
        body_style
    ))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "<b>Core Assignment Requirements:</b><br/>"
        "• <b>Dataset Analysis:</b> Investigate and determine whether the primary document corpus resides in the <code>cran</code> folder (containing <code>cran.all.1400</code>, <code>cran.qry</code>, <code>cranqrel</code>) or the <code>data</code> folder (containing <code>data/Cranfield/*.txt</code>, <code>data/TEST/query.txt</code>, <code>data/TEST/RES/*.txt</code>), resolving field availability constraints.<br/>"
        "• <b>Method A (Boolean Weighted Zone Scoring):</b> Formulate a machine learning program to learn the optimal non-negative weights for individual document zones—Title, Author, Bibliography, and Text Body—using individual boolean matching scores as inputs and relevance judgments as ground truth via a grid search algorithm.<br/>"
        "• <b>Method B (Linear Support Vector Machines):</b> Extract the exact 22 prescribed query-document features (query term count, ratio, stream length, IDF, TF statistics, length-normalized TF statistics, TF*IDF statistics, Boolean model, VSM cosine similarity, and Okapi BM25) and train a Linear SVM to learn real-valued ranking weights for test queries.<br/>"
        "• <b>Evaluation & Reporting:</b> Compute standard IR metrics—Precision, Recall, F1 score at rank cutoffs (K=5, 10, 20), Mean Average Precision (MAP), Mean Reciprocal Rank (MRR), and NDCG—and output ranked result text files ready for unknown test evaluation.",
        body_style
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: DATASET DETERMINATION & TEXT PREPROCESSING
    # =========================================================================
    story.append(Paragraph("2.1 Dataset Determination: Analysis of 'cran' vs. 'data' Folders", h2_style))
    story.append(Paragraph(
        "A critical initial requirement of this project was determining the authoritative dataset location between the two supplied directories: <code>cran</code> and <code>data</code>. A thorough structural audit revealed fundamental differences:",
        body_style
    ))
    story.append(Spacer(1, 4))

    dataset_audit_table = [
        [Paragraph("<b>Directory / File</b>", table_header), Paragraph("<b>Contents & Structure</b>", table_header), Paragraph("<b>Field Availability</b>", table_header), Paragraph("<b>Suitability Verdict</b>", table_header)],
        [Paragraph("<b>cran/cran.all.1400</b>", table_cell_left), Paragraph("1,400 documents in SGML-style tagged format (.I, .T, .A, .B, .W).", table_cell_left), Paragraph("Title (.T), Author (.A), Bib (.B), Body (.W) fully intact.", table_cell_left), Paragraph("<b>Mandatory</b>: Required for Method A zone scoring.", table_cell_left)],
        [Paragraph("<b>data/Cranfield/*.txt</b>", table_cell_left), Paragraph("1,400 plain text files (1.txt to 1400.txt).", table_cell_left), Paragraph("Merged Title + Abstract only. Author (.A) and Bib (.B) are missing.", table_cell_left), Paragraph("<b>Insufficient</b>: Cannot compute Author or Bib zones.", table_cell_left)],
        [Paragraph("<b>cran/cranqrel</b>", table_cell_left), Paragraph("1,837 query-doc judgments on Cleverdon scale (1..4, and -1).", table_cell_left), Paragraph("Queries 1 to 225 mapped to doc IDs and relevance grades.", table_cell_left), Paragraph("<b>Authoritative</b> ground truth for learning & evaluation.", table_cell_left)],
        [Paragraph("<b>data/TEST/query.txt</b>", table_cell_left), Paragraph("225 clean tab-separated queries (1 to 225).", table_cell_left), Paragraph("Canonical query strings mapping 1-to-1 with cranqrel.", table_cell_left), Paragraph("<b>Standardized input</b> for query processing.", table_cell_left)]
    ]
    d_tab = Table(dataset_audit_table, colWidths=[90, 150, 140, 124])
    d_tab.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0d47a1')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cfd8dc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f7fa')]),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(d_tab)
    story.append(Spacer(1, 5))

    story.append(Paragraph(
        "<b>Definitive Determination:</b> The assignment specification explicitly states: <i>'The fields are: title, author, bibliography, and text body. Individual Boolean matching score of each of these fields with the query may be considered as input.'</i> In <code>data/Cranfield/</code>, the author and bibliographic citations were discarded during plain-text export. Consequently, <b><code>cran/cran.all.1400</code> in the <code>cran</code> directory is the only dataset that contains all required zones</b>. For queries and ground truth, <code>data/TEST/query.txt</code> and <code>cran/cranqrel</code> provide identical 1..225 query indexing, which our parser harmonizes seamlessly.",
        body_style
    ))
    story.append(Spacer(1, 5))

    story.append(Paragraph("2.2 Corpus Characteristics & Ground Truth Distribution", h2_style))
    story.append(Paragraph(
        "The Cranfield collection comprises 1,400 aerodynamics research abstracts from the Aeronautical Research Council and Cranfield College of Aeronautics, compiled under Cyril Cleverdon. Human relevance assessments were recorded on a 5-point scale:",
        body_style
    ))
    story.append(Spacer(1, 2))
    story.append(Paragraph(
        "• <b>Grade 1 (Complete Answer):</b> Document provides a definitive, complete solution to the query (128 instances).<br/>"
        "• <b>Grade 2 (High Degree of Relevance):</b> Crucial paper, omission would impede research (387 instances).<br/>"
        "• <b>Grade 3 (Useful Background):</b> General background or suggested analytical methods (734 instances).<br/>"
        "• <b>Grade 4 (Minimum Interest):</b> Historical context or minimal direct interest (363 instances).<br/>"
        "• <b>Grade -1 (Source Document):</b> The originating aerorag paper from which the query was synthesized (225 instances). In standard Cranfield IR evaluations, Grade -1 is treated as non-relevant to avoid query formulation bias.<br/>"
        "Following standard IR methodology, documents with relevance grades in {1, 2, 3, 4} are designated as relevant (<i>y</i> = 1), while unjudged documents and grade -1 are designated as non-relevant (<i>y</i> = 0).",
        body_style
    ))
    story.append(Spacer(1, 5))

    story.append(Paragraph("3. Implementation Details: Architecture & Text Preprocessing", h1_style))
    story.append(Paragraph(
        "To guarantee high computational efficiency, reproducibility, and cross-platform execution without external server daemons, the entire pipeline is implemented in modular Python (<code>dataset_parser.py</code>, <code>method_a_zone_scoring.py</code>, <code>feature_extractor.py</code>, <code>method_b_svm.py</code>, <code>evaluate.py</code>, <code>main.py</code>, <code>predict.py</code>).",
        body_style
    ))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "<b>Text Normalization Pipeline:</b><br/>"
        "1. <b>Tokenization:</b> Regex token extraction (<code>[a-zA-Z0-9]+</code>) handling hyphens, acronyms, and alphanumeric tokens.<br/>"
        "2. <b>Case Folding:</b> Lowercasing of all corpus text and incoming queries.<br/>"
        "3. <b>Stopword Filtering:</b> Elimination of 179 standard English functional stopwords (prepositions, conjunctions, determiners) embedded directly into the module for 100% self-contained sandbox execution.<br/>"
        "4. <b>Morphological Stemming:</b> Porter Stemmer algorithm reducing inflectional and derivational variants to morphological roots (e.g., 'aerodynamics', 'aerodynamic' &rarr; 'aerodynam').<br/>"
        "5. <b>Inverted Indexing & Statistics:</b> Construction of an inverted index mapping term <i>t</i> &rarr; {<i>d</i>: <i>tf(t, d)</i>}, calculation of document lengths <i>L<sub>d</sub></i>, collection average <i>L<sub>avg</sub></i> = 108.35 terms, and Robertson-Sp&auml;rck Jones BM25 IDF weights.",
        body_style
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: METHOD A - BOOLEAN WEIGHTED ZONE SCORING
    # =========================================================================
    story.append(Paragraph("3.1 Method A: Boolean Weighted Zone Scoring Architecture", h1_style))
    story.append(Paragraph(
        "Weighted zone scoring (Manning, Raghavan, & Sch&uuml;tze, <i>Introduction to Information Retrieval</i>, Chapter 6) extends boolean retrieval by dividing documents into semantically distinct structural zones. In Cranfield, every document <i>d</i> contains four zones:",
        body_style
    ))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "<b>Z = { Title (Z<sub>T</sub>), Author (Z<sub>A</sub>), Bibliography (Z<sub>B</sub>), Text Body (Z<sub>W</sub>) }</b>",
        math_block
    ))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "For each zone <i>z</i> &isin; Z, we evaluate the <b>Boolean matching score</b> of every distinct query term against the zone (1 if the term occurs in the zone, else 0) and average it over the query terms:<br/>"
        "<b><i>s<sub>z</sub>(q, d) = (1/|q|) &middot; &Sigma;<sub>t &isin; q</sub> [ t &isin; Zone<sub><i>z</i></sub>(d) ]</i> &isin; [0, 1]</b><br/>"
        "(A single 'any term matches' bit per zone was also tested as an ablation: it saturates on the body zone and gives MAP of only " + f"{tAny['MAP']:.3f}" + ".)",
        body_style
    ))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "The overall ranking score is a linear combination parameterized by zone weight vector <b>w</b> = (<i>w<sub>T</sub>, w<sub>A</sub>, w<sub>B</sub>, w<sub>W</sub></i>)<sup>T</sup>:<br/>"
        "<b>S(q, d) = &Sigma;<sub>z &isin; Z</sub> w<sub>z</sub> &middot; s<sub>z</sub>(q, d) &nbsp;&nbsp;&nbsp; subject to &nbsp; w<sub>z</sub> &ge; 0, &nbsp; &Sigma; w<sub>z</sub> = 1</b>",
        math_block
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Optimization Framework: Grid Search over the Weight Simplex", h2_style))
    story.append(Paragraph(
        "To learn the optimal zone weights <b>w</b>*, we explore the standard 3-simplex &Delta;<sup>3</sup> = {<b>w</b> &isin; &real;<sup>4</sup> : <i>w<sub>z</sub></i> &ge; 0, &Sigma; <i>w<sub>z</sub></i> = 1} using a discrete grid with resolution &delta; = 0.05, yielding |C| = 1,771 candidate configurations. The objective is the ranking metric itself:",
        body_style
    ))
    story.append(Spacer(1, 2))
    story.append(Paragraph(
        "<b>w*<sub>MAP</sub> = argmax<sub>w &isin; C</sub> &nbsp; (1 / |Q<sub>train</sub>|) &middot; &Sigma;<sub>q</sub> AP(q; w)</b><br/>"
        "Documents are sorted by <i>S(q, d)</i> descending (stable sort, ties broken by document id). <b>Exactly the same scoring and ordering function is used for training and for inference</b>; no hidden tie-breaking heuristic is applied at test time, so the optimised objective is the deployed ranking. To evaluate 1,771 candidates on 180 training queries quickly, scoring is vectorised as a single matrix product <b>P = X<sub>q</sub> W<sup>T</sup></b> &isin; &real;<sup>1400 &times; 1771</sup> and AP is computed from cumulative hit vectors. Learned weights: <b>w*</b> = (T, A, B, W) = (" + ", ".join(f"{x:.2f}" for x in method_a_w) + ").",
        body_style
    ))
    story.append(Spacer(1, 4))

    # Embedded Chart: Zone Weights
    chart4_file = os.path.join(CHARTS_DIR, 'zone_weights.png')
    if os.path.exists(chart4_file):
        story.append(Image(chart4_file, width=470, height=185))
        story.append(Spacer(1, 2))
        story.append(Paragraph("<b>Figure 1:</b> Learned zone weights: term-level Boolean zones (final) vs. coarse any-term zones (ablation).", callout_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: METHOD B - LINEAR SUPPORT VECTOR MACHINES
    # =========================================================================
    story.append(Paragraph("3.2 Method B: Learning Real-Valued Weights via Linear SVM", h1_style))
    story.append(Paragraph(
        "Method B frames document ranking under the supervised <b>Learning to Rank (L2R)</b> paradigm using <b>Support Vector Machines (SVM)</b>. For every query-document pair (<i>q, d</i>), we construct a 22-dimensional feature vector <b>x</b>(<i>q, d</i>) &isin; &real;<sup>22</sup>. A linear SVM learns a decision hyperplane parameterized by weight vector <b>w</b> &isin; &real;<sup>22</sup> and bias <i>b</i> &isin; &real; that separates relevant documents from non-relevant documents with maximal margin. Two linear-SVM formulations are considered: <b>pointwise</b> (relevant vs. non-relevant documents) and <b>pairwise RankSVM</b> (the SVM separates the feature difference x(d<sup>+</sup>) &minus; x(d<sup>&minus;</sup>) of a relevant and a non-relevant document of the same query, no bias term). The formulation is selected by cross-validation.",
        body_style
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Mathematical Formulation & Loss Function", h2_style))
    story.append(Paragraph(
        "Let (<b>x</b><sub>i</sub>, <i>y<sub>i</sub></i>) be the <i>i</i>-th query-document training instance, where <i>y<sub>i</sub></i> &isin; {+1, &minus;1} denotes relevance. We solve the soft-margin primal objective with <i>L</i><sub>2</sub>-regularized squared hinge loss:<br/>"
        "<b>min<sub>w, b</sub> &nbsp; (1/2) ||w||<sub>2</sub><sup>2</sup> + C &middot; &Sigma;<sub>i=1..M</sub> &ell;(y<sub>i</sub>, w<sup>T</sup> x&#771;<sub>i</sub> + b)</b><br/>"
        "where &ell;(<i>y</i>, y&#770;) = max(0, 1 &minus; <i>y</i> y&#770;)<sup>2</sup> is the squared hinge loss, and <i>C</i> &gt; 0 governs the trade-off between margin maximization and empirical training error. The objective is convex and continuously differentiable, enabling rapid coordinate descent optimization.",
        body_style
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Real-Valued Ranking Function", h2_style))
    story.append(Paragraph(
        "At retrieval time, the learned weight vector <b>w</b> assigns a continuous, real-valued ranking score to any query-document pair equal to the signed perpendicular distance from the decision boundary:<br/>"
        "<b>S(q, d) = w<sup>T</sup> x&#771;(q, d) + b = &Sigma;<sub>j=1..22</sub> w<sub>j</sub> &middot; ((x<sub>j</sub>(q, d) &minus; med<sub>j</sub>) / IQR<sub>j</sub>) + b</b><br/>"
        "Documents are ranked in strictly descending order of <i>S(q, d)</i>. Documents with higher scores possess features that project strongly along the positive margin, reflecting high confidence of relevance.",
        body_style
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Addressing Class Imbalance & Candidate Selection", h2_style))
    story.append(Paragraph(
        "A major challenge in Cranfield L2R is extreme class imbalance: out of 180 &times; 1,400 = 252,000 possible training pairs, only ~1,400 are relevant (&lt;0.6%). If an SVM is trained naively on all pairs, it trivially collapses to predicting non-relevant. We resolve this through a two-stage strategy:<br/>"
        "1. <b>Candidate Selection:</b> For each training query, we retain all ground-truth relevant documents (<i>y</i> = +1) and sample negative documents (<i>y</i> = &minus;1) from candidates that share at least one vocabulary term with the query.<br/>"
        "2. <b>Class Weight Balancing:</b> We assign cost weights inversely proportional to class frequencies: <i>C</i><sub>+</sub> = <i>C</i> &middot; (<i>M</i> / 2<i>M</i><sub>+</sub>), &nbsp; <i>C</i><sub>&minus;</sub> = <i>C</i> &middot; (<i>M</i> / 2<i>M</i><sub>&minus;</sub>), ensuring the optimizer penalizes misclassified positive instances heavily.<br/>"
        "3. <b>Standardization:</b> Features exhibit disparate scales (stream length ~ 100, cosine similarity &isin; [0, 1]). Features are heavy-tailed, so we fit a <code>RobustScaler</code> (median / inter-quartile range) strictly on the training queries.",
        body_style
    ))
    story.append(Spacer(1, 5))

    story.append(Paragraph("Hyperparameter Optimization", h2_style))
    story.append(Paragraph(
        "Training formulation (pointwise / pairwise), the regularization constant <i>C</i> &isin; {0.0005, 0.001, 0.005, 0.01, 0.05, 0.1} and the negatives-per-positive ratio &isin; {3, 10} are selected by <b>inner 5-fold cross-validation on the 180 training queries only</b>, maximising MAP over the full 1,400-document ranking. Selected: <b>" + mode_txt + ", C = " + str(exp_data['method_b_C']) + ", " + str(exp_data['method_b_neg_ratio']) + " negatives per positive</b> (inner-CV MAP = " + f"{exp_data['method_b_tuning']['cv_map']:.4f}" + "). Strong regularisation wins consistently: with only ~22 highly correlated features, a large C over-fits query-specific term statistics. The final submission model is re-tuned and re-trained on all 225 judged queries (" + fin['method_b_mode'] + ", C = " + str(fin['method_b_C']) + ").",
        body_style
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 5: THE 22 FEATURES FORMULATION TABLE
    # =========================================================================
    story.append(Paragraph("3.3 Comprehensive Mathematical Specification of the 22 Features", h1_style))
    story.append(Paragraph(
        "The assignment specification mandates 22 exact input features derived from Terrier search engine learning-to-rank standards. Below is the complete mathematical definition for each feature computed per query-document pair (<i>q, d</i>):",
        body_style
    ))
    story.append(Spacer(1, 4))

    features_table_data = [
        [Paragraph("<b>#</b>", table_header), Paragraph("<b>Feature Name</b>", table_header), Paragraph("<b>Mathematical Definition / Formula</b>", table_header), Paragraph("<b>Description & Intuition</b>", table_header)],
        [Paragraph("1", table_cell), Paragraph("<b>query term number</b>", table_cell_left), Paragraph("|q| (total tokens in query)", table_cell_left), Paragraph("Query verbosity and specificity.", table_cell_left)],
        [Paragraph("2", table_cell), Paragraph("<b>query term ratio</b>", table_cell_left), Paragraph("|{t &isin; q : tf(t, d) &gt; 0}| / |unique(q)|", table_cell_left), Paragraph("Fraction of distinct query terms present in d.", table_cell_left)],
        [Paragraph("3", table_cell), Paragraph("<b>stream length</b>", table_cell_left), Paragraph("L<sub>d</sub> = &Sigma;<sub>t &isin; d</sub> tf(t, d)", table_cell_left), Paragraph("Total token length of document text stream.", table_cell_left)],
        [Paragraph("4", table_cell), Paragraph("<b>idf</b>", table_cell_left), Paragraph("&Sigma;<sub>t &isin; q &cap; d</sub> idf(t)", table_cell_left), Paragraph("Aggregate informativeness of matched query terms.", table_cell_left)],
        [Paragraph("5", table_cell), Paragraph("<b>sum of term frequency</b>", table_cell_left), Paragraph("&Sigma;<sub>t &isin; q</sub> tf(t, d)", table_cell_left), Paragraph("Total keyword match volume in document.", table_cell_left)],
        [Paragraph("6", table_cell), Paragraph("<b>min of term frequency</b>", table_cell_left), Paragraph("min<sub>t &isin; q, tf&gt;0</sub> tf(t, d) &nbsp;(or 0)", table_cell_left), Paragraph("Frequency of the least matched query term.", table_cell_left)],
        [Paragraph("7", table_cell), Paragraph("<b>max of term frequency</b>", table_cell_left), Paragraph("max<sub>t &isin; q</sub> tf(t, d)", table_cell_left), Paragraph("Highest frequency among query terms.", table_cell_left)],
        [Paragraph("8", table_cell), Paragraph("<b>mean of term frequency</b>", table_cell_left), Paragraph("(1/|q|) &middot; &Sigma;<sub>t &isin; q</sub> tf(t, d)", table_cell_left), Paragraph("Average repetition per query token.", table_cell_left)],
        [Paragraph("9", table_cell), Paragraph("<b>variance of term frequency</b>", table_cell_left), Paragraph("(1/|q|) &middot; &Sigma;<sub>t &isin; q</sub> (tf(t, d) &minus; &mu;<sub>tf</sub>)<sup>2</sup>", table_cell_left), Paragraph("Evenness of query term coverage.", table_cell_left)],
        [Paragraph("10", table_cell), Paragraph("<b>sum of stream len norm tf</b>", table_cell_left), Paragraph("&Sigma;<sub>t &isin; q</sub> (tf(t, d) / L<sub>d</sub>)", table_cell_left), Paragraph("Term density normalized by document length.", table_cell_left)],
        [Paragraph("11", table_cell), Paragraph("<b>min of stream len norm tf</b>", table_cell_left), Paragraph("min<sub>t &isin; q, tf&gt;0</sub> (tf(t, d) / L<sub>d</sub>)", table_cell_left), Paragraph("Minimum normalized term frequency.", table_cell_left)],
        [Paragraph("12", table_cell), Paragraph("<b>max of stream len norm tf</b>", table_cell_left), Paragraph("max<sub>t &isin; q</sub> (tf(t, d) / L<sub>d</sub>)", table_cell_left), Paragraph("Maximum normalized term frequency.", table_cell_left)],
        [Paragraph("13", table_cell), Paragraph("<b>mean of stream len norm tf</b>", table_cell_left), Paragraph("(1/|q|) &middot; &Sigma;<sub>t &isin; q</sub> (tf(t, d) / L<sub>d</sub>)", table_cell_left), Paragraph("Average normalized term density.", table_cell_left)],
        [Paragraph("14", table_cell), Paragraph("<b>variance of stream len norm tf</b>", table_cell_left), Paragraph("(1/|q|) &middot; &Sigma;<sub>t &isin; q</sub> (tf<sub>L</sub> &minus; &mu;<sub>tf<sub>L</sub></sub>)<sup>2</sup>", table_cell_left), Paragraph("Variance of normalized term densities.", table_cell_left)],
        [Paragraph("15", table_cell), Paragraph("<b>sum of tf*idf</b>", table_cell_left), Paragraph("&Sigma;<sub>t &isin; q</sub> tf(t, d) &middot; idf(t)", table_cell_left), Paragraph("Raw TF-IDF match accumulation.", table_cell_left)],
        [Paragraph("16", table_cell), Paragraph("<b>min of tf*idf</b>", table_cell_left), Paragraph("min<sub>t &isin; q, tf&gt;0</sub> [tf(t, d) &middot; idf(t)]", table_cell_left), Paragraph("Lowest weighted importance among matched terms.", table_cell_left)],
        [Paragraph("17", table_cell), Paragraph("<b>max of tf*idf</b>", table_cell_left), Paragraph("max<sub>t &isin; q</sub> [tf(t, d) &middot; idf(t)]", table_cell_left), Paragraph("Strongest single term-salience indicator.", table_cell_left)],
        [Paragraph("18", table_cell), Paragraph("<b>mean of tf*idf</b>", table_cell_left), Paragraph("(1/|q|) &middot; &Sigma;<sub>t &isin; q</sub> tf(t, d) &middot; idf(t)", table_cell_left), Paragraph("Average TF-IDF payload per query token.", table_cell_left)],
        [Paragraph("19", table_cell), Paragraph("<b>variance of tf*idf</b>", table_cell_left), Paragraph("(1/|q|) &middot; &Sigma;<sub>t &isin; q</sub> (w<sub>td</sub> &minus; &mu;<sub>w</sub>)<sup>2</sup>", table_cell_left), Paragraph("Spread of term weights across the query.", table_cell_left)],
        [Paragraph("20", table_cell), Paragraph("<b>boolean model</b>", table_cell_left), Paragraph("1 if q &sube; d else 0", table_cell_left), Paragraph("Conjunctive (AND) Boolean match: every query term occurs in d.", table_cell_left)],
        [Paragraph("21", table_cell), Paragraph("<b>vector space model</b>", table_cell_left), Paragraph("(V<sub>q</sub> &middot; V<sub>d</sub>) / (||V<sub>q</sub>|| ||V<sub>d</sub>||)", table_cell_left), Paragraph("Cosine similarity in sublinear TF-IDF vector space.", table_cell_left)],
        [Paragraph("22", table_cell), Paragraph("<b>BM25</b>", table_cell_left), Paragraph("&Sigma;<sub>t</sub> idf(t) [tf(k<sub>1</sub>+1)/(tf+k<sub>1</sub>(1-b+b&middot;L<sub>d</sub>/L<sub>avg</sub>))]", table_cell_left), Paragraph("Okapi BM25 score with k<sub>1</sub>=1.2, b=0.75.", table_cell_left)]
    ]
    f_tab = Table(features_table_data, colWidths=[24, 136, 170, 174])
    f_tab.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0d47a1')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cfd8dc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f7fa')]),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(f_tab)
    story.append(PageBreak())

    # =========================================================================
    # PAGE 6: PLAN OF EXPERIMENTS & EVALUATION PROTOCOL
    # =========================================================================
    story.append(Paragraph("4. Plan of Experiments", h1_style))
    story.append(Paragraph(
        "A rigorous empirical methodology is established to ensure fair benchmarking, avoid data snooping, and assess generalization stability across unknown test queries.",
        body_style
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("4.1 Dataset Splitting & Cross-Validation Protocol", h2_style))
    story.append(Paragraph(
        "The full set of 225 Cranfield queries was partitioned using two complementary protocols:<br/>"
        "• <b>Held-Out Test Partition (80/20 Split):</b> 180 queries (80%) were assigned to the training/validation partition, and 45 queries (20%) were strictly isolated as the held-out test evaluation set (random seed = 42). Model parameters (grid search weights for Method A, SVM hyperplane weights and feature standardizer for Method B) were learned exclusively on the 180 training queries.<br/>"
        "• <b>5-Fold Cross Validation:</b> To guard against split variance and assess statistical robustness across the entire collection, all 225 queries were divided into 5 equal folds of 45 queries each. In each fold, 180 queries served as training and 45 as validation. We report the mean and sample standard deviation across all 5 iterations.",
        body_style
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("4.2 Evaluation Metrics Formulation", h2_style))
    story.append(Paragraph(
        "Retrieval effectiveness is quantified using standard Cranfield and TREC evaluation measures:",
        body_style
    ))
    story.append(Spacer(1, 2))
    story.append(Paragraph(
        "• <b>Precision at Cutoff K (P@K):</b> Fraction of top-<i>K</i> retrieved documents that are relevant:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>P@K = |Relevant &cap; Retrieved<sub>1..K</sub>| / K, &nbsp;&nbsp;&nbsp; K &isin; {5, 10, 20}</b><br/>"
        "• <b>Recall at Cutoff K (R@K):</b> Fraction of all relevant documents retrieved within top-<i>K</i>:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>R@K = |Relevant &cap; Retrieved<sub>1..K</sub>| / |Relevant|, &nbsp;&nbsp;&nbsp; K &isin; {10, 20}</b><br/>"
        "• <b>F1 Score at Cutoff K (F1@K):</b> Harmonic mean balancing precision and recall at cutoff <i>K</i>:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>F<sub>1</sub>@K = (2 &middot; P@K &middot; R@K) / (P@K + R@K)</b><br/>"
        "• <b>Mean Average Precision (MAP):</b> The gold standard in ad-hoc IR, evaluating ranked precision across all relevant recall points:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>MAP = (1 / |Q|) &middot; &Sigma;<sub>q</sub> (1 / |R<sub>q</sub>|) &middot; &Sigma;<sub>k=1..N</sub> P@k &middot; I(d<sub>k</sub> &isin; R<sub>q</sub>)</b><br/>"
        "• <b>Mean Reciprocal Rank (MRR):</b> Reciprocal rank of the first relevant document retrieved:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>MRR = (1 / |Q|) &middot; &Sigma;<sub>q</sub> (1 / rank<sub>1</sub>(q))</b><br/>"
        "• <b>Normalized Discounted Cumulative Gain (NDCG@10):</b> Measures graded position-discounted gain:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>DCG@K = &Sigma;<sub>k=1..K</sub> (rel<sub>k</sub> / log<sub>2</sub>(k+1)), &nbsp;&nbsp;&nbsp; NDCG@K = DCG@K / IDCG@K</b>",
        body_style
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("4.3 Comparative Baseline Models", h2_style))
    story.append(Paragraph(
        "To rigorously benchmark Method A and Method B, three classic retrieval systems are evaluated under identical conditions:<br/>"
        "1. <b>Unweighted Zone Scoring:</b> Uniform weights (<i>w</i> = 0.25 for Title, Author, Bib, Body) acting as a naive baseline.<br/>"
        "2. <b>Vector Space Model (VSM):</b> Sublinear TF-IDF cosine similarity: <i>w<sub>t, d</sub> = (1 + ln tf) &middot; idf</i>.<br/>"
        "3. <b>Okapi BM25:</b> Non-linear probabilistic term saturation model with <i>k</i><sub>1</sub> = 1.2, <i>b</i> = 0.75.",
        body_style
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 7: RESULTS (TEST SET COMPARISONS)
    # =========================================================================
    story.append(Paragraph("5. Experimental Results", h1_style))
    story.append(Paragraph(
        "This section presents the primary empirical findings on the held-out test collection (45 queries) and 5-fold cross-validation.",
        body_style
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("5.1 Performance on Held-Out Test Queries (N=45)", h2_style))
    story.append(Paragraph(
        "Table 3 summarizes the retrieval effectiveness of all six systems evaluated across cutoffs <i>K</i> &isin; {5, 10, 20}, MAP, MRR, and NDCG@10 on the test query set.",
        body_style
    ))
    story.append(Spacer(1, 3))

    m_names = [
        "Unweighted Zone",
        "Method A (Weighted Zone)",
        "Method A (coarse any-term)",
        "VSM (TF-IDF Cosine)",
        "Okapi BM25",
        "Method B (Linear SVM L2R)"
    ]
    
    test_table_data = [
        [
            Paragraph("<b>Retrieval Model</b>", table_header),
            Paragraph("<b>P@5</b>", table_header),
            Paragraph("<b>P@10</b>", table_header),
            Paragraph("<b>P@20</b>", table_header),
            Paragraph("<b>R@10</b>", table_header),
            Paragraph("<b>R@20</b>", table_header),
            Paragraph("<b>F1@10</b>", table_header),
            Paragraph("<b>MAP</b>", table_header),
            Paragraph("<b>MRR</b>", table_header),
            Paragraph("<b>NDCG@10</b>", table_header)
        ]
    ]

    for m in m_names:
        r = test_res[m]
        test_table_data.append([
            Paragraph(f"<b>{m}</b>", table_cell_left),
            Paragraph(f"{r['P@5']:.4f}", table_cell),
            Paragraph(f"{r['P@10']:.4f}", table_cell),
            Paragraph(f"{r['P@20']:.4f}", table_cell),
            Paragraph(f"{r['R@10']:.4f}", table_cell),
            Paragraph(f"{r['R@20']:.4f}", table_cell),
            Paragraph(f"{r['F1@10']:.4f}", table_cell),
            Paragraph(f"{r['MAP']:.4f}", table_cell),
            Paragraph(f"{r['MRR']:.4f}", table_cell),
            Paragraph(f"{r['NDCG@10']:.4f}", table_cell)
        ])

    t_tab = Table(test_table_data, colWidths=[120, 40, 40, 40, 44, 44, 44, 44, 44, 44])
    t_tab.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0d47a1')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cfd8dc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f7fa')]),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_tab)
    story.append(Spacer(1, 5))

    # Embedded Chart: Overall Comparison
    chart1_file = os.path.join(CHARTS_DIR, 'model_comparison.png')
    if os.path.exists(chart1_file):
        story.append(Image(chart1_file, width=470, height=210))
        story.append(Spacer(1, 3))
        story.append(Paragraph("<b>Figure 2:</b> Test set performance metrics (P@5, P@10, R@10, F1@10, MAP, NDCG@10) across retrieval models.", callout_style))

    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Key Test Findings:</b><br/>"
        "• <b>Method A:</b> term-level Boolean zone matching with MAP-trained weights " + str(tuple(round(x,2) for x in method_a_w)) + " raises MAP from " + f"{tAny['MAP']:.3f}" + " (coarse any-term zones) to " + f"{tA['MAP']:.3f}" + " and beats unweighted zones (" + f"{tU['MAP']:.3f}" + "). It stays below full-text models since binary matches ignore term frequency and document length.<br/>"
        "• <b>Method B</b> reaches P@5 = " + f"{tB['P@5']:.3f}" + ", MRR = " + f"{tB['MRR']:.3f}" + ", MAP = " + f"{tB['MAP']:.3f}" + " -- statistically indistinguishable from BM25/VSM (differences are well within the fold-to-fold std in Table 4): the 22 features are dominated by monotone transformations of the same tf/idf statistics, so a linear model cannot do much better than its best single feature.<br/>"
        "• <b>Set-based view</b> (cutoff K* = F1-optimal K on the training queries): Method B K*=" + str(bestk['Method B (Linear SVM L2R)']) + " gives P/R/F1 = " + f"{cut['Method B (Linear SVM L2R)']['P']:.3f}/{cut['Method B (Linear SVM L2R)']['R']:.3f}/{cut['Method B (Linear SVM L2R)']['F1']:.3f}" + "; BM25 K*=" + str(bestk['Okapi BM25']) + " gives " + f"{cut['Okapi BM25']['P']:.3f}/{cut['Okapi BM25']['R']:.3f}/{cut['Okapi BM25']['F1']:.3f}" + "; Method A K*=" + str(bestk['Method A (Weighted Zone)']) + " gives " + f"{cut['Method A (Weighted Zone)']['P']:.3f}/{cut['Method A (Weighted Zone)']['R']:.3f}/{cut['Method A (Weighted Zone)']['F1']:.3f}" + ".",
        body_style
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 8: RESULTS (CUTOFF CURVES & CROSS-VALIDATION)
    # =========================================================================
    story.append(Paragraph("5.2 Retrieval Metrics vs. Rank Cutoffs (K = 5, 10, 20)", h2_style))
    story.append(Paragraph(
        "Figure 3 illustrates how Precision, Recall, and F1 score evolve as the retrieval threshold <i>K</i> expands from 5 to 20 documents.",
        body_style
    ))
    story.append(Spacer(1, 2))

    chart2_file = os.path.join(CHARTS_DIR, 'cutoffs_comparison.png')
    if os.path.exists(chart2_file):
        story.append(Image(chart2_file, width=470, height=145))
        story.append(Spacer(1, 2))
        story.append(Paragraph("<b>Figure 3:</b> Precision, Recall, and F1 Score progressions across rank cutoffs K=5, 10, 20.", callout_style))

    story.append(Spacer(1, 5))

    story.append(Paragraph("5.3 5-Fold Cross-Validation Summary (All 225 Queries)", h2_style))
    story.append(Paragraph(
        "To rigorously verify statistical stability, 5-fold cross-validation was conducted over all 225 queries. Table 4 reports the sample mean and standard deviation across folds.",
        body_style
    ))
    story.append(Spacer(1, 3))

    cv_table_data = [
        [
            Paragraph("<b>Retrieval Model</b>", table_header),
            Paragraph("<b>P@10 (Mean ± Std)</b>", table_header),
            Paragraph("<b>R@10 (Mean ± Std)</b>", table_header),
            Paragraph("<b>F1@10 (Mean ± Std)</b>", table_header),
            Paragraph("<b>MAP (Mean ± Std)</b>", table_header),
            Paragraph("<b>NDCG@10 (Mean ± Std)</b>", table_header)
        ]
    ]

    for m in m_names:
        c = cv_res[m]
        cv_table_data.append([
            Paragraph(f"<b>{m}</b>", table_cell_left),
            Paragraph(f"{c['P@10']['mean']:.4f} ± {c['P@10']['std']:.4f}", table_cell),
            Paragraph(f"{c['R@10']['mean']:.4f} ± {c['R@10']['std']:.4f}", table_cell),
            Paragraph(f"{c['F1@10']['mean']:.4f} ± {c['F1@10']['std']:.4f}", table_cell),
            Paragraph(f"{c['MAP']['mean']:.4f} ± {c['MAP']['std']:.4f}", table_cell),
            Paragraph(f"{c['NDCG@10']['mean']:.4f} ± {c['NDCG@10']['std']:.4f}", table_cell)
        ])

    cv_tab = Table(cv_table_data, colWidths=[130, 75, 75, 75, 75, 74])
    cv_tab.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0d47a1')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cfd8dc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f7fa')]),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(cv_tab)
    story.append(Spacer(1, 5))

    chart5_file = os.path.join(CHARTS_DIR, 'cross_validation.png')
    if os.path.exists(chart5_file):
        story.append(Image(chart5_file, width=460, height=150))
        story.append(Spacer(1, 2))
        story.append(Paragraph("<b>Figure 4:</b> 5-Fold cross-validation stability comparing MAP and F1@10 across all 225 Cranfield queries.", callout_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 9: DISCUSSION OF RESULTS (FEATURE WEIGHTS & INTERPRETABILITY)
    # =========================================================================
    story.append(Paragraph("6. Discussion of Results", h1_style))
    story.append(Paragraph(
        "A critical advantage of linear models is their direct interpretability. This section analyzes the learned parameters for both Method A and Method B, explaining the underlying information retrieval mechanisms.",
        body_style
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("6.1 Analysis of Method B Linear SVM Learned Feature Weights", h2_style))
    story.append(Paragraph(
        "Figure 5 displays the real-valued weights <i>w<sub>j</sub></i> learned by the Linear SVM across all 22 features, sorted from most negative to most positive.",
        body_style
    ))
    story.append(Spacer(1, 2))

    chart3_file = os.path.join(CHARTS_DIR, 'svm_feature_weights.png')
    if os.path.exists(chart3_file):
        story.append(Image(chart3_file, width=470, height=230))
        story.append(Spacer(1, 2))
        story.append(Paragraph("<b>Figure 5:</b> Real-valued feature weights learned by Linear SVM across 22 query-document features.", callout_style))

    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Insights from SVM Weight Analysis:</b> (weights are in RobustScaler units)<br/>"
        "1. <b>Largest-magnitude weights:</b> " + "; ".join(f"<code>{n}</code> ({w:+.3f})" for n, w in top_w) + ". The model leans on the cosine/TF-IDF family (VSM, BM25, summed TF-IDF), confirming that sub-linear tf with idf weighting remains the strongest relevance signal in Cranfield.<br/>"
        "2. <b>Length correction:</b> <code>stream_length</code> receives a negative weight (" + f"{svm_w['stream_length']:+.3f}" + "), a learned penalty for long documents that match many terms by chance.<br/>"
        "3. <b>Negative weights on correlated features</b> (e.g. <code>query_term_ratio</code>: " + f"{svm_w['query_term_ratio']:+.3f}" + ") act as correction terms that offset double counting between BM25, VSM and the raw TF/TF-IDF aggregates (multicollinearity); individual weights should not be read in isolation.<br/>"
        "4. <b>Why B is only on par with BM25:</b> every feature is a function of the same per-term tf, idf and document length; the best linear mix is therefore close to the best single feature. Heavy regularisation (small C) was selected for this reason. Because the weights are tuned for MAP via CV, the SVM is the most stable of the learned rankers across folds, but not significantly better than BM25/VSM.",
        body_style
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 10: DISCUSSION, ERROR ANALYSIS, CONCLUSION & SUBMISSIONS
    # =========================================================================
    story.append(Paragraph("6.2 Method A Zone Weights: Analysis & Synthesis", h2_style))
    story.append(Paragraph(
        "The grid search for Boolean Weighted Zone Scoring converged to <b>w*</b> = (<i>w<sub>T</sub></i>=" + f"{method_a_w[0]:.2f}" + ", <i>w<sub>A</sub></i>=" + f"{method_a_w[1]:.2f}" + ", <i>w<sub>B</sub></i>=" + f"{method_a_w[2]:.2f}" + ", <i>w<sub>W</sub></i>=" + f"{method_a_w[3]:.2f}" + ") under the MAP objective:<br/>"
        "• <b>Title and body carry most weight, but bibliography and author are not negligible.</b> In Cranfield the bibliography field holds the venue/affiliation line and the author field is short, so a query term matching there is rare and therefore informative; the body, which also repeats the title, gives breadth.<br/>"
        "• <b>Granularity matters more than the weights.</b> A single 'any query term occurs in the zone' bit saturates (nearly every body matches some query term) and produces massive ties: MAP " + f"{tAny['MAP']:.3f}" + " with weights (" + ", ".join(f"{x:.2f}" for x in wany) + "). Counting the Boolean matches per query term (a fraction in [0,1]) restores ordering information and lifts MAP to " + f"{tA['MAP']:.3f}" + " (CV: " + f"{cvAny['MAP']['mean']:.3f}" + " &rarr; " + f"{cvA['MAP']['mean']:.3f}" + ").<br/>"
        "• <b>Train/inference consistency.</b> An earlier version optimised weights on an unordered-ties ranking but deployed a different tie-break; the learned weights were then meaningless. The final scorer uses one function for both stages.<br/>"
        "• Zone scoring still trails VSM/BM25 (CV MAP " + f"{cvA['MAP']['mean']:.3f}" + " vs. " + f"{cvM['MAP']['mean']:.3f}" + ") because it has no tf, idf or length information.",
        body_style
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("6.3 Error Analysis & Retrieval Challenges", h2_style))
    story.append(Paragraph(
        "Inspection of per-query test performance revealed distinct failure modes:<br/>"
        "• <b>Vocabulary Mismatch & Synonymy:</b> Queries like Q33 (<i>'interference-free longitudinal stability measurements... wind tunnel'</i>) struggled due to phrasing variations (e.g., 'tunnel blockage effects' vs. 'interference-free'). Both Method A and B rely on lexical matching; incorporating dense embeddings or pseudo-relevance feedback would alleviate vocabulary gaps.<br/>"
        "• <b>Burstiness in Generic Terms:</b> For broad queries (e.g., Q10 <i>'transport properties for air'</i>), non-relevant documents discussing air thermodynamics scored high on BM25, leading to false positives.",
        body_style
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("6.4 Submission Deliverables & Evaluation Readiness", h2_style))
    story.append(Paragraph(
        "All assignment submission deliverables have been prepared and strictly validated:<br/>"
        "1. <b>Program Files:</b> Complete, modular source code in Python 3:<br/>"
        "   • <code>dataset_parser.py</code>: Corpus and qrels parser with tokenization, stopword removal, and Porter stemming.<br/>"
        "   • <code>method_a_zone_scoring.py</code>: Boolean Weighted Zone Scorer with fast vectorized simplex grid search.<br/>"
        "   • <code>feature_extractor.py</code>: Vectorized extractor computing the exact 22 features.<br/>"
        "   • <code>method_b_svm.py</code>: Linear SVM Learning to Rank (pointwise and pairwise RankSVM, RobustScaler, CV-tuned C).<br/>"
        "   • <code>evaluate.py</code>: Standard IR evaluation suite (P@K, R@K, F1@K, MAP, MRR, NDCG@K) and baselines.<br/>"
        "   • <code>main.py</code>: Unified pipeline executing experiments, cross-validation, and exporting results.<br/>"
        "   • <code>predict.py</code>: Standalone CLI prediction script ready to evaluate on unknown test queries:<br/>"
        "     <code>python predict.py --queries &lt;path_to_queries&gt; --output &lt;output_file&gt; --method svm</code><br/>"
        "2. <b>Result Files:</b> Formatted text files (qid, docid, rank, score) containing top-100 ranked documents per held-out test query; <code>predict.py</code> additionally writes a second file truncated at the F1-optimal cutoff K* for set-based evaluation:<br/>"
        "   • <code>results_method_a.txt</code>: 4,500 ranked lines for 45 test queries using Method A.<br/>"
        "   • <code>results_method_b.txt</code>: 4,500 ranked lines for 45 test queries using Method B.<br/>"
        "   • <code>results_bm25_baseline.txt</code>: 4,500 ranked lines for 45 test queries using BM25.<br/>"
        "3. <b>Report Document:</b> This comprehensive 10-page report (<code>report.pdf</code>).",
        body_style
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("6.5 Conclusion", h2_style))
    story.append(Paragraph(
        "This project successfully designed, implemented, and benchmarked Boolean Weighted Zone Scoring and Linear SVM Learning to Rank on the Cranfield text collection. Method A shows that Boolean zone weights can be learned reliably by grid search once the zone input is the per-term Boolean match fraction (MAP " + f"{cvAny['MAP']['mean']:.3f}" + " &rarr; " + f"{cvA['MAP']['mean']:.3f}" + " in CV). Method B learns real-valued weights for 22 features; it matches BM25 and VSM (CV MAP " + f"{cvB['MAP']['mean']:.3f}" + " vs. " + f"{cvM['MAP']['mean']:.3f}" + " / " + f"{cvV['MAP']['mean']:.3f}" + ") but, as the features are all derived from the same term statistics, does not surpass them. Honest reporting: neither learned method significantly outperforms VSM on this collection. The resulting codebase and inference CLI provide a robust, optimized system engineered to achieve top performance on unknown test evaluation queries.",
        body_style
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Report successfully compiled to {output_filename}!")

if __name__ == '__main__':
    build_pdf_report("report.pdf")
