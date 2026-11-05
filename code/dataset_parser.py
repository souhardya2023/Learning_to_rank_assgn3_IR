"""
dataset_parser.py - Parser and Preprocessor for the Cranfield Collection.

Handles:
- Parsing cran.all.1400 (Title, Author, Bibliography, Text Body).
- Parsing queries from data/TEST/query.txt and cran/cran.qry.
- Parsing relevance judgments from cran/cranqrel and data/TEST/RES/*.txt.
- Preprocessing: tokenization, punctuation removal, stopword removal, Porter stemming.
- Vocabulary building, inverted index creation, and document statistics.
"""

import os
import re
import math
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer

# Standard 179 English stopwords (self-contained, no external download needed)
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it',
    "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
    'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by',
    'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
    'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
    'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o',
    're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
    "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
    "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
    'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
    "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

STEMMER = PorterStemmer()

def tokenize(text, stem=True, remove_stop=True):
    """
    Tokenizes raw text into normalized, filtered, and stemmed terms.
    """
    if not text:
        return []
    # Replace dashes, slashes, punctuation with space
    text = text.replace('-dash', ' ')
    tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
    
    result = []
    for token in tokens:
        # Keep numeric terms or alphabetic terms of length >= 2
        if len(token) < 2 and not token.isdigit():
            continue
        if remove_stop and token in STOPWORDS:
            continue
        if stem:
            token = STEMMER.stem(token)
        result.append(token)
    return result

class CranfieldCorpus:
    def __init__(self, cran_dir=None, data_dir=None):
        # Dynamically resolve cran_dir and data_dir based on project structure
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(script_dir)

        if cran_dir is None:
            candidates_cran = [
                os.path.join(root_dir, 'data', 'cran'),
                os.path.join('data', 'cran'),
                os.path.join(root_dir, 'cran'),
                'cran',
                os.path.join('..', 'data', 'cran'),
                os.path.join('..', 'cran')
            ]
            self.cran_dir = next((p for p in candidates_cran if os.path.isdir(p)), 'data/cran')
        else:
            self.cran_dir = cran_dir

        if data_dir is None:
            candidates_data = [
                os.path.join(root_dir, 'data'),
                'data',
                os.path.join('..', 'data')
            ]
            self.data_dir = next((p for p in candidates_data if os.path.isdir(p)), 'data')
        else:
            self.data_dir = data_dir

        self.docs = {}          # doc_id -> {'title': str, 'author': str, 'bib': str, 'body': str}
        self.doc_tokens = {}    # doc_id -> {'title': list, 'author': list, 'bib': list, 'body': list, 'all': list}
        self.queries = {}       # qid -> query_str
        self.query_tokens = {}  # qid -> list of tokens
        self.qrels = defaultdict(dict)  # qid -> {doc_id: relevance_grade}
        
        # Inverted index and stats over 'all' text
        self.inverted_index = defaultdict(dict)  # term -> {doc_id: tf}
        self.doc_lengths = {}                    # doc_id -> length (terms in all)
        self.df = defaultdict(int)               # term -> df
        self.idf = {}                            # term -> idf
        self.bm25_idf = {}                       # term -> bm25 idf
        self.avg_doc_len = 0.0
        
        self.load_documents()
        self.load_queries()
        self.load_qrels()
        self.build_index()

    def load_documents(self):
        """
        Parses cran/cran.all.1400 with .T, .A, .B, .W sections.
        """
        doc_path = os.path.join(self.cran_dir, 'cran.all.1400')
        if not os.path.exists(doc_path):
            raise FileNotFoundError(f"Missing Cranfield doc file: {doc_path}")

        with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        entries = content.split('.I ')
        for entry in entries[1:]:
            lines = entry.split('\n')
            doc_id = int(lines[0].strip())
            
            title, author, bib, body = [], [], [], []
            current_field = None
            
            for line in lines[1:]:
                if line.startswith('.'):
                    tag = line[:2]
                    if tag in ('.T', '.A', '.B', '.W'):
                        current_field = tag
                    else:
                        current_field = None
                else:
                    if current_field == '.T':
                        title.append(line)
                    elif current_field == '.A':
                        author.append(line)
                    elif current_field == '.B':
                        bib.append(line)
                    elif current_field == '.W':
                        body.append(line)
                        
            title_str = ' '.join(title).strip()
            author_str = ' '.join(author).strip()
            bib_str = ' '.join(bib).strip()
            body_str = ' '.join(body).strip()
            
            self.docs[doc_id] = {
                'title': title_str,
                'author': author_str,
                'bib': bib_str,
                'body': body_str
            }
            
            # Tokenize individual fields
            t_toks = tokenize(title_str)
            a_toks = tokenize(author_str)
            b_toks = tokenize(bib_str)
            w_toks = tokenize(body_str)
            all_toks = t_toks + a_toks + b_toks + w_toks
            
            self.doc_tokens[doc_id] = {
                'title': t_toks,
                'author': a_toks,
                'bib': b_toks,
                'body': w_toks,
                'all': all_toks
            }
            self.doc_lengths[doc_id] = len(all_toks)

    def load_queries(self):
        """
        Loads queries from data/TEST/query.txt if available, else cran/cran.qry.
        """
        test_query_path = os.path.join(self.data_dir, 'TEST', 'query.txt')
        if os.path.exists(test_query_path):
            with open(test_query_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        qid = int(parts[0])
                        qtext = parts[1].strip()
                        self.queries[qid] = qtext
                        self.query_tokens[qid] = tokenize(qtext)
        else:
            cran_qry_path = os.path.join(self.cran_dir, 'cran.qry')
            with open(cran_qry_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            entries = content.split('.I ')
            for idx, entry in enumerate(entries[1:], start=1):
                lines = entry.split('\n')
                text_lines = [l for l in lines[1:] if not l.startswith('.W')]
                qtext = ' '.join(text_lines).strip()
                self.queries[idx] = qtext
                self.query_tokens[idx] = tokenize(qtext)

    def load_qrels(self):
        """
        Loads relevance judgments from cran/cranqrel.
        Relevance codes:
        1 = Complete answer
        2 = High relevance
        3 = Useful
        4 = Minimum interest
        -1 = Source document (treated as non-relevant)
        """
        qrel_path = os.path.join(self.cran_dir, 'cranqrel')
        if os.path.exists(qrel_path):
            with open(qrel_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        qid = int(parts[0])
                        did = int(parts[1])
                        rel = int(parts[2])
                        self.qrels[qid][did] = rel
        else:
            res_dir = os.path.join(self.data_dir, 'TEST', 'RES')
            if os.path.exists(res_dir):
                for fname in os.listdir(res_dir):
                    if fname.endswith('.txt'):
                        qid = int(fname.replace('.txt', ''))
                        fpath = os.path.join(res_dir, fname)
                        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 2:
                                    did = int(parts[1])
                                    rel = int(parts[2]) if len(parts) >= 3 else 3
                                    self.qrels[qid][did] = rel

    def build_index(self):
        """
        Builds inverted index and calculates IDF & BM25 IDF across all documents.
        """
        num_docs = len(self.doc_tokens)
        total_len = sum(self.doc_lengths.values())
        self.avg_doc_len = total_len / num_docs if num_docs > 0 else 1.0

        for doc_id, fields in self.doc_tokens.items():
            counts = Counter(fields['all'])
            for term, count in counts.items():
                self.inverted_index[term][doc_id] = count
                self.df[term] += 1

        for term, doc_freq in self.df.items():
            # Standard smoothed IDF
            self.idf[term] = math.log((num_docs + 1) / (doc_freq + 0.5)) + 1.0
            # Robertson-Spärck Jones BM25 IDF
            self.bm25_idf[term] = math.log((num_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

    def is_relevant(self, qid, doc_id, strict=False):
        """
        Returns True if doc_id is relevant to qid.
        Standard: grades {1, 2, 3, 4} are relevant (Cleverdon scale).
        Strict: only grades {1, 2} are relevant.
        Grade -1 is non-relevant.
        """
        if qid not in self.qrels or doc_id not in self.qrels[qid]:
            return False
        grade = self.qrels[qid][doc_id]
        if strict:
            return grade in (1, 2)
        return grade in (1, 2, 3, 4)

    def get_relevant_docs(self, qid, strict=False):
        """
        Returns set of relevant doc IDs for query qid.
        """
        if qid not in self.qrels:
            return set()
        return {did for did in self.qrels[qid] if self.is_relevant(qid, did, strict=strict)}
