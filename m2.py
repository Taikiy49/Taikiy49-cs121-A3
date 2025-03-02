import os
import json
import math
import spacy
from collections import defaultdict
from bs4 import BeautifulSoup

# Load spaCy
print("Loading the spaCy model...")
nlp = spacy.load("en_core_web_sm")
print("spaCy model loaded.")

class SearchEngine:
    def __init__(self, dataset_path=None):
        # in m2 we only need to load the pre-built index, so dataset_path is not going to be used
        self.inverted_index = defaultdict(list)
        self.doc_freq = defaultdict(int)
        self.total_docs = 0
        self.index_file = "inverted_index.json"

    def load_index(self):
        if os.path.exists(self.index_file):
            with open(self.index_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
                self.inverted_index = defaultdict(list, data['index'])
                self.doc_freq = defaultdict(int, data['doc_freq'])
                self.total_docs = data['total_docs']
            print(f"Index loaded from {self.index_file}")
        else:
            print("No index file found. Please build the index first (run m1.py).")

    def tf_idf(self, term, doc_id):
        postings = dict(self.inverted_index.get(term, []))
        tf = postings.get(doc_id, 0)
        if tf == 0:
            return 0
        # Use smoothed IDF calculation
        idf = math.log((self.total_docs + 1) / (self.doc_freq.get(term, 1) + 1)) + 1
        return tf * idf

    def search(self, query):
        # Process the query in the same way as indexing (tokenization + lemmatization)
        query_terms = [token.lemma_.lower() for token in nlp(query) if token.is_alpha]
        doc_scores = defaultdict(float)
        doc_sets = []

        # Get the set of documents for each term
        for term in query_terms:
            docs = set(doc_id for doc_id, _ in self.inverted_index.get(term, []))
            doc_sets.append(docs)

        # Only consider documents that contain **all** query terms (Boolean AND)
        common_docs = set.intersection(*doc_sets) if doc_sets else set()

        # Rank the results using tf-idf
        for doc_id in common_docs:
            for term in query_terms:
                doc_scores[doc_id] += self.tf_idf(term, doc_id)

        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_docs[:5]

if __name__ == "__main__":
    # Create a SearchEngine instance and load the index
    engine = SearchEngine()
    engine.load_index()

    # Define the queries to test retrieval
    queries = [
        "Iftekhar Ahmed",
        "machine learning",
        "ACM",
        "master of software engineering"
    ]

    print("\n--- Search Results (Top 5 URLs for each query) ---")
    for query in queries:
        results = engine.search(query)
        print(f"\nQuery: '{query}'")
        if results:
            for rank, (url, score) in enumerate(results, start=1):
                print(f"{rank}. URL: {url} (Score: {score:.4f})")
        else:
            print("No results found.")
