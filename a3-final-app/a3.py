from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import math
import spacy
import requests
from collections import defaultdict
from bs4 import BeautifulSoup
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer

import nltk
nltk.download('punkt')

# Load spaCy model
print("Loading the spaCy model...")
nlp = spacy.load("en_core_web_sm")
print("spaCy model loaded.")

app = Flask(__name__)
CORS(app)

class SearchEngine:
    def __init__(self):
        self.inverted_index = defaultdict(list)
        self.doc_freq = defaultdict(int)
        self.total_docs = 0
        self.index_file = "inverted_index.json"
        self.load_index()

    def load_index(self):
        if os.path.exists(self.index_file):
            with open(self.index_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
                self.inverted_index = defaultdict(list, data['index'])
                self.doc_freq = defaultdict(int, data['doc_freq'])
                self.total_docs = data['total_docs']
            print(f"Index loaded from {self.index_file}")
        else:
            print("No index file found. Please build the index first.")

    def tf_idf(self, term, doc_id):
        postings = dict(self.inverted_index.get(term, []))
        tf = postings.get(doc_id, 0)
        if tf == 0:
            return 0
        idf = math.log((self.total_docs + 1) / (self.doc_freq.get(term, 1) + 1)) + 1
        return tf * idf

    def search(self, query):
        query_terms = [token.lemma_.lower() for token in nlp(query) if token.is_alpha]
        doc_scores = defaultdict(float)
        doc_sets = [set(doc_id for doc_id, _ in self.inverted_index.get(term, [])) for term in query_terms]
        common_docs = set.intersection(*doc_sets) if doc_sets else set()

        for doc_id in common_docs:
            for term in query_terms:
                doc_scores[doc_id] += self.tf_idf(term, doc_id)

        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [{"url": doc_id, "score": score} for doc_id, score in ranked_docs[:5]]

search_engine = SearchEngine()

@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400

    results = search_engine.search(query)
    return jsonify({"results": results})


import re

def summarize_url(url):
    """Fetch and summarize the content of a webpage without using sumy."""
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract paragraphs and clean text
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text() for p in paragraphs)
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces

        # Split into sentences
        sentences = text.split('. ')
        summary = ". ".join(sentences[:3])  # Take the first 3 sentences

        return summary if summary else "No relevant summary available."
    except Exception as e:
        return f"Error summarizing content: {str(e)}"


    
@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.json
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "URL cannot be empty."}), 400

    summary = summarize_url(url)
    return jsonify({"summary": summary})

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Prevents pickle issues
    app.run(debug=True, port=5000)
