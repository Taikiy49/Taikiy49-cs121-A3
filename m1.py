import os
import json
import math
import nltk
from bs4 import BeautifulSoup
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt')

class SearchEngine:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.inverted_index = defaultdict(list)
        self.doc_freq = defaultdict(int)
        self.total_docs = 0
        self.stemmer = PorterStemmer()
        self.index_file = "inverted_index.json"

    def parse_html(self, html_content):
        """Extract text from HTML content and treat headings and bold text as important."""
        soup = BeautifulSoup(html_content, 'html.parser')
        important_tags = ['title', 'h1', 'h2', 'h3', 'b', 'strong']
        
        words = []
        for tag in important_tags:
            for element in soup.find_all(tag):
                words.extend(word_tokenize(element.get_text().lower()))

        # Add regular text
        words.extend(word_tokenize(soup.get_text().lower()))
        return words

    def build_index(self):
        """Creates the inverted index from the dataset."""
        for domain in os.listdir(self.dataset_path):
            domain_path = os.path.join(self.dataset_path, domain)
            if os.path.isdir(domain_path):
                for filename in os.listdir(domain_path):
                    file_path = os.path.join(domain_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        words = self.parse_html(data['content'])
                        doc_id = data['url']
                        
                        # Stemming and word frequency count
                        word_freq = defaultdict(int)
                        for word in words:
                            stemmed_word = self.stemmer.stem(word)
                            word_freq[stemmed_word] += 1
                        
                        # Add to inverted index
                        for word, freq in word_freq.items():
                            self.inverted_index[word].append((doc_id, freq))
                            self.doc_freq[word] += 1
                        
                        self.total_docs += 1
        
        self.save_index()

    def save_index(self):
        """Saves the inverted index to a JSON file."""
        with open(self.index_file, 'w', encoding='utf-8') as file:
            json.dump({
                'index': {key: val for key, val in self.inverted_index.items()},
                'doc_freq': dict(self.doc_freq),
                'total_docs': self.total_docs
            }, file)

    def load_index(self):
        """Loads the inverted index from a JSON file if it exists."""
        if os.path.exists(self.index_file):
            with open(self.index_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
                self.inverted_index = defaultdict(list, {key: val for key, val in data['index'].items()})
                self.doc_freq = defaultdict(int, data['doc_freq'])
                self.total_docs = data['total_docs']
        else:
            print("No index file found. Please run build_index() first.")

    def tf_idf(self, term, doc_id):
        """Computes the TF-IDF score for a term in a document."""
        postings = dict(self.inverted_index.get(term, []))
        tf = postings.get(doc_id, 0)
        if tf == 0:
            return 0
        idf = math.log((self.total_docs + 1) / (self.doc_freq.get(term, 1) + 1)) + 1
        return tf * idf

    def search(self, query):
        """Searches for documents matching the query using boolean AND with TF-IDF ranking."""
        query_terms = [self.stemmer.stem(word) for word in word_tokenize(query.lower())]
        doc_scores = defaultdict(float)
        doc_sets = []

        for term in query_terms:
            docs = set(doc_id for doc_id, _ in self.inverted_index.get(term, []))
            doc_sets.append(docs)

        # Find common documents containing all query terms (boolean AND)
        common_docs = set.intersection(*doc_sets) if doc_sets else set()

        # Rank the common documents using TF-IDF
        for doc_id in common_docs:
            for term in query_terms:
                doc_scores[doc_id] += self.tf_idf(term, doc_id)

        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_docs[:5]  # Return top 5 results

# this is my tests for now
dataset_path = 'test-directory' # test directory is erroring right now
search_engine = SearchEngine(dataset_path) # runn that
search_engine.build_index()  
search_engine.load_index()  

query = "machine learning" # gonna first test it with machine learnign but we still have to test all the other ones later
results = search_engine.search(query)

print("the top 5 results are: ", results)
