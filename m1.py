import os
import json
import math
import spacy
from bs4 import BeautifulSoup
from collections import defaultdict
import sys
# fuk nltk we're using spacy due to loading issues
# uhmm importing spaCy so we can tokenize and lemmatize instead of using nltk
print('loading the spacy model')
nlp = spacy.load("en_core_web_sm")
print('spacy model was loaded :)')

class SearchEngine:
    # imma set up the basic stuff here like the index, doc frequency, and file names
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.inverted_index = defaultdict(list)
        self.doc_freq = defaultdict(int)
        self.total_docs = 0
        self.index_file = "inverted_index.json"

    # parsing the HTML and trying to grab important words like titles and headings
    def parse_html(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        important_tags = ['title', 'h1', 'h2', 'h3', 'b', 'strong']

        words = []
        for tag in important_tags:
            for element in soup.find_all(tag):
                # uhmm extracting the text and tokenizing + lemmatizing with spaCy
                words.extend([token.lemma_.lower() for token in nlp(element.get_text()) if token.is_alpha])

        # gonna add the rest of the text from the page too
        words.extend([token.lemma_.lower() for token in nlp(soup.get_text()) if token.is_alpha])
        return words

    # imma loop through all files in the dataset and build the inverted index
    def build_index(self):
        for domain in os.listdir(self.dataset_path):
            domain_path = os.path.join(self.dataset_path, domain)
            if os.path.isdir(domain_path):
                for filename in os.listdir(domain_path):
                    file_path = os.path.join(domain_path, filename)

                    print(f"processing file: {file_path}")
                    with open(file_path, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        words = self.parse_html(data['content'])
                        doc_id = data['url']

                        # gonna count term frequencies so we can do tf-idf later
                        word_freq = defaultdict(int)
                        for word in words:
                            word_freq[word] += 1

                        # okay now storing this in the inverted index
                        for word, freq in word_freq.items():
                            self.inverted_index[word].append((doc_id, freq))
                            self.doc_freq[word] += 1
                        
                        self.total_docs += 1
        
        # uhmm saving the index so we don't have to do this every time
        self.save_index()

    # imma save the index to a JSON file
    def save_index(self):
        with open(self.index_file, 'w', encoding='utf-8') as file:
            json.dump({
                'index': {key: val for key, val in self.inverted_index.items()},
                'doc_freq': dict(self.doc_freq),
                'total_docs': self.total_docs
            }, file)

    # this function loads the index from a file so we don’t have to rebuild it
    def load_index(self):
        if os.path.exists(self.index_file):
            with open(self.index_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
                self.inverted_index = defaultdict(list, {key: val for key, val in data['index'].items()})
                self.doc_freq = defaultdict(int, data['doc_freq'])
                self.total_docs = data['total_docs']
        else:
            print("No index file found. Please run build_index() first.")

    # imma do the tf-idf calculation here for ranking later
    def tf_idf(self, term, doc_id):
        postings = dict(self.inverted_index.get(term, []))
        tf = postings.get(doc_id, 0)
        if tf == 0:
            return 0
        idf = math.log((self.total_docs + 1) / (self.doc_freq.get(term, 1) + 1)) + 1
        return tf * idf

    # okay now we do the actual searching
    def search(self, query):
        # imma process the query just like we did for indexing
        query_terms = [token.lemma_.lower() for token in nlp(query) if token.is_alpha]
        doc_scores = defaultdict(float)
        doc_sets = []

        for term in query_terms:
            docs = set(doc_id for doc_id, _ in self.inverted_index.get(term, []))
            doc_sets.append(docs)

        # uhmm only return documents that have **all** the query terms (boolean AND)
        common_docs = set.intersection(*doc_sets) if doc_sets else set()

        # okay now let's rank the results using tf-idf
        for doc_id in common_docs:
            for term in query_terms:
                doc_scores[doc_id] += self.tf_idf(term, doc_id)

        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_docs[:5]  # this will return the top 5 results
    
    def read_file(self,file_input):
        file_size = os.path.getsize('inverted_index.json')
        stop_words = ('a','about','above','after','again','against','all','am','an',"and","any","are","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can't","cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few","for","from","further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such","than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't","you","you'd","you'll","you're","you've", "your",'yours','yourself','yourselves')
        with open(file_input, 'r') as file:
            data = json.load(file)
        total_docs = data['total_docs']
        data_list_words = list(data['doc_freq'])
        count = 0
        for words in data_list_words:
            if(words not in stop_words):
                count += 1
        print('Total Number of Docs:',total_docs)
        print("Number of Unique words:",count)
        print("The size of the file is: ", file_size/1024, "KB")


# okay now let's test this
dataset_path = 'test-directory'  # gonna make sure this directory actually exists
search_engine = SearchEngine(dataset_path)  
search_engine.build_index()  # gonna build the index first
search_engine.load_index()   # now we load it


query = "machine learning"  # first test query, gotta test more later
results = search_engine.search(query)
 


print("the top 5 results are: ", results)
search_engine.read_file('inverted_index.json') 