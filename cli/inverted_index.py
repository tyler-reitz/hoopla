import os
import pickle
import string
import math

from collections import Counter, defaultdict
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

BM25_K1 = 1.5


def tokenize(text):
    return text.lower().translate(str.maketrans("", "", string.punctuation)).split()

def load(func):
    def wrapper(self, *args, **kwargs):
        self.load()
        return func(self, *args, **kwargs)
    return wrapper

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = defaultdict(list)
        self.trmfrq = defaultdict(Counter)

    @load
    def bm_25_tf(self, doc_id, term, k1=BM25_K1):
        tf = self.tf(doc_id, term)
        bm25tf = (tf * (k1 + 1)) / (tf + k1)
        return bm25tf

    @load
    def bm_25_idf(self, term):
        N = len(self.docmap)
        df = len(self.index[stemmer.stem(term)])
        return math.log(((N - df + 0.5) / (df + 0.5) + 1))

    def tfidf(self, doc_id, term):
        return self.tf(doc_id, term) * self.idf(term)

    def idf(self, term):
        return math.log((len(self.docmap) + 1) / (len(self.index[stemmer.stem(term)]) + 1)) 

    def tf(self, doc_id, term):
        return self.trmfrq[int(doc_id)][stemmer.stem(term)]

    def __add_document(self, doc_id, text):
        for token in tokenize(text):
            self.index[stemmer.stem(token)].add(doc_id)
            self.trmfrq[doc_id][stemmer.stem(token)]+=1

    def get_document(self, term):
        return sorted(list(self.index[stemmer.stem(term).lower()]))

    def build(self, json):
        for item in json:
            self.docmap[item['id']]=item
            self.__add_document(item['id'], f"{item['title']} {item['description']}")

    def save(self):
        os.makedirs("./cache", exist_ok=True)
        with open("./cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("./cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
        with open("./cache/trmfrq.pkl", "wb") as f:
            pickle.dump(self.trmfrq, f)

    def load(self):
        try:
            with open("./cache/index.pkl", "rb") as f:
                self.index = pickle.load(f)
            with open("./cache/docmap.pkl", "rb") as f:
                self.docmap = pickle.load(f)
            with open("./cache/trmfrq.pkl", "rb") as f:
                self.trmfrq = pickle.load(f)
        except FileNotFoundError as e:
            print(f"Couldn't open {e.filename}")
            raise e
