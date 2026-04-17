import os
import pickle
import string
import math

from collections import Counter, defaultdict
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

BM25_K1 = 1.5
BM25_B = 0.75


def tokenize(text):
    return text.lower().translate(str.maketrans("", "", string.punctuation)).split()

def load(func):
    def wrapper(self, *args, **kwargs):
        if not self.index or not self.docmap or not self.trmfrq or not self.doclens:
            self.load()
        return func(self, *args, **kwargs)
    return wrapper

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = defaultdict(list)
        self.trmfrq = defaultdict(Counter)
        self.doclens = defaultdict(int)

    def __avg_doc_len(self):
        return sum(self.doclens.values()) / len(self.doclens) if self.doclens else 0

    @load
    def bm_25_search(self, term, limit):
        scores = defaultdict(float)
        for token in tokenize(term):
            for doc_id in self.index[stemmer.stem(token)]:
                    scores[doc_id] += self.bm_25(doc_id, stemmer.stem(token))
        top_n = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit])
        return top_n

    @load
    def bm_25(self, doc_id, term):
        return self.bm_25_tf(doc_id, stemmer.stem(term)) * self.bm_25_idf(stemmer.stem(term))

    @load
    def bm_25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.tf(doc_id, term)
        lengh_norm = 1 - b + b * (self.doclens[doc_id] / self.__avg_doc_len())
        bm25tf = (tf * (k1 + 1)) / (tf + k1 * lengh_norm)
        return bm25tf

    @load
    def bm_25_idf(self, term):
        N = len(self.docmap)
        df = len(self.index[stemmer.stem(term)])
        return math.log(((N - df + 0.5) / (df + 0.5) + 1))

    @load
    def tfidf(self, doc_id, term):
        return self.tf(doc_id, term) * self.idf(term)

    @load
    def idf(self, term):
        return math.log((len(self.docmap) + 1) / (len(self.index[stemmer.stem(term)]) + 1)) 

    @load
    def tf(self, doc_id, term):
        return self.trmfrq[doc_id][stemmer.stem(term)]

    def __add_document(self, doc_id, text):
        tokens = tokenize(text)
        for token in tokens:
            self.index[stemmer.stem(token)].add(doc_id)
            self.trmfrq[doc_id][stemmer.stem(token)]+=1
            self.doclens[doc_id]=len([stemmer.stem(token) for token in tokens])

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
        with open("./cache/doclens.pkl", "wb") as f:
            pickle.dump(self.doclens, f)

    def load(self):
        try:
            with open("./cache/index.pkl", "rb") as f:
                self.index = pickle.load(f)
            with open("./cache/docmap.pkl", "rb") as f:
                self.docmap = pickle.load(f)
            with open("./cache/trmfrq.pkl", "rb") as f:
                self.trmfrq = pickle.load(f)
            with open("./cache/doclens.pkl", "rb") as f:
                self.doclens = pickle.load(f)
        except FileNotFoundError as e:
            print(f"Couldn't open {e.filename}")
            raise e
