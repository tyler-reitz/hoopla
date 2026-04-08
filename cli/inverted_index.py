import os
import pickle
import string

from collections import defaultdict
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()


def tokenize(text):
    return text.lower().translate(str.maketrans("", "", string.punctuation)).split()

class InvertedIndex:
    def __init__(self):
        # map tokens to sets of doc IDs
        self.index = defaultdict(set)
        # map doc IDs to docs
        self.docmap = defaultdict(list)

    def __add_document(self, doc_id, text):
        for token in tokenize(text):
            self.index[stemmer.stem(token)].add(doc_id)

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

    def load(self):
        try:
            with open("./cache/index.pkl", "rb") as f:
                self.index = pickle.load(f)
            with open("./cache/docmap.pkl", "rb") as ff:
                self.docmap = pickle.load(ff)
        except FileNotFoundError as e:
            print(f"Couldn't open {e.filename}")
            raise e

