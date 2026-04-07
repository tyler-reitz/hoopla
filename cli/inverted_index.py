import os
import pickle
import string

from collections import defaultdict


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
            self.index[token].add(doc_id)

    def get_document(self, term):
        return list(self.index[term.lower()])

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
        pass
