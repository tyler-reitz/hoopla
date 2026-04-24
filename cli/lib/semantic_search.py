import json
import os

import numpy as np

from sentence_transformers import SentenceTransformer


title_desc = lambda mov: f"{mov['title']}: {mov['description']}"

def load_documents(func):
    def wrapper(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc['id']]=doc
        return func(self, documents)
    return wrapper

class SemanticSearch:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embeddings(self, text):
        if not text.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return self.model.encode([text])[0]

    @load_documents
    def build_embeddings(self, documents):
        self.embeddings = self.model.encode(
                list(map(title_desc, documents)), show_progress_bar=True)
        np.save('cache/movie_embeddings.npy', self.embeddings)
        return self.embeddings

    @load_documents
    def load_or_create_embeddings(self, documents):
        if os.path.exists('cache/movie_embeddings.npy'):
            self.embeddings = np.asarray(np.load('cache/movie_embeddings.npy'))
            if len(self.embeddings) == len(self.documents or []):
                return self.embeddings
        else:
            return self.build_embeddings(documents)

def embed_query_text(query):
    ss = SemanticSearch()
    embedding = ss.generate_embeddings(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def verify_embeddings():
    print("Verifying embeddings")
    ss = SemanticSearch()
    with open("data/movies.json") as f:
        results = json.load(f)
    embeddings = ss.load_or_create_embeddings(results['movies'])
    print(f"Number of docs: {len(results['movies'])}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_text(text):
    ss = SemanticSearch()
    embeddings = ss.generate_embeddings(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embeddings[:3]}")
    print(f"Dimensions: {embeddings.shape[0]}")

def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")
