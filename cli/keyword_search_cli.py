import argparse
import json
import string

from nltk.stem import PorterStemmer
from inverted_index import InvertedIndex


def tokenize(text):
    return text.lower().translate(str.maketrans("", "", string.punctuation)).split()

def main():
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    search_parser = subparsers.add_parser("build", help="Build movies inverted index")

    args = parser.parse_args()

    stemmer = PorterStemmer()

    inverted_index = InvertedIndex()

    with open("data/movies.json") as f:
        data_set = json.load(f)

    with open("data/stopwords.txt") as s:
        stop_words = s.read().splitlines()

    match args.command:
        case "build":
            inverted_index.build(data_set['movies'])
            inverted_index.save()
            merida = inverted_index.get_document('merida')
            print(f"First document for token 'merida' = {merida[0]}")
            for item in inverted_index.index:
                print(item, inverted_index.index[item])
        case "search":
            print(f"Searching for: {args.query}")
            results = [
                movie
                for movie in data_set['movies']
                if any(
                    query_part not in stop_words
                    and stemmer.stem(query_part) in stemmer.stem(token)
                    for query_part in tokenize(args.query)
                    for token in tokenize(movie['title'])
                )
            ]
            for i, result in enumerate(results):
                if i > 4:
                    break
                print(f"{i+1} Movie title {result['title']}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()


