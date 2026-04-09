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

    subparsers.add_parser("build", help="Build movies inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency per document")
    tf_parser.add_argument("doc_id", type=str, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Search term")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency")
    idf_parser.add_argument("term", type=str, help="Search term")

    args = parser.parse_args()

    # stemmer = PorterStemmer()

    inverted_index = InvertedIndex()

    with open("data/movies.json") as f:
        data_set = json.load(f)

    # with open("data/stopwords.txt") as s:
    #     stop_words = s.read().splitlines()

    match args.command:
        case "idf":
            inverted_index.load()
            idf = inverted_index.idf(args.term)
            print(f"Inverse document frequency of {args.term}: {idf:.2f}")
        case "tf":
            inverted_index.load()
            trmfrq = inverted_index.tf(args.doc_id, tokenize(args.term)[0])
            print(trmfrq)
        case "build":
            inverted_index.build(data_set['movies'])
            inverted_index.save()
            for item in inverted_index.index:
                print(item, inverted_index.index[item])
        case "search":
            print(f"Searching for: {args.query}")
            try:
                inverted_index.load()
                results = [
                    inverted_index.docmap[doc]
                    for token in tokenize(args.query)
                    for doc in inverted_index.get_document(token)
                ]
                for i, result in enumerate(results):
                    if i > 4:
                        break
                    print(f"{i+1} {result['title']} {result['id']}")
            except:
                print("Coudl not load index")
                exit(1)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()


