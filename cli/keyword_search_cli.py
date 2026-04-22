import argparse
import json
import string

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
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Search term")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency")
    idf_parser.add_argument("term", type=str, help="Search term")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get tf-idf for a term")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Search term")

    bm25idf_parser = subparsers.add_parser("bm25idf", help="Search movies using BM25IDF")
    bm25idf_parser.add_argument("term", type=str, help="Search query")

    bm25tf_parser = subparsers.add_parser("bm25tf", help="Search movies using BM25TF")
    bm25tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25tf_parser.add_argument("term", type=str, help="Search term")
    bm25tf_parser.add_argument("k1", type=float, nargs='?', default=1.5, help="Tunable BM25 K1")
    bm25tf_parser.add_argument("b", type=float, nargs='?', default=0.75, help="Tunable BM25 B")

    bm25_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25_parser.add_argument("query", type=str, help="Query to search for")
    bm25_parser.add_argument("-n", "--limit", type=int, help="Limit to n results", default=5)

    args = parser.parse_args()

    inverted_index = InvertedIndex()

    with open("data/movies.json") as f:
        data_set = json.load(f)

    match args.command:
        case "bm25search":
            matches = inverted_index.bm_25_search(args.query, args.limit)
            for n, doc_id in enumerate(matches, 1):
                doc = inverted_index.docmap[doc_id]
                print(f"{n}. ({doc_id}) {doc['title']} - Score {matches[doc_id]:.2f}")
        case "bm25tf":
            bm25tf = inverted_index.bm_25_tf(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case "bm25idf":
            bm25idf = inverted_index.bm_25_idf(args.term)
            print(f"Bm25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "tfidf":
            inverted_index.load()
            tfidf = inverted_index.tfidf(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tfidf:.2f}")
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
