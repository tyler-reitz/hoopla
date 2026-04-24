#!/usr/bin/env python3

import argparse

from lib.semantic_search import embed_query_text, verify_model, embed_text, verify_embeddings

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available command")

    subparsers.add_parser("verify", help="Verify Semantic Search Model is loaded")
    subparsers.add_parser("verify_embeddings", help="Verify Semantic Search embeddings")

    embed_parser = subparsers.add_parser("embed_text", help="Generate an embedding for a single text")
    embed_parser.add_argument("text", type=str, help="Text to embed")

    embedqry_parser = subparsers.add_parser("embedquery", help="Generate an embedding for a query")
    embedqry_parser.add_argument("query", type=str, help="Query to embed")

    args = parser.parse_args()

    match args.command:
        case "embedquery":
            embed_query_text(args.query)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case "verify":
            verify_model()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
