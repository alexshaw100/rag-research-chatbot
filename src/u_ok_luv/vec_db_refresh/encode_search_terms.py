"""
For now all search terms will be encoded in Github using base64 encoding.

Before searching using the Arxiv command line tool, these search terms will
be decoded.
"""
import argparse
import base64
import pandas as pd

def encode_search_terms(csv: str):
    """
    Encodes the items in a single column csv using base64.
    """
    c = pd.read_csv(csv)
    c['Term'] = c['Term'].map(lambda t: base64.b64encode(t.encode('utf8')).decode('utf8'))
    c.to_csv(csv, index=False)
    return c['Term']

def decode_search_terms(csv: str):
    """
    Encodes the items in a single column csv using base64.
    """
    c = pd.read_csv(csv)
    c['Term'] = c['Term'].map(lambda t: base64.b64decode(t.encode('utf8')).decode('utf8'))
    c.to_csv(csv, index=False)
    return c['Term']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='The path to the csv file with the terms to be encoded or decoded.')
    parser.add_argument('--decode', action="store_true", default=False, help='Will decode the file if provided.')
    args = parser.parse_args()

    if args.decode:
        print(decode_search_terms(args.path))
    else:
        print(encode_search_terms(args.path))
