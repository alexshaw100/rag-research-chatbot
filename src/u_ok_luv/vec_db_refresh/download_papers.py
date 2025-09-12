import argparse
import arxiv
import csv
import os
import textwrap
import requests
import importlib.resources as resources
from pathlib import Path

from u_ok_luv.vec_db_refresh.encode_search_terms import read_search_terms

PACKAGE = "u_ok_luv.vec_db_refresh.search_terms"
SAVE_EXT = "csv"


def load_all_search_terms(package: str = PACKAGE) -> dict[str, list[str]]:
    search_terms_dir = resources.files(package)
    text_files = [p for p in search_terms_dir.iterdir() if p.suffix == ".txt"]
    all_terms = {}
    for txt in text_files:
        all_terms[Path(txt).stem] = read_search_terms(txt)
    return all_terms


def save_collected_data(data: str, csv_path: str):
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ["paper_id", "title", "authors", "published", "chunk_id", "text_chunk", "pdf_url"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    print(f"Saved {len(data)} text chunks from arXiv papers to '{csv_path}'")


def query_arxiv_papers(terms: list[str]):
    health_query = " OR ".join(f'{term.lower()}' for term in terms)
    # query = f'(artificial intelligence OR machine learning OR deep learning) AND ({health_query})'

    search = arxiv.Search(
        query=health_query,
        max_results=500,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    data = []
    client = arxiv.Client()
    
    for result in client.results(search):
        wrapped_summary = textwrap.wrap(result.summary.strip().replace("\n", " "), width=500)
        
        for i, chunk in enumerate(wrapped_summary):
            data.append({
                "paper_id": result.entry_id,
                "title": result.title,
                "authors": ", ".join(author.name for author in result.authors),
                "published": result.published.strftime("%Y-%m-%d"),
                "chunk_id": i,
                "text_chunk": chunk,
                "pdf_url": result.pdf_url
            })
    return data

def download_papers():
    parser = argparse.ArgumentParser()
    parser.add_argument('--terms-dir', default=PACKAGE, help="The directory containing txt files with search terms.")
    parser.add_argument('--save-folder', default="ai_womens_health_arxiv_chunks/", help="The directory containing txt files with search terms.")
    args = parser.parse_args()

    dir = (Path(os.getcwd()) / args.save_folder)
    dir.mkdir(exist_ok=True)

    all_terms = load_all_search_terms(args.terms_dir)

    for doc, terms in all_terms.items():
        save_file_name = dir / f"{doc}.{SAVE_EXT}"
        try:
            paper_data = query_arxiv_papers(terms)
            save_collected_data(paper_data, save_file_name)
        except arxiv.UnexpectedEmptyPageError as e:
            print(f"Page unexpectedly empty error for topic: {doc}.")
    

if __name__ == "__main__":
    download_papers()
