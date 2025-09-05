import argparse
import arxiv
import csv
import importlib.resources as resources
import textwrap

from u_ok_luv.vec_db_refresh.encode_search_terms import read_search_terms

PACKAGE = "u_ok_luv.vec_db_refresh.search_terms"


def load_all_search_terms(package: str = PACKAGE):
    search_terms_dir = resources.files(package)
    text_files = [p for p in search_terms_dir.iterdir() if p.suffix == ".txt"]
    all_terms = []
    for txt in text_files:
        terms = read_search_terms(txt)
        all_terms.extend(terms)
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
    health_query = " OR ".join(f'"{term}"' for term in terms)
    query = f'(artificial intelligence OR machine learning OR deep learning) AND ({health_query})'

    search = arxiv.Search(
        query=query,
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
    parser.add_argument('--save-file', default="ai_womens_health_arxiv_chunks.csv", help="The directory containing txt files with search terms.")
    args = parser.parse_args()

    all_terms = load_all_search_terms(args.terms_dir)

    save_collected_data(query_arxiv_papers(all_terms), args.save_file)

if __name__ == "__main__":
    download_papers()
