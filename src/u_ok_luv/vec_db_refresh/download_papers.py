import argparse
import arxiv
import csv
import os
import textwrap
import requests
import time
import re
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from importlib import resources
from datetime import datetime, timedelta, timezone
from pathlib import Path

from u_ok_luv.vec_db_refresh.encode_search_terms import read_search_terms

PACKAGE = "u_ok_luv.vec_db_refresh.search_terms"
SAVE_EXT = "csv"
STATUS_OK = 200


def load_all_search_terms(package: str = PACKAGE) -> dict[str, list[str]]:
    search_terms_dir = resources.files(package)
    text_files = [p for p in search_terms_dir.iterdir() if p.suffix == ".txt"]
    all_terms = {}
    for txt in text_files:
        all_terms[Path(txt).stem] = read_search_terms(txt)
    return all_terms


def save_collected_data(data: list[dict], csv_path: str):
    """
    Saves data collected from both Arxiv and MedRxiv.
    Ensures cells are single-line (no embedded newlines) for easier CSV consumers.
    """
    fieldnames: list[str] = []
    if data:
        keys = set()
        for row in data:
            keys.update(row.keys())
        preferred = [
            "paper_id",
            "title",
            "authors",
            "published",
            "chunk_id",
            "text_chunk",
            "pdf_url",
            "abstract",
            "doi",
        ]
        fieldnames = [k for k in preferred if k in keys] + [k for k in sorted(keys) if k not in preferred]

    def _normalize_cell(value) -> str:
        if value is None:
            return ""
        s = str(value)
        # Use regex to collapse all whitespace (including newlines/tabs) to single spaces
        s = re.sub(r"\s+", " ", s).strip()
        return s

    with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            # Ensure missing keys are written as empty strings and sanitize values
            writer.writerow({k: _normalize_cell(row.get(k, "")) for k in fieldnames})
    print(f"Saved {len(data)} text chunks from arXiv/medRxiv papers to '{csv_path}'")


def query_arxiv_papers(terms: list[str], max_results: int = 500, wrap_width: int = 500):
    health_query = " OR ".join(f'{term.lower()}' for term in terms)

    search = arxiv.Search(
        query=health_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    data = []
    client = arxiv.Client()
    
    for result in client.results(search):
        wrapped_summary = textwrap.wrap(result.summary.strip().replace("\n", " "), width=wrap_width)
        
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


def query_medrxiv_papers(terms: list[str],
                         server: str = "medrxiv",
                         days_back: int = 7,
                         max_results: int = 500,
                         page_size: int = 100,
                         base_url: str = "https://api.biorxiv.org",
                         user_agent: str | None = None,
                         retries: int = 3,
                         backoff_seconds: float = 1.5,
                         timeout_seconds: float = 15.0,
                         wrap_width: int = 500,
                         session: requests.Session | None = None):
    """
    Query medRxiv for preprints matching any of the terms in `terms`
    using the API, for the past `days_back` days (or fewer if limited),
    then process abstracts into chunks.

    Args:
        terms: list of search terms (strings).
        server: "medrxiv" (default) or "biorxiv" etc.
        days_back: how many past days to pull from.
        max_results: cap on number of results to return.

    Returns:
        A list of dicts with keys roughly: paper_id, title, authors, published,
        chunk_id, text_chunk, abstract (or summary), doi, etc.
    """
    # Prepare filters and date window (YYYY-MM-DD/YYYY-MM-DD per API docs)
    interval = _build_interval(days_back)

    collected = []
    cursor = 0

    while True:
        url = _build_medrxiv_url(base_url, server, interval, cursor)
        headers = _build_headers(user_agent)
        data = _request_json(session, url, headers, timeout_seconds, retries, backoff_seconds)
        if data is None:
            return collected

        items = data.get("collection", [])
        if not items:
            break

        for item in items:
            rows = _process_medrxiv_item(item, terms, wrap_width)
            for row in rows:
                collected.append(row)
                if len(collected) >= max_results:
                    break
            if len(collected) >= max_results:
                break

        # check if we need to paginate further
        cursor += len(items)
        collected_len, len_items = len(collected), len(items)
        if collected_len >= max_results or len_items < page_size:
            break

    return collected


def _build_interval(days_back: int) -> str:
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)
    return f"{start_date:%Y-%m-%d}/{end_date:%Y-%m-%d}"


def _build_medrxiv_url(base_url: str, server: str, interval: str, cursor: int) -> str:
    return f"{base_url}/details/{server}/{interval}/{cursor}/json"


def _build_headers(user_agent: str | None) -> dict:
    headers = {"Accept": "application/json"}
    if user_agent:
        headers["User-Agent"] = user_agent
    return headers


def _request_json(
    session: requests.Session | None,
    url: str,
    headers: dict,
    timeout_seconds: float,
    retries: int,
    backoff_seconds: float,
) -> dict | None:
    attempt = 0
    http = session or requests
    while True:
        resp = http.get(url, headers=headers, timeout=timeout_seconds)
        if resp.status_code == STATUS_OK:
            try:
                return resp.json()
            except Exception:
                print(f"Failed to decode JSON from {url}")
                return None
        attempt += 1
        if resp.status_code in (403, 429, 502, 503, 504) and attempt <= retries:
            sleep_for = backoff_seconds * attempt
            print(f"medRxiv API status {resp.status_code}; retrying in {sleep_for:.1f}s (attempt {attempt}/{retries})...")
            time.sleep(sleep_for)
            continue
        print(f"medRxiv API request failed with status {resp.status_code}, url {url}")
        return None


def _process_medrxiv_item(item: dict, lower_terms: list[str], wrap_width: int) -> list[dict]:
    abstract = (item.get("abstract") or "").strip()
    title = (item.get("title") or "").strip()
    authors_raw = (item.get("authors") or "").strip()
    authors_list = [a.strip() for a in authors_raw.split(";") if a.strip()]
    authors = ", ".join(authors_list)
    doi = item.get("doi", "") or ""
    date_str = item.get("date", "") or ""
    published = _safe_date_iso(date_str)

    ta_lower = (title + " " + abstract).lower()
    if lower_terms and not any(term in ta_lower for term in lower_terms):
        return []

    wrapped = textwrap.wrap(abstract.replace("\n", " "), width=wrap_width)
    rows: list[dict] = []
    for i, chunk in enumerate(wrapped):
        rows.append({
            "paper_id": doi or f"{item.get('server', '')}_{item.get('version', '')}",
            "title": title,
            "authors": authors,
            "published": published,
            "chunk_id": i,
            "text_chunk": chunk,
            "abstract": abstract,
            "doi": doi,
        })
    return rows


def _safe_date_iso(date_str: str) -> str:
    if not date_str:
        return ""
    try:
        return datetime.fromisoformat(date_str[:10]).date().isoformat()
    except Exception:
        return date_str


def build_session(user_agent: str | None = None, total_retries: int = 3, backoff_factor: float = 0.5) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[403, 429, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    if user_agent:
        session.headers.update({"User-Agent": user_agent})
    session.headers.update({"Accept": "application/json"})
    return session


def dedupe_rows(rows: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    out: list[dict] = []
    for r in rows:
        key = (
            (r.get("doi") or r.get("paper_id") or ""),
            r.get("published", ""),
            r.get("chunk_id"),
            r.get("text_chunk", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out

def download_papers():
    parser = argparse.ArgumentParser()
    parser.add_argument('--terms-dir', default=PACKAGE, help="Import path to the package containing base64-encoded search term .txt files.")
    parser.add_argument('--save-folder', default="ai_womens_health_paper_chunks/", help="Output folder for combined paper chunks (source-agnostic).")
    parser.add_argument('--max-results', type=int, default=500, help='Maximum number of chunks to collect per source.')
    parser.add_argument('--wrap-width', type=int, default=500, help='Character width for chunk wrapping of abstracts/summaries.')
    # medRxiv-specific controls
    parser.add_argument('--medrxiv-server', default="medrxiv", help='Preprint server for medRxiv API (e.g., "medrxiv", "biorxiv").')
    parser.add_argument('--medrxiv-days-back', type=int, default=7, help='Number of past days to query from medRxiv API.')
    parser.add_argument('--medrxiv-base-url', default="https://api.biorxiv.org", help='Base URL for the medRxiv/BioRxiv public API.')
    parser.add_argument('--medrxiv-user-agent', default="my-app/1.0 (mailto:you@example.com)", help='Custom User-Agent including contact info. Strongly recommended to avoid 403 errors.')
    parser.add_argument('--medrxiv-retries', type=int, default=3, help='Retries for transient HTTP errors (403/429/5xx).')
    parser.add_argument('--medrxiv-backoff-seconds', type=float, default=1.5, help='Base backoff seconds between retries.')
    parser.add_argument('--medrxiv-timeout-seconds', type=float, default=15.0, help='Timeout for medRxiv HTTP requests.')
    args = parser.parse_args()

    dir = (Path(os.getcwd()) / args.save_folder)
    dir.mkdir(exist_ok=True)

    all_terms = load_all_search_terms(args.terms_dir)

    # Build a shared HTTP session for medRxiv to reuse connections
    session = build_session(user_agent=args.medrxiv_user_agent, total_retries=args.medrxiv_retries, backoff_factor=max(0.1, args.medrxiv_backoff_seconds/3))

    for doc, terms in all_terms.items():
        save_file_name = dir / f"{doc}.{SAVE_EXT}"
        try:
            arxiv_data = query_arxiv_papers(terms, max_results=args.max_results, wrap_width=args.wrap_width)
            medrxiv_data = query_medrxiv_papers(
                terms,
                server=args.medrxiv_server,
                days_back=args.medrxiv_days_back,
                max_results=args.max_results,
                base_url=args.medrxiv_base_url,
                user_agent=args.medrxiv_user_agent,
                retries=args.medrxiv_retries,
                backoff_seconds=args.medrxiv_backoff_seconds,
                timeout_seconds=args.medrxiv_timeout_seconds,
                wrap_width=args.wrap_width,
                session=session,
            )
            combined = dedupe_rows((arxiv_data or []) + (medrxiv_data or []))
            save_collected_data(combined, save_file_name)
        except arxiv.UnexpectedEmptyPageError:
            print(f"Page unexpectedly empty error for topic: {doc}.")
        except Exception as e:
            print(f"Error downloading papers for topic '{doc}': {e}")
    

if __name__ == "__main__":
    download_papers()
