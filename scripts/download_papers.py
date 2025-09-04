import arxiv
import csv
import textwrap

# Read terms from file (assuming '|' delimiter)
with open("health_terms.txt", "r") as f:
    terms = f.read().strip().split('|')

# Create query string for PDF search (joined with OR)
health_query = " OR ".join(f'"{term}"' for term in terms)

print(health_query)

query = f'(artificial intelligence OR machine learning OR deep learning) AND ({health_query})'

# Initialize arXiv search
search = arxiv.Search(
    query=query,
    max_results=500,
    sort_by=arxiv.SortCriterion.SubmittedDate,
    sort_order=arxiv.SortOrder.Descending
)

# Extract data
data = []
client = arxiv.Client()

for result in client.results(search):
    # Wrap long summaries into smaller chunks (e.g. ~500 characters)
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

# Save to CSV
csv_filename = "ai_womens_health_arxiv_chunks.csv"
with open(csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
    fieldnames = ["paper_id", "title", "authors", "published", "chunk_id", "text_chunk", "pdf_url"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for row in data:
        writer.writerow(row)

print(f"Saved {len(data)} text chunks from arXiv papers to '{csv_filename}'")
