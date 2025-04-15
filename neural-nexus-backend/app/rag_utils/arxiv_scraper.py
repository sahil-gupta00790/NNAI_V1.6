# src/arxiv_scraper.py
import arxiv
import json
import os
from dotenv import load_dotenv

load_dotenv() # Loads .env file for potential API keys, though not used by arxiv lib

def scrape_arxiv(query="neural network architecture OR large language model", max_results=100, data_dir="data/research_papers"):
    """Scrapes arXiv based on a query and saves results to JSON files."""
    client = arxiv.Client(
        page_size=100,
        delay_seconds=5,
        num_retries=3
    )

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    papers_metadata = []
    download_count = 0
    save_dir = os.path.join(data_dir, "raw_pdfs") # Directory to save PDFs
    metadata_file = os.path.join(data_dir, "arxiv_metadata.json") # File for metadata

    os.makedirs(save_dir, exist_ok=True)
    print(f"Scraping arXiv for query: '{query}'...")

    try:
        results = list(client.results(search)) # Get all results first
        print(f"Found {len(results)} potential papers.")

        for i, result in enumerate(results):
            print(f"Processing paper {i+1}/{len(results)}: '{result.title}'")
            paper_meta = {
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "summary": result.summary,
                "published": result.published.isoformat(),
                "pdf_url": result.pdf_url,
                "arxiv_id": result.entry_id.split('/')[-1] # Extract ID
            }

            # Attempt to download PDF
            try:
                # Sanitize filename - replace invalid chars, limit length
                safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in result.title)
                safe_title = safe_title[:100] # Limit filename length
                pdf_filename = f"{paper_meta['arxiv_id']}_{safe_title}.pdf"
                output_path = os.path.join(save_dir, pdf_filename)

                # Download only if it doesn't exist
                if not os.path.exists(output_path):
                    result.download_pdf(dirpath=save_dir, filename=pdf_filename)
                    print(f"  Downloaded: {pdf_filename}")
                    download_count += 1
                else:
                    print(f"  Already exists: {pdf_filename}")

                paper_meta["local_pdf_path"] = output_path # Store path in metadata
                papers_metadata.append(paper_meta)

            except Exception as download_err:
                print(f"  Failed to download PDF for {paper_meta['arxiv_id']}: {download_err}")
                # Still add metadata even if PDF download fails
                paper_meta["local_pdf_path"] = None
                papers_metadata.append(paper_meta)

        # Save all metadata to a single JSON file
        os.makedirs(data_dir, exist_ok=True) # Ensure base data dir exists
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(papers_metadata, f, indent=4, ensure_ascii=False)
        print(f"\nScraping finished. Downloaded {download_count} new PDFs.")
        print(f"Metadata for {len(papers_metadata)} papers saved to {metadata_file}")
        return True

    except Exception as e:
        print(f"Scraping failed during search or processing: {str(e)}")
        # Save any metadata collected so far
        if papers_metadata:
             with open(metadata_file, "w", encoding="utf-8") as f:
                 json.dump(papers_metadata, f, indent=4, ensure_ascii=False)
             print(f"Saved partial metadata for {len(papers_metadata)} papers.")
        return False

if __name__ == "__main__":
    scrape_arxiv()
