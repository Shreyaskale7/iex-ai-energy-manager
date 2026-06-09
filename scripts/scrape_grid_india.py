"""Scaffolding for Grid-India (POSOCO) Daily Report Scraper."""

import os
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import logging

try:
    import pdfplumber
    HAS_PLUMBER = True
except ImportError:
    HAS_PLUMBER = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("grid_india_scraper")

GRID_INDIA_URL = "https://grid-india.in/daily-reports/"
DOWNLOAD_DIR = Path("data/external/grid_reports")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

def fetch_latest_report_url() -> str | None:
    """Scrapes the Grid-India website to find the URL of the latest daily report."""
    logger.info(f"Fetching latest reports from {GRID_INDIA_URL}")
    try:
        response = requests.get(GRID_INDIA_URL, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # NOTE: This selector will need to be adjusted based on the actual live HTML structure of Grid-India
        # This is a scaffold looking for the first PDF link in the report table
        for link in soup.find_all('a'):
            href = link.get('href', '')
            if 'daily-report' in href.lower() and href.endswith('.pdf'):
                return href
                
        logger.warning("Could not find a valid PDF report link on the page.")
        return None
    except Exception as e:
        logger.error(f"Failed to fetch report URL: {e}")
        return None

def download_report(url: str) -> Path | None:
    """Downloads the PDF report to the external data directory."""
    try:
        filename = url.split("/")[-1]
        out_path = DOWNLOAD_DIR / filename
        
        if out_path.exists():
            logger.info(f"Report already downloaded: {out_path}")
            return out_path
            
        logger.info(f"Downloading report to {out_path}...")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(out_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        return out_path
    except Exception as e:
        logger.error(f"Failed to download report: {e}")
        return None

def extract_grid_frequency(pdf_path: Path):
    """Scaffold for extracting the grid frequency from the downloaded PDF."""
    if not HAS_PLUMBER:
        logger.warning("pdfplumber not installed. Skipping extraction.")
        return None
        
    logger.info(f"Extracting data from {pdf_path}")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Scaffold: The user will need to adjust the page number and table coordinates
            # based on the visual layout of the Grid-India PDF.
            first_page = pdf.pages[0]
            text = first_page.extract_text()
            
            # Example heuristic search
            for line in text.split('\n'):
                if 'Maximum Frequency' in line or 'Minimum Frequency' in line:
                    logger.info(f"Found Frequency Data: {line}")
                    
            # For exact table extraction, user should use:
            # tables = first_page.extract_tables()
            
    except Exception as e:
        logger.error(f"Extraction failed: {e}")

if __name__ == "__main__":
    report_url = fetch_latest_report_url()
    if report_url:
        pdf_path = download_report(report_url)
        if pdf_path:
            extract_grid_frequency(pdf_path)
    else:
        logger.info("Scraping finished with no new reports.")
