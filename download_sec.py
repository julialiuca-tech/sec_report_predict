# -*- coding: utf-8 -*-
"""
Enhanced script to discover and download ALL available SEC financial statement datasets.

This script:
1. Builds quarterly dataset candidates from 2011q1 through the current calendar quarter
2. Confirms which datasets are published on sec.gov
3. Downloads and extracts all available datasets
4. Skips already downloaded datasets

Created on Fri Aug 12 10:19:50 2024
Enhanced to discover all available datasets

@author: U.S. Securities and Exchange Commission.
Enhanced by: AI Assistant
"""

import requests
import zipfile
import os
from datetime import date
from io import BytesIO
from urllib.parse import urljoin, urlparse
import time
from pathlib import Path

from config import DATA_BASE_DIR

SEC_DATASETS_BASE_URL = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/"
SEC_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 (Contact: your-email@example.com)'
}
# Pre-2011 filings are sparse/unreliable while companies adopted the mandatory format.
SEC_DATA_START_YEAR = 2011


def iter_quarter_labels(start_year=SEC_DATA_START_YEAR, end_date=None):
    """
    Yield quarterly dataset labels from start_year q1 through the current calendar quarter.

    Parameters
    ----------
    start_year : int
        First year to include (default 2011; earlier SEC datasets are less dependable).
    end_date : date, optional
        Reference date for the newest calendar quarter. Defaults to today.

    Yields
    ------
    str
        Dataset labels such as '2011q1', '2026q1'.
    """
    if end_date is None:
        end_date = date.today()

    end_year = end_date.year
    end_quarter = (end_date.month - 1) // 3 + 1

    for year in range(start_year, end_year + 1):
        last_quarter = end_quarter if year == end_year else 4
        for quarter in range(1, last_quarter + 1):
            yield f"{year}q{quarter}"


def url_exists(url, headers=None, timeout=30):
    """Return True if the remote URL responds with HTTP 200."""
    if headers is None:
        headers = SEC_HEADERS
    try:
        response = requests.head(url, headers=headers, timeout=timeout, allow_redirects=True)
        if response.status_code == 200:
            return True
        # Some servers reject HEAD; fall back to a ranged GET.
        if response.status_code in (403, 405):
            response = requests.get(
                url, headers=headers, timeout=timeout, stream=True, allow_redirects=True
            )
            exists = response.status_code == 200
            response.close()
            return exists
        return False
    except requests.RequestException:
        return False


def get_available_datasets(base_url=SEC_DATASETS_BASE_URL, start_year=SEC_DATA_START_YEAR):
    """
    Discover available SEC quarterly datasets from start_year through the newest published quarter.

    The SEC directory listing is often unavailable, so this builds candidate quarterly URLs
    (2011q1 .. current calendar quarter) and keeps those that exist on the server.

    Parameters
    ----------
    base_url : str
        Base URL of the SEC financial statement datasets
    start_year : int
        First year to include

    Returns
    -------
    list
        List of discovered dataset URLs
    """
    print(f"Discovering available datasets from {base_url}...")
    print(f"Scanning quarterly datasets from {start_year}q1 through the current quarter...")

    candidates = [urljoin(base_url, f"{label}.zip") for label in iter_quarter_labels(start_year)]
    dataset_urls = []

    for i, url in enumerate(candidates, 1):
        label = get_dataset_name_from_url(url)
        if url_exists(url):
            dataset_urls.append(url)
            print(f"  [{i}/{len(candidates)}] {label}: available")
        else:
            print(f"  [{i}/{len(candidates)}] {label}: not published yet / missing")
        # Be polite to sec.gov while probing.
        time.sleep(0.2)

    if not dataset_urls:
        print("No datasets confirmed via HEAD checks. Falling back to candidate URL list...")
        return candidates

    print(f"Discovered {len(dataset_urls)} available quarterly datasets "
          f"({get_dataset_name_from_url(dataset_urls[0])} .. "
          f"{get_dataset_name_from_url(dataset_urls[-1])}).")
    return dataset_urls

def download_and_unzip(url, extract_to='.', max_retries=3):
    """
    Downloads a ZIP file from a URL and extracts its contents with retry logic
    
    Parameters
    ----------
    url : str
        URL pointing to the ZIP file.
    extract_to : str, optional
        Directory path where the contents will be extracted.
    max_retries : int, optional
        Maximum number of retry attempts for failed downloads.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    for attempt in range(max_retries):
        try:
            print(f"Downloading ZIP file from {url} (attempt {attempt + 1}/{max_retries})...")
            response = requests.get(url, headers=SEC_HEADERS, timeout=300)  # 5 minute timeout
            
            # Check if file exists (404 means file doesn't exist)
            if response.status_code == 404:
                print(f"Dataset not available: {url}")
                return False
                
            response.raise_for_status()
            
            # Create a ZipFile object from the bytes of the ZIP file
            zip_file = zipfile.ZipFile(BytesIO(response.content))
            
            # Extract the contents of the Zip file
            print(f"Extracting the contents to {extract_to}...")
            zip_file.extractall(path=extract_to)
            zip_file.close()
            print("Extraction complete.")
            return True
            
        except requests.RequestException as e:
            print(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"Failed to download after {max_retries} attempts: {url}")
                return False
        except Exception as e:
            print(f"Unexpected error processing {url}: {e}")
            return False

def get_dataset_name_from_url(url):
    """
    Extracts the dataset name from the URL
    
    Parameters
    ----------
    url : str
        URL of the dataset
        
    Returns
    -------
    str
        Dataset name (e.g., '2022q1', '2023')
    """
    filename = os.path.basename(urlparse(url).path)
    return filename.replace('.zip', '')

def sort_datasets_by_recency(dataset_urls):
    """
    Sorts datasets by recency, with most recent first
    
    Parameters
    ----------
    dataset_urls : list
        List of dataset URLs
        
    Returns
    -------
    list
        Sorted list of dataset URLs (most recent first)
    """
    def get_sort_key(url):
        """Extract sortable key from URL"""
        dataset_name = get_dataset_name_from_url(url)
        
        # Handle quarterly datasets (e.g., 2024q2)
        if 'q' in dataset_name:
            year, quarter = dataset_name.split('q')
            return (int(year), int(quarter), 0)  # 0 for quarterly
        
        # Handle annual datasets (e.g., 2024)
        elif dataset_name.isdigit():
            return (int(dataset_name), 0, 1)  # 1 for annual
        
        # Handle other patterns (e.g., 202412 for monthly)
        elif len(dataset_name) == 6 and dataset_name.isdigit():
            year = int(dataset_name[:4])
            month = int(dataset_name[4:6])
            return (year, month, 2)  # 2 for monthly
        
        # Default fallback
        return (0, 0, 3)
    
    # Sort by recency (most recent first)
    sorted_datasets = sorted(dataset_urls, key=get_sort_key, reverse=True)

    print(f"Datasets sorted by recency (most recent first), {len(sorted_datasets)} total:")
    preview = 10
    for i, url in enumerate(sorted_datasets[:preview], 1):
        dataset_name = get_dataset_name_from_url(url)
        print(f"  {i:2d}. {dataset_name}")
    if len(sorted_datasets) > preview:
        oldest = get_dataset_name_from_url(sorted_datasets[-1])
        print(f"  ... and {len(sorted_datasets) - preview} older datasets through {oldest}")

    return sorted_datasets

def download_all_sec_datasets():
    """
    Main function to discover and download all available SEC datasets.
    
    Returns:
        bool: True if any new datasets were downloaded, False otherwise.
    """
    print("SEC Financial Statement Datasets Downloader")
    print("=" * 50)
    
    # Create the data directory if it doesn't exist
    data_dir = Path(DATA_BASE_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Discover available datasets
    dataset_urls = get_available_datasets()
    
    if not dataset_urls:
        print("No datasets discovered. Exiting.")
        return False
    
    # Sort datasets by recency (most recent first)
    dataset_urls = sort_datasets_by_recency(dataset_urls)
    
    # Track progress
    successful_downloads = 0
    failed_downloads = 0
    skipped_downloads = 0
    
    print(f"\nStarting download of {len(dataset_urls)} datasets...")
    print("-" * 50)
    
    for i, url in enumerate(dataset_urls, 1):
        dataset_name = get_dataset_name_from_url(url)
        extract_to = data_dir / dataset_name
        
        print(f"\n[{i}/{len(dataset_urls)}] Processing: {dataset_name}")
        
        # Check if dataset already exists
        if extract_to.exists() and any(extract_to.iterdir()):
            print(f"Dataset {dataset_name} already exists, skipping...")
            skipped_downloads += 1
            continue
        
        # Download and extract
        if download_and_unzip(url, extract_to):
            successful_downloads += 1
            print(f"✓ Successfully downloaded {dataset_name}")
        else:
            failed_downloads += 1
            print(f"✗ Failed to download {dataset_name}")
        
        # Add a small delay to be respectful to the server
        time.sleep(1)
    
    # Summary
    print("\n" + "=" * 50)
    print("DOWNLOAD SUMMARY")
    print("=" * 50)
    print(f"Total datasets discovered: {len(dataset_urls)}")
    print(f"Successfully downloaded: {successful_downloads}")
    print(f"Failed downloads: {failed_downloads}")
    print(f"Skipped (already exists): {skipped_downloads}")
    
    if failed_downloads > 0:
        print(f"\nNote: {failed_downloads} datasets failed to download.")
        print("This could be due to:")
        print("- Files not yet available on the server")
        print("- Network issues")
        print("- Server restrictions")
        print("\nYou can run this script again to retry failed downloads.")
    
    # Return True if any new data was downloaded
    return successful_downloads > 0

if __name__ == "__main__":
    download_all_sec_datasets()
