# -*- coding: utf-8 -*-
"""
Enhanced script to discover and download ALL available SEC financial statement datasets.

This script:
1. Scrapes the SEC server to discover all available datasets
2. Downloads and extracts all discovered datasets
3. Provides progress tracking and error handling
4. Skips already downloaded datasets

Created on Fri Aug 12 10:19:50 2024
Enhanced to discover all available datasets

@author: U.S. Securities and Exchange Commission.
Enhanced by: AI Assistant
"""

import requests
import zipfile
import os
import re
from io import BytesIO
from urllib.parse import urljoin, urlparse
import time
from pathlib import Path

from config import DATA_BASE_DIR

def get_available_datasets(base_url="https://www.sec.gov/files/dera/data/financial-statement-data-sets/"):
    """
    Scrapes the SEC server to discover all available dataset files
    
    Parameters
    ----------
    base_url : str
        Base URL of the SEC financial statement datasets
        
    Returns
    -------
    list
        List of discovered dataset URLs
    """
    print(f"Discovering available datasets from {base_url}...")
    
    # Define headers with a User-Agent (required by SEC)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 (Contact: your-email@example.com)'
    }
    
    try:
        # Get the main page content
        response = requests.get(base_url, headers=headers)
        response.raise_for_status()
        
        # Look for ZIP files in the HTML content
        content = response.text
        
        # Pattern to match ZIP files (quarterly and annual datasets)
        # This pattern looks for links ending in .zip
        zip_pattern = r'href=["\']([^"\']*\.zip)["\']'
        zip_files = re.findall(zip_pattern, content)
        
        # Also look for direct file listings if the page structure is different
        # Some servers list files directly
        if not zip_files:
            # Alternative pattern for file listings
            file_pattern = r'([0-9]{4}q[1-4]\.zip)|([0-9]{4}\.zip)'
            file_matches = re.findall(file_pattern, content)
            zip_files = [match[0] if match[0] else match[1] for match in file_matches if any(match)]
        
        # Convert relative URLs to absolute URLs
        dataset_urls = []
        for zip_file in zip_files:
            if zip_file.startswith('http'):
                dataset_urls.append(zip_file)
            else:
                dataset_urls.append(urljoin(base_url, zip_file))
        
        # Remove duplicates and sort
        dataset_urls = sorted(list(set(dataset_urls)))
        
        print(f"Discovered {len(dataset_urls)} datasets:")
        for url in dataset_urls:
            print(f"  - {url}")
            
        return dataset_urls
        
    except requests.RequestException as e:
        print(f"Error discovering datasets: {e}")
        print("Falling back to known dataset list...")
        return get_fallback_datasets()

def get_fallback_datasets():
    """
    Fallback method that returns a comprehensive list of known SEC datasets
    based on typical naming patterns and available years
    
    Returns
    -------
    list
        List of known dataset URLs
    """
    base_url = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/"
    
    # Generate URLs for common patterns
    datasets = []
    
    # Quarterly datasets (2009-2024)
    for year in range(2024, 2026):
        for quarter in range(1, 5):
            # # Skip future quarters
            # if year == 2024 and quarter > 2:  # As of 2024, Q3 and Q4 might not be available yet
            #     continue
            datasets.append(f"{base_url}{year}q{quarter}.zip")
    
    # Annual datasets (if they exist)
    for year in range(2024, 2026):
        datasets.append(f"{base_url}{year}.zip")
    
    print(f"Using fallback list with {len(datasets)} potential datasets")
    return datasets

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
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 (Contact: your-email@example.com)'
    }
    
    for attempt in range(max_retries):
        try:
            print(f"Downloading ZIP file from {url} (attempt {attempt + 1}/{max_retries})...")
            response = requests.get(url, headers=headers, timeout=300)  # 5 minute timeout
            
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
    
    print("Datasets sorted by recency (most recent first):")
    for i, url in enumerate(sorted_datasets[:10], 1):  # Show first 10
        dataset_name = get_dataset_name_from_url(url)
        print(f"  {i:2d}. {dataset_name}")
    if len(sorted_datasets) > 10:
        print(f"  ... and {len(sorted_datasets) - 10} more datasets")
    
    return sorted_datasets

def main():
    """
    Main function to discover and download all available SEC datasets
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
        return
    
    # Sort datasets by recency (most recent first)
    print(f"\nSorting {len(dataset_urls)} datasets by recency...")
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

if __name__ == "__main__":
    main()
