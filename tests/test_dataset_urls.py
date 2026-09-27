#!/usr/bin/env python3
"""
Test script to find working BEIR dataset URLs.
This will help us identify the correct download location.
"""

import requests
from pathlib import Path

def test_url(url, description):
    """Test if a URL returns a valid zip file"""
    print(f"\nTesting {description}...")
    print(f"URL: {url}")
    
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        content_type = response.headers.get('content-type', '')
        content_length = response.headers.get('content-length', 'unknown')
        
        print(f"  Status: {response.status_code}")
        print(f"  Content-Type: {content_type}")
        print(f"  Content-Length: {content_length}")
        
        if response.status_code == 200 and 'zip' in content_type.lower():
            print("  ✓ Looks like a valid zip file!")
            return True
        elif response.status_code == 200:
            # Try to download a small portion to check
            print("  Content-Type suggests it might not be a zip, downloading sample...")
            response = requests.get(url, timeout=30)
            first_bytes = response.content[:100]
            if first_bytes.startswith(b'PK'):
                print("  ✓ Actually IS a zip file (starts with PK)!")
                return True
            else:
                print(f"  ✗ Not a zip file. First bytes: {first_bytes[:20]}")
                return False
        else:
            print(f"  ✗ HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    """Test various BEIR dataset URLs"""
    
    # Known BEIR datasets with their sizes (from documentation)
    datasets = [
        {
            "name": "nfcorpus",
            "description": "NF Corpus - News articles",
            "urls": [
                "https://public.ukp.informatik.tu-darmstadt.de/thakur/beir/datasets/download/nfcorpus.zip",
                "https://tu-darmstadt.de/beir/datasets/download/nfcorpus.zip"
            ],
            "size": "~130K documents"
        },
        {
            "name": "scidocs",
            "description": "SCIDOCS - Scientific documents",
            "urls": [
                "https://public.ukp.informatik.tu-darmstadt.de/thakur/beir/datasets/download/scidocs.zip",
                "https://tu-darmstadt.de/beir/datasets/download/scidocs.zip"
            ],
            "size": "~25K documents"
        },
        {
            "name": "fiqa",
            "description": "FIQA - Financial domain QA",
            "urls": [
                "https://public.ukp.informatik.tu-darmstadt.de/thakur/beir/datasets/download/fiqa.zip",
                "https://tu-darmstadt.de/beir/datasets/download/fiqa.zip"
            ],
            "size": "~50K documents"
        },
        {
            "name": "dbpedia-entity",
            "description": "DBpedia Entity - Entity search",
            "urls": [
                "https://public.ukp.informatik.tu-darmstadt.de/thakur/beir/datasets/download/dbpedia-entity.zip",
                "https://tu-darmstadt.de/beir/datasets/download/dbpedia-entity.zip"
            ],
            "size": "~400K documents"
        }
    ]
    
    print("=" * 80)
    print("Testing BEIR Dataset URLs")
    print("=" * 80)
    
    working_urls = []
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset['name']} - {dataset['description']}")
        print(f"Expected size: {dataset['size']}")
        print('='*60)
        
        for url in dataset['urls']:
            if test_url(url, f"{dataset['name']} URL {dataset['urls'].index(url)+1}"):
                working_urls.append({
                    "name": dataset['name'],
                    "url": url,
                    "size": dataset['size']
                })
    
    print("\n" + "=" * 80)
    if working_urls:
        print("SUMMARY: Working URLs found!")
        print("=" * 80)
        for w in working_urls:
            print(f"\nDataset: {w['name']}")
            print(f"Size: {w['size']}")
            print(f"URL: {w['url']}")
    else:
        print("NO WORKING URLs FOUND")
        print("This suggests the BEIR dataset hosting might be down or moved.")
        print("\nAlternative approach: Use BEIR's download script directly")
        print("Try: python -m beir.datasets.download nfcorpus")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
