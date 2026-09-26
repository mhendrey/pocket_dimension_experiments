#!/usr/bin/env python3
"""
Download a BEIR dataset using BEIR's built-in download functionality.
This is more reliable than direct URL downloads.
"""

import sys
from pathlib import Path

def download_beir_dataset(dataset_name: str, output_dir: str | None = None) -> bool:
    """
    Download a BEIR dataset using BEIR's built-in downloader.
    
    Args:
        dataset_name: Name of the BEIR dataset (e.g., 'nfcorpus', 'scidocs')
        output_dir: Optional output directory
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from beir.datasets.data_loader import GenericDataLoader
        from beir.datasets.download import download
        
        print(f"Downloading {dataset_name} dataset...")
        
        # Use BEIR's built-in download function
        data_path = Path(output_dir) if output_dir else Path(dataset_name)
        
        # Download the dataset
        download(data_path=str(data_path), url=None, name=dataset_name)
        
        print(f"✓ Dataset downloaded to {data_path}")
        
        # List downloaded files
        if data_path.exists():
            print("\nDownloaded files:")
            for item in sorted(data_path.rglob("*")):
                if item.is_file():
                    size = item.stat().st_size / 1024 / 1024  # MB
                    print(f"  {item.relative_to(data_path)} ({size:.2f} MB)")
        
        return True
        
    except Exception as e:
        print(f"✗ Error downloading {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def list_available_datasets():
    """List some commonly used BEIR datasets with their sizes"""
    datasets = [
        {"name": "nfcorpus", "description": "News articles (130K docs)", "size": "~130K"},
        {"name": "scidocs", "description": "Scientific documents (25K docs)", "size": "~25K"},
        {"name": "fiqa", "description": "Financial QA (50K docs)", "size": "~50K"},
        {"name": "dbpedia-entity", "description": "Entity search (400K docs)", "size": "~400K"},
    ]
    
    print("\nAvailable BEIR datasets:")
    print("=" * 80)
    for ds in datasets:
        print(f"  {ds['name']:20s} - {ds['description']}")
    print("=" * 80)

if __name__ == "__main__":
    # List available datasets
    list_available_datasets()
    
    # Try to download a medium-sized dataset (nfcorpus has ~130K documents)
    print("\nAttempting to download 'nfcorpus' (~130K documents)...")
    success = download_beir_dataset("nfcorpus")
    
    if success:
        print("\n✓ Download successful! You can now use this dataset in your experiments.")
        sys.exit(0)
    else:
        print("\n✗ Download failed. Try another dataset or check your network connection.")
        sys.exit(1)
