#!/usr/bin/env python3
"""
Simple test script to download and load the Quora dataset.
Run this in your terminal where you have better network access.
"""

import sys
from pathlib import Path

def test_quora_download():
    """Test downloading and loading the Quora dataset"""
    
    try:
        from beir.util import download_and_unzip
        from beir.datasets.data_loader import GenericDataLoader
        
        data_path = Path("quora")
        
        # Try primary URL first
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/beir/datasets/download/quora.zip"
        print(f"Attempting to download from: {url}")
        
        try:
            result_path = download_and_unzip(url=url, out_dir=str(data_path))
            print(f"✓ Download successful! Result path: {result_path}")
        except Exception as e:
            print(f"✗ Primary URL failed: {e}")
            # Try alternative URL
            url = "https://tu-darmstadt.de/beir/datasets/download/quora.zip"
            print(f"\nAttempting to download from: {url}")
            result_path = download_and_unzip(url=url, out_dir=str(data_path))
            print(f"✓ Download successful! Result path: {result_path}")
        
        # List what was downloaded
        if data_path.exists():
            print(f"\nDownloaded files:")
            for item in data_path.rglob("*"):
                if item.is_file():
                    size = item.stat().st_size / 1024 / 1024  # MB
                    print(f"  {item.relative_to(data_path)} ({size:.2f} MB)")
        
        # Try to load the dataset
        print("\nLoading dataset...")
        loader = GenericDataLoader(data_folder=str(data_path))
        corpus, queries, qrels = loader.load(split="test")
        
        print(f"✓ Dataset loaded successfully!")
        print(f"  Documents: {len(corpus)}")
        print(f"  Queries: {len(queries)}")
        print(f"  Query-relevance pairs: {sum(len(rels) for rels in qrels.values())}")
        
        # Show sample
        if corpus:
            sample_doc_id = list(corpus.keys())[0]
            print(f"\nSample document ({sample_doc_id}):")
            text = corpus[sample_doc_id]['text'][:200] + "..." if len(corpus[sample_doc_id]['text']) > 200 else corpus[sample_doc_id]['text']
            print(f"  {text}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_quora_download()
    sys.exit(0 if success else 1)
