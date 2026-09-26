#!/usr/bin/env python3
"""
Test script to verify Quora dataset loading works correctly.
This tests the _load_quora_dataset function without running full experiments.
"""

from pathlib import Path
import sys
import json

def test_quora_loading():
    """Test that we can load and sample from the Quora dataset"""
    
    # Import the function we want to test
    sys.path.insert(0, '/home/matthew/git/pocket_dimension_experiments')
    
    try:
        from main import _load_quora_dataset
        
        print("Testing Quora dataset loading with sample_size=100...")
        corpus, queries, qrels = _load_quora_dataset(sample_size=100)
        
        print(f"✓ Successfully loaded {len(corpus)} documents and {len(queries)} queries")
        print(f"✓ Number of query-relevance pairs: {sum(len(rels) for rels in qrels.values())}")
        
        # Verify the data structure
        assert isinstance(corpus, dict), "Corpus should be a dict"
        assert isinstance(queries, dict), "Queries should be a dict"
        assert isinstance(qrels, dict), "Qrels should be a dict"
        
        # Check that all documents have text
        for doc_id, doc_data in corpus.items():
            assert "text" in doc_data, f"Document {doc_id} missing 'text' field"
            assert isinstance(doc_data["text"], str), f"Document {doc_id} text should be string"
        
        # Check that all queries have text
        for qid, query_text in queries.items():
            assert isinstance(query_text, str), f"Query {qid} should be string"
        
        print("✓ All data structure validations passed")
        
        # Show a sample document and query
        sample_doc_id = list(corpus.keys())[0]
        sample_query_id = list(queries.keys())[0]
        
        print(f"\nSample document ({sample_doc_id}):")
        print(f"  {corpus[sample_doc_id]['text'][:100]}...")
        
        print(f"\nSample query ({sample_query_id}):")
        print(f"  {queries[sample_query_id][:100]}...")
        
        if sample_query_id in qrels:
            relevant_docs = list(qrels[sample_query_id].keys())
            print(f"\nRelevant documents for this query: {relevant_docs[:5]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading Quora dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_quora_loading()
    sys.exit(0 if success else 1)
