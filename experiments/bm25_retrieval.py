import json
from rank_bm25 import BM25Okapi
from pathlib import Path

class BM25Retrieval:
    def __init__(self, data_path=None):
        script_dir = Path(__file__).parent
        if data_path is None:
            data_path = script_dir.parent.parent / "data/averimatec_with_fulltext/combined_data_deduplicated.json"
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.train_texts = [item['full_text'] for item in self.data]
        self.tokenized_train = [text.lower().split() for text in self.train_texts]
        self.bm25 = BM25Okapi(self.tokenized_train)
    
    def get_icl_examples(self, query_text, num_examples=5):
        tokenized_query = query_text.lower().split()
        top_indices = self.bm25.get_top_n(tokenized_query, self.data, n=num_examples)
        return top_indices
    
    def get_icl_examples_leave_one_out(self, query_text, current_item_id, num_examples=5):
        """
        Get ICL examples using leave-one-out method to prevent data leakage.
        
        Args:
            query_text: The text to find similar examples for
            current_item_id: The ID of the current item to exclude from retrieval
            num_examples: Number of examples to retrieve
        """
        # Create a filtered dataset excluding the current item
        filtered_data = [item for item in self.data if item.get('article') != current_item_id]
        
        if len(filtered_data) == 0:
            print(f"Warning: No data available after filtering out item {current_item_id}")
            return []
        
        # Create BM25 index on filtered data
        filtered_texts = [item['full_text'] for item in filtered_data]
        tokenized_filtered = [text.lower().split() for text in filtered_texts]
        filtered_bm25 = BM25Okapi(tokenized_filtered)
        
        # Get top examples from filtered data
        tokenized_query = query_text.lower().split()
        top_indices = filtered_bm25.get_top_n(tokenized_query, filtered_data, n=num_examples)
        return top_indices

def test_bm25_retrieval():
    """Test the BM25 retrieval functionality."""
    print("Testing BM25 Retrieval...")
    
    # Initialize the retriever
    retriever = BM25Retrieval()
    print(f"Loaded {len(retriever.data)} examples from dataset")
    
    # Test queries
    test_queries = [
        "claim about donald trump",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test {i}: Query = '{query}' ---")
        
        # Get similar examples
        similar_examples = retriever.get_icl_examples(query, num_examples=5)
        
        print(f"Found {len(similar_examples)} similar examples:")
        # print(similar_examples)
        for j, example in enumerate(similar_examples, 1):
            # Truncate long text for display
            display_text = example['full_text'][:200] + "..." if len(example['full_text']) > 200 else example['full_text']
            print(f"  {j}. {display_text}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_bm25_retrieval()