from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
import os
import json
import argparse
from pathlib import Path
import sys

# Add the src directory to the path so we can import the llm module
sys.path.append(str(Path(__file__).parent.parent))
from clients.openrouter_client import OpenRouterClient
from experiments.bm25_retrieval import BM25Retrieval

load_dotenv(override=True)

class ClaimResponse(BaseModel):
    claims: List[str] = Field(description="List of extracted claims. Prefer single claim when possible, but include multiple claims if the text contains distinct factual assertions that cannot be combined into one coherent claim.")

class ClaimAnalyzer:
    def __init__(self, client_type: str = "openrouter", model: str = None):
        # Create client based on the specified type
        if client_type.lower() == "openrouter":
            self.client = OpenRouterClient()
            self.model = model or "google/gemini-2.0-flash-001"
        else:
            raise ValueError(f"Unsupported client type: {client_type}. Use 'openrouter'")
        
        # Load prompts from markdown files
        self.text_only_prompt = self._load_prompt("text_only_baseline.md")
        self.image_text_prompt = self._load_prompt("image_text_baseline.md")
        
        # Initialize BM25 retriever for ICL
        self.bm25_retriever = BM25Retrieval()

    def _load_prompt(self, filename: str) -> str:
        """Load prompt from markdown file"""
        prompt_path = Path(__file__).parent.parent / "prompts" / "experiments" / filename
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        except Exception as e:
            raise Exception(f"Error loading prompt from {filename}: {str(e)}")

    def analyze_text_only(self, text: str) -> ClaimResponse:
        text_to_extract = f"Extract the claim(s) from the following text: {text}"
        return self.client.generate(self.text_only_prompt, text_to_extract, response_schema=ClaimResponse, model=self.model)

    def analyze_image_text(self, text: str, image_paths: List[str] = None) -> ClaimResponse:
        text_to_extract = f"Extract the claim(s) from the following text: {text}"
        
        if not image_paths:
            image_paths = []
        
        return self.client.generate(self.image_text_prompt, text_to_extract, images=image_paths, response_schema=ClaimResponse, model=self.model)

    def analyze_image_text_icl(self, text: str, image_paths: List[str] = None, current_item_id: str = None) -> ClaimResponse:
        # Get similar examples using BM25 with leave-one-out method
        if current_item_id:
            similar_examples = self.bm25_retriever.get_icl_examples_leave_one_out(text, current_item_id, num_examples=5)
        else:
            # Fallback to regular retrieval if no item ID provided
            similar_examples = self.bm25_retriever.get_icl_examples(text, num_examples=5)
        
        # Start with the base prompt
        text_to_extract = f"Extract the claim(s) from the following text: {text}"
        if not image_paths:
            image_paths = []
        
        # Get the images directory path for constructing full paths
        script_dir = Path(__file__).parent
        images_dir = script_dir.parent.parent / "data/averimatec_with_fulltext/all_images"
        
        # Prepare examples and their corresponding images
        examples = []
        
        for i, example in enumerate(similar_examples):
            # Construct full image paths for the example
            example_image_paths = []
            if 'claim_images' in example:
                for img_name in example['claim_images']:
                    img_path = images_dir / img_name
                    if img_path.exists():
                        example_image_paths.append(str(img_path))
                    else:
                        print(f"Warning: Example image not found: {img_path}")
            
            example_obj = {
                "text": f'EXAMPLE {i+1}:\n Extract the claim(s) from the following text: {example.get("full_text", "")}',
                "images": example_image_paths,
                "claim_text": f'Extracted Claims: {example.get("claim_text", "")}'
            }
            examples.append(example_obj)
        
        # Only pass current images to LLM (not example images)
        if not image_paths:
            image_paths = []
        
        return self.client.generate(self.image_text_prompt, text_to_extract, images=image_paths, examples=examples, response_schema=ClaimResponse, model=self.model)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run baseline claim extraction experiments')
    parser.add_argument('--client', type=str, choices=['openrouter'], 
                       default='openrouter', help='Choose the LLM client to use')
    parser.add_argument('--model', type=str, default='google/gemini-2.0-flash-001',
                       help='Specify the model to use (for file naming)')
    args = parser.parse_args()
    
    client_type = args.client
    model = args.model
    analyzer = ClaimAnalyzer(client_type=client_type)
    
    # Get the script's directory and construct paths relative to it
    script_dir = Path(__file__).parent
    json_path = script_dir.parent.parent / "data/averimatec_with_fulltext/combined_data_deduplicated.json"
    images_dir = script_dir.parent.parent / "data/averimatec_with_fulltext/all_images"
    
    # Sanitize model name for filename (replace slashes and other invalid chars)
    safe_model_name = model.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
    output_path = script_dir.parent / f"experiment_results/baseline_{safe_model_name}.json"
    
    print(f"Using {client_type.upper()} client with model: {analyzer.model}")
    print(f"Results will be saved to: {output_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Load existing results if file exists
    existing_results = []
    if output_path.exists():
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            print(f"Loaded {len(existing_results)} existing results from checkpoint")
        except json.JSONDecodeError:
            print("Warning: Existing results file is corrupted, starting fresh")
            existing_results = []
    
    # Create a set of already processed articles for quick lookup
    processed_articles = {result.get('article') for result in existing_results}
    
    # Process each item in the JSON
    for i, item in enumerate(data):
        # Skip if already processed
        if item.get('article') in processed_articles:
            print(f"Skipping already processed article: {item.get('article')}")
            continue
        
        print("\n" + "="*80)
        print(f"Processing claim {i+1}/{len(data)}: {item['claim_text']}")
        print("="*80)

        full_text = item.get('full_text', '')
        claim_date = item.get('date', '')
        claim_speaker = item.get('metadata', {}).get('speaker', '')
        claim_reporting_source = item.get('metadata', {}).get('reporting_source', '')
        full_text = f"On {claim_date}, {claim_speaker} posted on {claim_reporting_source}: {full_text}"
        
        # Get image paths
        image_paths = []
        if 'claim_images' in item:
            for img_name in item['claim_images']:
                img_path = images_dir / img_name
                if img_path.exists():
                    image_paths.append(str(img_path))
                else:
                    print(f"Warning: Image not found: {img_path}")
        
        try:
            # Text-only analysis
            text_result = analyzer.analyze_text_only(full_text)
            print("\nText-only analysis result:")
            print(text_result.model_dump_json(indent=2))
            
            # Image-text analysis
            image_text_result = analyzer.analyze_image_text(full_text, image_paths)
            print("\nImage-text analysis result:")
            print(image_text_result.model_dump_json(indent=2))
            
            # Image-text ICL analysis with leave-one-out
            image_text_icl_result = analyzer.analyze_image_text_icl(full_text, image_paths, current_item_id=item.get('article'))
            print("\nImage-text ICL analysis result:")
            print(image_text_icl_result.model_dump_json(indent=2))
            
            # Create result object
            result = {
                "article": item.get('article'),
                "full_text": full_text,
                "image_links": item.get('claim_images'),
                "ground_truth_image_used": item.get("metadata", {}).get('image_used'),
                "ground_truth_claim": item.get('claim_text'),
                "text_only_results": text_result.model_dump(),
                "image_text_results": image_text_result.model_dump(),
                "image_text_icl_results": image_text_icl_result.model_dump()
            }
            
            # Append to existing results
            existing_results.append(result)
            
            # Save checkpoint after each successful processing
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(existing_results, f, indent=2, ensure_ascii=False)
            
            # Update processed articles set
            processed_articles.add(item.get('article'))
            
            # Print progress
            print(f"Successfully processed claim with {len(image_paths)} images")
            print(f"Checkpoint saved. Total results: {len(existing_results)}")
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
    
    print(f"\nAll processing complete. Final results saved to {output_path}")
    print(f"Total results: {len(existing_results)}")

if __name__ == "__main__":
    main()