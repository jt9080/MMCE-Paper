from dotenv import load_dotenv
import os
import json
import argparse
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from src.clients.openrouter_client import OpenRouterClient
from src.clients.azure_client import AzureClient
from src.experiments.bm25_retrieval import BM25Retrieval

load_dotenv(override=True)


class ContextualInsights(BaseModel):
    """Schema for contextual insights response"""
    intent: str
    tone: str
    context: str
    visual_context: str


class ClaimResponse(BaseModel):
    claims: List[str] = Field(description="List of extracted claims. Prefer single claim when possible, but include multiple claims if the text contains distinct factual assertions that cannot be combined into one coherent claim.")


class SimpleContextualAnalyzer:
    def __init__(self, client_type: str = "openrouter", model: Optional[str] = None, include_vision: bool = False):
        if client_type.lower() == "openrouter":
            self.client = OpenRouterClient()
            self.model = model or "google/gemini-2.0-flash-001"
        else:
            raise ValueError(f"Unsupported client type: {client_type}. Use 'openrouter'")
        
        # Load the base claim extraction prompt
        self.base_prompt = self._load_base_prompt()
        
        # Initialize BM25 retriever for ICL
        self.bm25_retriever = BM25Retrieval()
        
        # Initialize image analysis cache
        self.cache_file = Path(__file__).parent / "image_analysis_cache.json"
        self.image_cache = self._load_image_cache()
        
        # Initialize contextual analysis cache (model-specific)
        safe_model_name = self.model.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
        self.contextual_cache_file = Path(__file__).parent / f"contextual_analysis_cache_{safe_model_name}.json"
        self.contextual_cache = self._load_contextual_cache()
        
        # Store vision flag
        self.include_vision = include_vision

    def _load_image_cache(self) -> dict:
        """Load image analysis cache from file"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                    print(f"Loaded {len(cache)} cached image analyses")
                    return cache
            except (json.JSONDecodeError, Exception) as e:
                print(f"Warning: Could not load image cache: {e}")
                return {}
        return {}
    
    def _save_image_cache(self):
        """Save image analysis cache to file"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.image_cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save image cache: {e}")
    
    def _load_contextual_cache(self) -> dict:
        """Load contextual analysis cache from file"""
        if self.contextual_cache_file.exists():
            try:
                with open(self.contextual_cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                    print(f"Loaded {len(cache)} cached contextual analyses")
                    return cache
            except (json.JSONDecodeError, Exception) as e:
                print(f"Warning: Could not load contextual cache: {e}")
                return {}
        return {}
    
    def _save_contextual_cache(self):
        """Save contextual analysis cache to file"""
        try:
            with open(self.contextual_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.contextual_cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save contextual cache: {e}")

    def _get_relative_path(self, image_path: str) -> str:
        """Convert absolute path to relative path for cache consistency"""
        try:
            # Get the project root directory
            project_root = Path(__file__).parent.parent.parent
            
            # Convert to Path object and make it relative to project root
            path_obj = Path(image_path)
            if path_obj.is_absolute():
                try:
                    relative_path = path_obj.relative_to(project_root)
                    return str(relative_path)
                except ValueError:
                    # If path is not under project root, return as is
                    return image_path
            else:
                return image_path
        except Exception:
            # If anything goes wrong, return the original path
            return image_path

    def _load_base_prompt(self) -> str:
        """Load the base claim extraction prompt"""
        # Try to load the specific prompt for this approach first
        prompt_path = Path(__file__).parent.parent / "prompts" / "experiments" / "image_text_baseline.md"
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def _analyze_image(self, image_path: str) -> dict:
        """Simple image analysis using Azure Vision API with caching"""
        # Convert absolute path to relative path for cache key
        cache_key = self._get_relative_path(image_path)
        
        # Check cache first
        if cache_key in self.image_cache:
            print(f"  Image cache hit")
            return self.image_cache[cache_key]
        
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Azure Vision API call timed out")
        
        try:
            azure_client = AzureClient()            
            # Set a 30-second timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)
            details = azure_client.get_image_details(image_path)
            signal.alarm(0)  # Cancel the alarm
            
            result = {
                "captions": details.get("captions", []),
                "tags": details.get("tags", []),
                "ocrs": details.get("ocr", []),
                "success": True
            }
            
            # Cache the result using relative path
            self.image_cache[cache_key] = result
            self._save_image_cache()
            
            print(f"  Image analysis successful for: {image_path}")
            return result
        except TimeoutError:
            signal.alarm(0)
            print(f"Warning: Image analysis timed out for {image_path}")
            return {"captions": [], "tags": [], "ocrs": [], "success": False}
        except Exception as e:
            signal.alarm(0)
            print(f"Warning: Image analysis failed for {image_path}: {e}")
            return {"captions": [], "tags": [], "ocrs": [], "success": False}

    def _get_contextual_insights(self, text: str, image_paths: List[str], article_url: str = None) -> dict:
        """Get contextual insights with caching"""
        # Use article URL as cache key if available, otherwise fall back to text+images
        if article_url:
            cache_key = article_url
        else:
            cache_key = f"{text}|{sorted(image_paths)}"
        
        # Check cache first
        if cache_key in self.contextual_cache:
            print(f"  Contextual analysis cache hit")
            return self.contextual_cache[cache_key]
        
        # Load the contextual insights prompt from file
        prompt_path = Path(__file__).parent.parent / "prompts" / "experiments" / "contextual_insights_prompt.md"
        with open(prompt_path, 'r', encoding='utf-8') as f:
            context_prompt = f.read().strip()

        # Limit images to 29 to avoid the 30-image limit (only for Qwen models)
        if "qwen" in self.model.lower():
            limited_image_paths = image_paths[:29] if len(image_paths) > 29 else image_paths
            if len(image_paths) > 29:
                print(f"  Warning: Limited images from {len(image_paths)} to 29 to avoid Qwen API limit")
        else:
            limited_image_paths = image_paths

        response = self.client.generate(
            context_prompt, 
            text, 
            images=limited_image_paths, 
            model=self.model,
            response_schema=ContextualInsights
        )
        
        # Return the ContextualInsights object as a dict
        if isinstance(response, ContextualInsights):
            result = {
                "intent": response.intent,
                "tone": response.tone,
                "context": response.context,
                "visual_context": response.visual_context
            }
        elif isinstance(response, dict):
            # If response is already a dict, use it
            result = response
        else:
            # Handle plain text response
            result = self._parse_contextual_text_response(str(response))
        
        # Cache the result
        self.contextual_cache[cache_key] = result
        self._save_contextual_cache()
        
        print(f"  Contextual analysis completed and cached")
        return result
    
    def _parse_contextual_text_response(self, text: str) -> dict:
        """Parse plain text contextual analysis response into structured format"""
        import re
        
        # Initialize result with empty strings
        result = {
            "intent": "",
            "tone": "",
            "context": "",
            "visual_context": ""
        }
        
        # Try to extract structured information using regex patterns
        # Look for **bold** headers followed by content
        patterns = {
            "intent": r"\*\*Intent:\*\*\s*(.*?)(?=\*\*|$)",
            "tone": r"\*\*Tone:\*\*\s*(.*?)(?=\*\*|$)",
            "context": r"\*\*Context:\*\*\s*(.*?)(?=\*\*|$)",
            "visual_context": r"\*\*Visual Context:\*\*\s*(.*?)(?=\*\*|$)"
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                result[key] = match.group(1).strip()
        
        # If no structured patterns found, try to extract from the existing format
        if not any(result.values()):
            # The current format seems to be a single paragraph, so we'll need to parse it differently
            # For now, put the entire text in the context field
            result["context"] = text.strip()
            result["intent"] = "Unable to parse structured response"
            result["tone"] = "Unable to parse structured response"
            result["visual_context"] = "Unable to parse structured response"
        
        return result

    def _create_enhanced_prompt(self, text: str, context_insights: dict, image_data: dict) -> str:
        """Create an enhanced prompt that follows the base prompt structure"""
        # Start with the base prompt
        enhanced_prompt = self.base_prompt
        
        # Add contextual insights to the CONTEXT ANALYSIS section
        enhanced_prompt += f"\n\n# EXTRA CONTEXTUAL INFORMATION\n"
        enhanced_prompt += "You will be given more contextual insights to help you better understand the intent, tone, and visual context of the post. This could be relevant if the post text is ambiguous, or contain humour, sarcasm, irony, etc. Include relevant context in the claim you extract. **However, it is very important that you still focus on what the image claims to show, and not add unnecessary details, broader commentary or context. You must keep the final claim concise.**\n\n"

        enhanced_prompt += "**Contextual Insights:**\n"
        enhanced_prompt += f"• Intent: {context_insights['intent']}\n"
        enhanced_prompt += f"• Tone: {context_insights['tone']}\n"
        enhanced_prompt += f"• Context: {context_insights['context']}\n"
        enhanced_prompt += f"• Visual Context: {context_insights['visual_context']}\n\n"
        
        # Add image data as supporting evidence in the CONTEXT ANALYSIS section
        if image_data.get('captions'):
            enhanced_prompt += "**Image Captions (for context):**\n"
            for i, caption in enumerate(image_data['captions'], 1):
                enhanced_prompt += f"{i}. {caption}\n"
            enhanced_prompt += "\n"
        
        if image_data.get('tags'):
            enhanced_prompt += f"**Image Tags (for context):** {', '.join(image_data['tags'])}\n\n"
        
        if image_data.get('ocrs'):
            enhanced_prompt += f"**Image OCR Text (for context):** {' '.join(image_data['ocrs'])}\n\n"
        
        # Add the input text at the end
        enhanced_prompt += f"\n# INPUT TEXT AND IMAGE(S)\n\n{text}"
        
        # Save enhanced prompt to file for review
        prompt_output_path = Path(__file__).parent.parent / "prompts" / "experiments" / "enhanced_prompt_output.md"
        with open(prompt_output_path, 'w', encoding='utf-8') as f:
            f.write(enhanced_prompt)
        
        return enhanced_prompt

    def analyze_post(self, text: str, image_paths: Optional[List[str]] = None, current_item_id: str = None, article_url: str = None) -> dict:
        """Main analysis method"""
        if not image_paths:
            image_paths = []
        try:
            # Step 1: Image analysis (conditional based on include_vision flag)
            if self.include_vision and image_paths:
                print(f"Starting image analysis for {len(image_paths)} images...")
                # Process all images, not just the first one
                all_captions = []
                all_tags = []
                all_ocrs = []
                success_flags = []
                
                for i, image_path in enumerate(image_paths):
                    print(f"Analyzing image {i+1}/{len(image_paths)}")
                    try:
                        single_image_data = self._analyze_image(image_path)
                        all_captions.extend(single_image_data.get("captions", []))
                        all_tags.extend(single_image_data.get("tags", []))
                        all_ocrs.extend(single_image_data.get("ocrs", []))
                        success_flags.append(single_image_data.get("success", False))
                    except Exception as e:
                        print(f"Error analyzing image {i+1}: {e}")
                        success_flags.append(False)
                        # Continue with next image instead of failing completely
                    
                
                image_data = {
                    "captions": all_captions,
                    "tags": all_tags,
                    "ocrs": all_ocrs,
                    "success": any(success_flags)
                }
                print("Completed image analysis")
                
                # Check if image analysis failed - if so, stop completely
                if image_paths and not image_data.get("success", False):
                    print("Image analysis failed - stopping processing")
                    raise Exception("Image analysis failed - cannot proceed with claim extraction")
            else:
                # Set empty image data for compatibility when vision is disabled
                image_data = {"captions": [], "tags": [], "ocrs": [], "success": True}
                if not self.include_vision:
                    print("Skipped image analysis (--include-vision not specified)")
                else:
                    print("Skipped image analysis (no images provided)")
            
            # Step 2: Get contextual insights by analyzing the image directly
            context_insights = self._get_contextual_insights(text, image_paths, article_url)
            print("Completed contextual insights")
            
            # Step 3: Create enhanced prompt with both contextual insights and image data
            enhanced_prompt = self._create_enhanced_prompt(text, context_insights, image_data)
            
            # Step 3.5: Get ICL examples if current_item_id is provided
            examples = []
            if current_item_id:
                # Get similar examples using BM25 with leave-one-out method
                similar_examples = self.bm25_retriever.get_icl_examples_leave_one_out(text, current_item_id, num_examples=5)
                
                # Get the images directory path for constructing full paths
                script_dir = Path(__file__).parent
                images_dir = script_dir.parent.parent / "data/averimatec_with_fulltext/all_images"
                
                # Prepare examples and their corresponding images
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
                print(f"Adding {len(examples)} ICL examples")
            print("Completed ICL examples")
            
            # Step 4: Extract claim using enhanced prompt with ICL examples
            # Pass raw image paths directly to client (like baseline) - client will handle limiting
            claim_result = self.client.generate(
                enhanced_prompt,
                text,
                images=image_paths,
                examples=examples,
                response_schema=ClaimResponse,
                model=self.model
            )
            
            return {
                "image_data": image_data,
                "contextual_insights": context_insights,
                "extracted_claims": claim_result.claims,
            }
            
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            raise

    def analyze_post_with_metadata(self, full_text: str, claim_date: str, claim_speaker: str, claim_reporting_source: str, image_paths: Optional[List[str]] = None, current_item_id: str = None, article_url: str = None) -> dict:
        """Analyze post with metadata like baseline script"""
        # Format text like baseline script
        formatted_text = f"On {claim_date}, {claim_speaker} posted on {claim_reporting_source}: {full_text}"
        
        # Use the existing analyze_post method with ICL
        result = self.analyze_post(formatted_text, image_paths, current_item_id, article_url)
        
        # Add metadata to result
        result.update({
            "claim_date": claim_date,
            "claim_speaker": claim_speaker,
            "claim_reporting_source": claim_reporting_source,
            "formatted_text": formatted_text
        })
        
        return result

def main():
    parser = argparse.ArgumentParser(description='Run simplified claim extraction with contextual enhancement')
    parser.add_argument('--client', type=str, choices=['openrouter'], default='openrouter')
    parser.add_argument('--model', type=str, default='google/gemini-2.0-flash-001', help='Specify the model to use (for file naming)')
    parser.add_argument('--include-vision', action='store_true', help='Include image analysis in the processing pipeline')
    args = parser.parse_args()
    
    analyzer = SimpleContextualAnalyzer(client_type=args.client, model=args.model, include_vision=args.include_vision)
    
    # Batch processing mode (like baseline script)
    print(f"Using {args.client.upper()} client with model: {analyzer.model}")
    
    # Get the script's directory and construct paths relative to it
    script_dir = Path(__file__).parent
    json_path = script_dir.parent.parent / "data/averimatec_with_fulltext/combined_data_deduplicated.json"
    images_dir = script_dir.parent.parent / "data/averimatec_with_fulltext/all_images"
    
    # Sanitize model name for filename (replace slashes and other invalid chars)
    safe_model_name = args.model.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
    
    # Add _with_vision suffix if vision is enabled
    vision_suffix = "_with_vision" if args.include_vision else ""
    output_path = script_dir.parent / f"experiment_results/my_approach{vision_suffix}_{safe_model_name}.json"
    
    print(f"Results will be saved to: {output_path}")
    
    # Load data
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
            # Analyze with metadata like baseline script
            analysis_result = analyzer.analyze_post_with_metadata(
                full_text, claim_date, claim_speaker, claim_reporting_source, image_paths, current_item_id=item.get('article'), article_url=item.get('article')
            )
            
            # Create result object
            result = {
                "article": item.get('article'),
                "full_text": full_text,
                "image_links": item.get('claim_images'),
                "ground_truth_image_used": item.get("metadata", {}).get('image_used'),
                "ground_truth_claim": item.get('claim_text'),
                "my_approach_results": analysis_result
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
            print(f"Individual Claims: {analysis_result['extracted_claims']}")
            print(f"Checkpoint saved. Total results: {len(existing_results)}")
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            continue
    
    # Final cache save to ensure all cached data is persisted
    analyzer._save_image_cache()
    analyzer._save_contextual_cache()
    
    print(f"\nAll processing complete. Final results saved to {output_path}")
    print(f"Total results: {len(existing_results)}")


if __name__ == "__main__":
    main()
