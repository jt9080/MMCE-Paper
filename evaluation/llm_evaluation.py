import json
import numpy as np
import os
from dotenv import load_dotenv
from tqdm import tqdm
import time
from pydantic import BaseModel, Field
import sys
import argparse
from typing import Optional, Dict, Any, List, Union
from ..clients.openrouter_client import OpenRouterClient

load_dotenv(override=True)

def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_checkpoint(checkpoint_path):
    """Load existing results from checkpoint file"""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return []

def save_checkpoint(checkpoint_path, results):
    """Save results to checkpoint file"""
    with open(checkpoint_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

# Define response schemas for different evaluation types
class SimilarityScoreResponse(BaseModel):
    score: float = Field(..., ge=1, le=4, description="Similarity score between 1 and 4")
    reasoning: str = Field(..., description="Brief explanation of the score")

class EntailmentResponse(BaseModel):
    classification: str = Field(..., description="Classification: entailed, partially_entailed, or not_entailed")
    reasoning: str = Field(..., description="Brief explanation of the classification")

class DecontextualizationResponse(BaseModel):
    classification: str = Field(..., description="Classification: fully_decontextualized, partially_decontextualized, or not_decontextualized")
    reasoning: str = Field(..., description="Brief explanation of the classification")

# Map evaluation types to their response schemas
EVALUATION_SCHEMAS = {
    'reference_based': SimilarityScoreResponse,
    'entailment': EntailmentResponse,
    'decontextualization': DecontextualizationResponse
}

def get_evaluation_field_name(model_name: str, evaluation_type: str) -> str:
    """Determine the field name for evaluation results based on model"""
    if model_name and "gpt" in model_name.lower():
        return "gpt"
    elif model_name and "claude" in model_name.lower():
        return "claude"
    else:
        # Default to evaluation_type for unknown models
        return evaluation_type

def create_client(client_type: str = "openrouter", model: Optional[str] = None):
    """Create client based on the specified type"""
    if client_type.lower() == "openrouter":
        client = OpenRouterClient()
        model = model or "google/gemini-2.5-flash-lite"
    else:
        raise ValueError(f"Unsupported client type: {client_type}. Use 'openrouter'")
    
    return client, model

def get_all_claims(item, result_type):
    """Extract all claims from item based on result type for evaluation purposes"""
    if result_type == 'text_only':
        claims = item.get('text_only_results', {}).get('claims', [])
    elif result_type == 'image_text':
        claims = item.get('image_text_results', {}).get('claims', [])
    elif result_type == 'image_text_icl':
        claims = item.get('image_text_icl_results', {}).get('claims', [])
    elif result_type == 'two_stage_final_claim':
        final_claim = item.get('two_stage_results', {}).get('final_claim')
        return [final_claim] if final_claim else []
    elif result_type == 'my_approach':
        # Handle my_approach_results structure
        my_approach_results = item.get('my_approach_results', {})
        # Try extracted_claims first, then extracted_claim as fallback
        claims = my_approach_results.get('extracted_claims', [])
        if not claims:
            single_claim = my_approach_results.get('extracted_claim')
            if single_claim:
                claims = [single_claim]
        return claims
    else:
        raise ValueError(f"Unknown result_type: {result_type}")
    
    # Handle the new claims format (list of strings)
    if isinstance(claims, list):
        return claims
    elif isinstance(claims, str):
        # Handle legacy format where claim might still be a string
        return [claims]
    else:
        return []

def get_available_approaches(data):
    """Determine available approaches based on data structure"""
    if not data:
        return []
    
    sample = data[0]
    approaches = []
    
    # Check for baseline approaches
    if 'text_only_results' in sample:
        approaches.extend(['text_only', 'image_text', 'image_text_icl'])
    
    # Check for new approach
    if 'two_stage_results' in sample and 'final_claim' in sample['two_stage_results']:
        approaches.append('two_stage_final_claim')
    
    # Check for my_approach_results
    if 'my_approach_results' in sample:
        approaches.append('my_approach')
    
    return approaches

def get_evaluation_prompts():
    """Load all evaluation prompts from the prompts/evaluation directory"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = os.path.join(os.path.dirname(script_dir), 'prompts', 'evaluation')
    
    prompts = {}
    for filename in os.listdir(prompts_dir):
        if filename.endswith('.md'):
            evaluation_type = filename[:-3]  # Remove .md extension
            prompt_path = os.path.join(prompts_dir, filename)
            with open(prompt_path, 'r') as f:
                prompts[evaluation_type] = f.read()
    
    return prompts

def format_prompt_for_evaluation(evaluation_type: str, prompt_template: str, claim: str, ground_truth: Optional[str] = None, original_text: Optional[str] = None, image_paths: Optional[List[str]] = None):
    """Format prompt based on evaluation type"""
    if evaluation_type == 'reference_based':
        return f"{prompt_template}\n\nGenerated Claim:\n{claim}\nReference Claim:\n{ground_truth}"
    elif evaluation_type == 'entailment':
        # Entailment evaluation needs the original post content (text and image) plus reference claim
        image_info = f"Images: {len(image_paths) if image_paths else 0} images available" if image_paths else "Images: No images available"
        post_content = f"Text: {original_text or 'N/A'}\n{image_info}"
        return f"{prompt_template}\n\n{post_content}\n\nGenerated Claim:\n{claim}"
    elif evaluation_type == 'decontextualization':
        # Decontextualization evaluation only needs the claim itself
        return f"{prompt_template}\n\nGenerated Claim:\n{claim}"
    else:
        raise ValueError(f"Unknown evaluation type: {evaluation_type}")

def evaluate_all_claims_and_find_best(client, model_name, evaluation_type: str, prompt_template: str,
                                    claims: List[str], ground_truth: Optional[str] = None, 
                                    original_text: Optional[str] = None, image_paths: Optional[List[str]] = None):
    """Evaluate all claims and return the best one with its score"""
    if not claims:
        return None, None
    
    if len(claims) == 1:
        # Only one claim, evaluate it directly
        try:
            result = get_llm_evaluation_with_retry(
                client, model_name, evaluation_type, prompt_template,
                claims[0], ground_truth, original_text, image_paths
            )
            return claims[0], result
        except Exception as e:
            print(f"Error evaluating single claim: {str(e)}")
            return None, None
    
    # Multiple claims - evaluate each one and find the best
    best_claim = None
    best_score = -1
    best_result = None
    
    for i, claim in enumerate(claims):
        try:
            result = get_llm_evaluation_with_retry(
                client, model_name, evaluation_type, prompt_template,
                claim, ground_truth, original_text, image_paths
            )
            
            if result is not None:
                # Extract score based on evaluation type
                if evaluation_type == 'reference_based':
                    score = result.score
                elif evaluation_type == 'entailment':
                    # For entailment, convert to numeric score
                    if result.classification == 'entailed':
                        score = 4
                    elif result.classification == 'partially_entailed':
                        score = 3
                    elif result.classification == 'not_entailed':
                        score = 1
                    else:
                        score = 0
                elif evaluation_type == 'decontextualization':
                    # For decontextualization, convert to numeric score
                    if result.classification == 'fully_decontextualized':
                        score = 4
                    elif result.classification == 'partially_decontextualized':
                        score = 3
                    elif result.classification == 'not_decontextualized':
                        score = 1
                    else:
                        score = 0
                else:
                    score = 0
                
                if score > best_score:
                    best_score = score
                    best_claim = claim
                    best_result = result
                    
        except Exception as e:
            print(f"Error evaluating claim {i+1}: {str(e)}")
            continue
    
    return best_claim, best_result

def get_llm_evaluation_with_retry(client, model_name, evaluation_type: str, prompt_template: str, 
                                 claim: str, ground_truth: Optional[str] = None, original_text: Optional[str] = None, 
                                 image_paths: Optional[List[str]] = None):
    """Get LLM evaluation with retry logic"""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Format the prompt based on evaluation type
            prompt = format_prompt_for_evaluation(
                evaluation_type, prompt_template, claim, ground_truth, original_text, image_paths
            )

            # Get the appropriate response schema
            response_schema = EVALUATION_SCHEMAS[evaluation_type]

            # Use the client to generate response
            # For context-dependent evaluations, pass images to the client
            if evaluation_type == 'entailment' and image_paths:
                response = client.generate(
                    prompt,
                    "",  # text_to_extract is empty since the prompt contains everything
                    images=image_paths,
                    response_schema=response_schema,
                    model=model_name
                )
            else:
                response = client.generate(
                    prompt,
                    "",  # text_to_extract is empty since the prompt contains everything
                    response_schema=response_schema,
                    model=model_name
                )
            
            # If response is already a response schema object, return the score
            if isinstance(response, response_schema):
                return response
            else:
                # If response is a string, try to parse it
                if isinstance(response, str):
                    parsed_response = response_schema.model_validate_json(response)
                    return parsed_response
                else:
                    # Handle other response types
                    return response
                    
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg or "quota" in error_msg:
                if attempt < max_retries - 1:
                    tqdm.write(f"Rate limit hit, waiting for 1 minute... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(60)  # Wait for 1 minute
                else:
                    raise
            else:
                if attempt < max_retries - 1:
                    tqdm.write(f"Error occurred, retrying... (Attempt {attempt + 1}/{max_retries}): {str(e)}")
                    time.sleep(2)  # Short delay before retry
                else:
                    raise

def calculate_llm_evaluations(data, client_type: str = "openrouter", model: Optional[str] = None, 
                            checkpoint_path: Optional[str] = None, approach: Optional[str] = None, 
                            evaluation_types: Optional[str] = None):
    # Initialize client based on the specified type
    client, model_name = create_client(client_type, model)
    
    # Load all evaluation prompts
    evaluation_prompts = get_evaluation_prompts()
    
    # Use the single evaluation type specified
    evaluation_type = evaluation_types
    if not evaluation_type:
        raise ValueError("evaluation_types parameter is required")
    if evaluation_type not in evaluation_prompts:
        raise ValueError(f"Evaluation type '{evaluation_type}' not found. Available types: {list(evaluation_prompts.keys())}")
    
    # Determine the field name for storing evaluation results
    evaluation_field_name = get_evaluation_field_name(model_name, evaluation_type)
    
    print(f"Using evaluation type: {evaluation_type}")
    print(f"Using field name: {evaluation_field_name}")
    
    # Load existing results if checkpoint exists
    existing_results = load_checkpoint(checkpoint_path) if checkpoint_path else []
    existing_results_map = {result['article']: result for result in existing_results}
    results = existing_results.copy()
    
    # Determine which approaches to process
    available_approaches = get_available_approaches(data)
    if approach:
        if approach not in available_approaches:
            raise ValueError(f"Approach '{approach}' not available. Available approaches: {available_approaches}")
        approaches_to_process = [approach]
    else:
        approaches_to_process = available_approaches
    
    print(f"Processing approaches: {approaches_to_process}")
    
    for idx, item in enumerate(tqdm(data, desc="Processing LLM evaluations")):
        tqdm.write(f"Processing claim: {item.get('ground_truth_claim', 'unknown')}")
        article = item.get('article', '')
        
        # Check if this article already exists in checkpoint
        existing_item = existing_results_map.get(article)
        if existing_item:
            # Use existing results as base, but we'll still process approaches that need evaluation
            result_entry = existing_item.copy()
        else:
            # Initialize new result entry
            result_entry = {
                'article': item.get('article', ''),
                'ground_truth_claim': item['ground_truth_claim'],
            }
            
            # Add baseline results if they exist
            if 'text_only_results' in item:
                result_entry['text_only_results'] = item['text_only_results']
            if 'image_text_results' in item:
                result_entry['image_text_results'] = item['image_text_results']
            if 'image_text_icl_results' in item:
                result_entry['image_text_icl_results'] = item['image_text_icl_results']
            
            # Add new approach results if they exist
            if 'two_stage_results' in item:
                result_entry['two_stage_results'] = item['two_stage_results']
            
            # Add my_approach results if they exist
            if 'my_approach_results' in item:
                result_entry['my_approach_results'] = item['my_approach_results']
        
        # Initialize evaluation scores for the specific evaluation type within each approach
        for approach_name in approaches_to_process:
            if approach_name == 'text_only' and 'text_only_results' in result_entry:
                if evaluation_field_name not in result_entry['text_only_results']:
                    result_entry['text_only_results'][evaluation_field_name] = None
            elif approach_name == 'image_text' and 'image_text_results' in result_entry:
                if evaluation_field_name not in result_entry['image_text_results']:
                    result_entry['image_text_results'][evaluation_field_name] = None
            elif approach_name == 'image_text_icl' and 'image_text_icl_results' in result_entry:
                if evaluation_field_name not in result_entry['image_text_icl_results']:
                    result_entry['image_text_icl_results'][evaluation_field_name] = None
            elif approach_name == 'two_stage_final_claim' and 'two_stage_results' in result_entry:
                if evaluation_field_name not in result_entry['two_stage_results']:
                    result_entry['two_stage_results'][evaluation_field_name] = None
            elif approach_name == 'my_approach' and 'my_approach_results' in result_entry:
                if evaluation_field_name not in result_entry['my_approach_results']:
                    result_entry['my_approach_results'][evaluation_field_name] = None
        
        # Extract common data once (same for all approaches)
        # The full_text field already contains metadata for baseline approaches
        original_text = item.get('full_text', '')
        
        # Get image paths from image_links field (only for non-reference-based evaluations)
        image_paths: List[str] = []
        if evaluation_type != 'reference_based':
            if 'image_links' in item:
                # Get the images directory path for constructing full paths
                script_dir = os.path.dirname(os.path.abspath(__file__))
                images_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), 'data/averimatec_with_fulltext/all_images')
                
                tqdm.write(f"Processing {len(item['image_links'])} images for claim: {item.get('ground_truth_claim', 'unknown')}")
                for img_name in item['image_links']:
                    img_path = os.path.join(images_dir, img_name)
                    if os.path.exists(img_path):
                        image_paths.append(img_path)
                        tqdm.write(f"  ✓ Found image: {img_name}")
                    else:
                        tqdm.write(f"  ✗ Warning: Image not found: {img_path}")
            else:
                tqdm.write(f"No image_links field found for claim: {item.get('ground_truth_claim', 'unknown')}")
        
        # Process each approach for this claim
        for approach_name in approaches_to_process:
            all_claims = get_all_claims(item, approach_name)
            
            if not all_claims:
                tqdm.write(f"    Skipping {approach_name} - no valid claims found")
                continue
            
            tqdm.write(f"    Processing {approach_name} with {len(all_claims)} claim(s)")
            
            # Process the specific evaluation type for this claim
            # Determine where to store the evaluation result
            if approach_name == 'text_only':
                result_location = result_entry['text_only_results']
            elif approach_name == 'image_text':
                result_location = result_entry['image_text_results']
            elif approach_name == 'image_text_icl':
                result_location = result_entry['image_text_icl_results']
            elif approach_name == 'two_stage_final_claim':
                result_location = result_entry['two_stage_results']
            elif approach_name == 'my_approach':
                result_location = result_entry['my_approach_results']
            else:
                tqdm.write(f"Unknown approach: {approach_name}")
                continue
            
            # Skip if evaluation already calculated
            if evaluation_field_name in result_location and result_location[evaluation_field_name] is not None:
                continue
            
            try:
                # Use the same text for evaluation for all approaches
                evaluation_text = original_text
                
                # Evaluate all claims and find the best one
                best_claim, evaluation_result = evaluate_all_claims_and_find_best(
                    client, model_name, evaluation_type, evaluation_prompts[evaluation_type],
                    all_claims, item['ground_truth_claim'], evaluation_text, image_paths
                )
                
                if evaluation_result is not None and best_claim is not None:
                    # Save the evaluation result
                    result_location[evaluation_field_name] = evaluation_result.model_dump()
                    
                    # Replace the claims array with just the best claim (as a string)
                    if approach_name in ['text_only', 'image_text', 'image_text_icl']:
                        result_location['claims'] = best_claim
                    elif approach_name == 'my_approach':
                        # For my_approach, update the extracted_claim field
                        result_location['extracted_claim'] = best_claim
                        # Also update extracted_claims to be consistent
                        result_location['extracted_claims'] = [best_claim]
                    
                    tqdm.write(f"    ✓ Successfully evaluated {evaluation_type} for {approach_name}")
                    tqdm.write(f"      Best claim selected: {best_claim[:100]}...")
                    if len(all_claims) > 1:
                        tqdm.write(f"      (Evaluated {len(all_claims)} claims, selected the highest scoring one)")
                else:
                    tqdm.write(f"    ✗ No valid evaluation result for {approach_name}")
            except Exception as e:
                tqdm.write(f"    ✗ Error evaluating {evaluation_type} for {approach_name}: {str(e)}")
        
        # Only append if this is a new item (not from existing checkpoint)
        if not existing_item:
            results.append(result_entry)
        
        if checkpoint_path:
            save_checkpoint(checkpoint_path, results)
            tqdm.write(f"✓ Checkpoint saved after processing item {idx + 1}/{len(data)}")
    
    # Compute final results for the specific evaluation type
    def compute_evaluation_metrics(eval_type: str, approach_name: str):
        if eval_type == 'reference_based':
            # Numeric score analysis for reference-based evaluation
            scores = []
            for result in results:
                # Determine where to look for the evaluation result
                if approach_name == 'text_only' and 'text_only_results' in result:
                    result_location = result['text_only_results']
                elif approach_name == 'image_text' and 'image_text_results' in result:
                    result_location = result['image_text_results']
                elif approach_name == 'image_text_icl' and 'image_text_icl_results' in result:
                    result_location = result['image_text_icl_results']
                elif approach_name == 'two_stage_final_claim' and 'two_stage_results' in result:
                    result_location = result['two_stage_results']
                elif approach_name == 'my_approach' and 'my_approach_results' in result:
                    result_location = result['my_approach_results']
                else:
                    continue
                
                # Use the dynamic field name for looking up results
                if evaluation_field_name in result_location and result_location[evaluation_field_name] is not None:
                    score = result_location[evaluation_field_name].get('score', 0)
                    scores.append(score)
            
            if not scores:
                return {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0
                }
            
            return {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores))
            }
        else:
            # Categorical analysis for entailment and decontextualization evaluation
            classifications = []
            for result in results:
                # Determine where to look for the evaluation result
                if approach_name == 'text_only' and 'text_only_results' in result:
                    result_location = result['text_only_results']
                elif approach_name == 'image_text' and 'image_text_results' in result:
                    result_location = result['image_text_results']
                elif approach_name == 'image_text_icl' and 'image_text_icl_results' in result:
                    result_location = result['image_text_icl_results']
                elif approach_name == 'two_stage_final_claim' and 'two_stage_results' in result:
                    result_location = result['two_stage_results']
                elif approach_name == 'my_approach' and 'my_approach_results' in result:
                    result_location = result['my_approach_results']
                else:
                    continue
                
                # Use the dynamic field name for looking up results
                if evaluation_field_name in result_location and result_location[evaluation_field_name] is not None:
                    classification = result_location[evaluation_field_name].get('classification', 'not_entailed' if eval_type == 'entailment' else 'not_decontextualized')
                    classifications.append(classification)
            
            if not classifications:
                if eval_type == 'entailment':
                    return {
                        'total_count': 0,
                        'entailed_count': 0,
                        'partially_entailed_count': 0,
                        'not_entailed_count': 0,
                        'entailed_percentage': 0.0,
                        'partially_entailed_percentage': 0.0,
                        'not_entailed_percentage': 0.0
                    }
                else:  # decontextualization
                    return {
                        'total_count': 0,
                        'fully_decontextualized_count': 0,
                        'partially_decontextualized_count': 0,
                        'not_decontextualized_count': 0,
                        'fully_decontextualized_percentage': 0.0,
                        'partially_decontextualized_percentage': 0.0,
                        'not_decontextualized_percentage': 0.0
                    }
            
            total = len(classifications)
            
            if eval_type == 'entailment':
                entailed_count = classifications.count('entailed')
                partially_entailed_count = classifications.count('partially_entailed')
                not_entailed_count = classifications.count('not_entailed')
                
                return {
                    'total_count': total,
                    'entailed_count': entailed_count,
                    'partially_entailed_count': partially_entailed_count,
                    'not_entailed_count': not_entailed_count,
                    'entailed_percentage': round((entailed_count / total) * 100, 2),
                    'partially_entailed_percentage': round((partially_entailed_count / total) * 100, 2),
                    'not_entailed_percentage': round((not_entailed_count / total) * 100, 2)
                }
            else:  # decontextualization
                fully_decontextualized_count = classifications.count('fully_decontextualized')
                partially_decontextualized_count = classifications.count('partially_decontextualized')
                not_decontextualized_count = classifications.count('not_decontextualized')
                
                return {
                    'total_count': total,
                    'fully_decontextualized_count': fully_decontextualized_count,
                    'partially_decontextualized_count': partially_decontextualized_count,
                    'not_decontextualized_count': not_decontextualized_count,
                    'fully_decontextualized_percentage': round((fully_decontextualized_count / total) * 100, 2),
                    'partially_decontextualized_percentage': round((partially_decontextualized_count / total) * 100, 2),
                    'not_decontextualized_percentage': round((not_decontextualized_count / total) * 100, 2)
                }
    
    # Build final results
    final_results: Dict[str, Any] = {
        'detailed_results': results
    }
    
    # Add metrics for the specific evaluation type
    final_results[f'{evaluation_type}_metrics'] = {}
    for approach_name in approaches_to_process:
        final_results[f'{evaluation_type}_metrics'][approach_name] = compute_evaluation_metrics(evaluation_type, approach_name)
    
    return final_results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run LLM evaluations')
    parser.add_argument('--client', type=str, choices=['openrouter'], 
                       default='openrouter', help='Choose the LLM client to use')
    parser.add_argument('--model', type=str, default=None,
                       help='Specify the model to use (e.g., "google/gemini-2.0-flash-001" for OpenRouter)')
    parser.add_argument('--file', type=str, required=True,
                       help='Path to results file (JSON)')
    parser.add_argument('--approach', type=str, default=None,
                       help='Specific approach to evaluate (text_only, image_text, image_text_icl, two_stage_final_claim, my_approach)')
    parser.add_argument('--eval', type=str, choices=['reference_based', 'entailment', 'decontextualization'], 
                       required=True, help='Specific evaluation type to run')
    args = parser.parse_args()
    
    client_type = args.client
    model = args.model
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Determine data path
    data_path = args.file
    
    approach_name = os.path.splitext(os.path.basename(args.file))[0]
    eval_type = args.eval
    checkpoint_filename = f'llm_eval_{eval_type}_{approach_name}.json'
    
    checkpoint_path = os.path.join(os.path.dirname(script_dir), 'evaluation_results', checkpoint_filename)
    
    data = load_data(data_path)
    
    # Calculate LLM evaluations
    print(f"Using {client_type.upper()} client with model: {model or 'default'}")
    print(f"Data file: {data_path}")
    print(f"Checkpoint will be saved to: {checkpoint_path}")
    print("Processing LLM evaluations...")
    
    results = calculate_llm_evaluations(
        data, 
        client_type=client_type, 
        model=model, 
        checkpoint_path=checkpoint_path,
        approach=args.approach,
        evaluation_types=args.eval
    )

    print(f"Checkpoint saved to: {checkpoint_path}")

if __name__ == "__main__":
    main()