import os
import requests
import base64
import time
from datetime import datetime, timedelta
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(override=True)

class OpenRouterClient:
    def __init__(self, rate_limit: int = None):  # Make rate_limit optional
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or pass it to the constructor.")
        
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.rate_limit = rate_limit
        self.request_times = []
    
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def generate(self, prompt: str, text_to_extract: Optional[str], images: Optional[List[str]] = None, examples: Optional[List[dict]] = None, response_schema: Optional[BaseModel] = None, model: str = "google/gemini-2.0-flash-001") -> Union[str, BaseModel]:
        """Generate response from prompt and optional images, with optional examples and their corresponding images"""
        
        # Handle multi-modal input limits for Qwen models
        if "qwen" in model.lower():
            # Qwen models have a 30-image limit per conversation
            print(f"📊 IMAGE COUNT ANALYSIS FOR QWEN:")
            print(f"  Main claim images: {len(images) if images else 0}")
            if examples:
                for i, example in enumerate(examples):
                    example_images = len(example.get('images', [])) if 'images' in example else 0
                    print(f"  ICL Example {i+1} images: {example_images}")
            
            # Count total images
            total_images = 0
            if images:
                total_images += len(images)
            if examples:
                for example in examples:
                    if 'images' in example and example['images']:
                        total_images += len(example['images'])
            
            print(f"  Total images: {total_images}")
            
            if total_images > 30:
                print(f"Warning: Total images ({total_images}) exceeds 30-image limit for Qwen model. Reducing ICL examples.")
                
                # Calculate available slots for images (30 total)
                available_image_slots = 30
                
                # First, limit main images if needed
                main_images_count = len(images) if images else 0
                remaining_slots = available_image_slots
                
                if main_images_count > remaining_slots:
                    images = images[:remaining_slots]
                    print(f"Limited main images to {remaining_slots}")
                    remaining_slots = 0
                else:
                    remaining_slots -= main_images_count
                
                # Then, limit ICL example images to fit remaining slots
                if examples and remaining_slots > 0:
                    limited_examples = []
                    for example in examples:
                        if remaining_slots <= 0:
                            # No more slots, remove images from this example
                            limited_example = example.copy()
                            limited_example['images'] = []
                            limited_examples.append(limited_example)
                        else:
                            # Limit this example's images to fit remaining slots
                            limited_example = example.copy()
                            if 'images' in example and example['images']:
                                limited_example['images'] = example['images'][:remaining_slots]
                                remaining_slots -= len(limited_example['images'])
                            limited_examples.append(limited_example)
                    examples = limited_examples
                    print(f"Limited ICL example images to fit within {available_image_slots} total image slots")
                elif examples:
                    # No remaining slots, remove all ICL example images
                    limited_examples = []
                    for example in examples:
                        limited_example = example.copy()
                        limited_example['images'] = []
                        limited_examples.append(limited_example)
                    examples = limited_examples
                    print(f"Removed all images from {len(examples)} ICL examples (no remaining slots)")
        
        # Handle structured output differently for different models
        if response_schema:
            if "gpt-4o-mini" in model:
                payload = {
                    "model": model,
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"}
                }
            else:
                payload = {
                    "model": model,
                    "temperature": 0.0,
                    "response_format": {
                        "type": "json_object",
                        "schema": response_schema.model_json_schema()
                    }
                }
        else:
            payload = {
                "model": model,
                "temperature": 0.0
            }
        
        # Build content structure for all models
        contents = [{"type": "text", "text": prompt}]

        if examples:
            for example in examples:
                # Add example text as a separate content part
                contents.append({"type": "text", "text": example['text']})
                
                # Add example images as separate content parts
                if 'images' in example and example['images']:
                    for img_path in example['images']:
                        image_data = self._encode_image(img_path)
                        contents.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        })
                
                # Add example claim_text as a separate content part
                contents.append({"type": "text", "text": example['claim_text']})

        # Only add text_to_extract if it's not None or empty
        if text_to_extract:
            contents.append({"type": "text", "text": "# INPUT (SOCIAL MEDIA POST)\n" + text_to_extract})
        
        # Add images as separate content parts
        if images:
            for img_path in images:
                image_data = self._encode_image(img_path)
                contents.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                })
        
        # Print contents in a more readable format
        # print("CONTENTS:")
        # for i, content in enumerate(contents):
        #     if "text" in content:
        #         print(f"  [{i}] TEXT: {content['text'][:100]}{'...' if len(content['text']) > 100 else ''}")
        #     elif "image_url" in content:
        #         data_length = len(content['image_url']['url'])
        #         print(f"  [{i}] IMAGE: {content['image_url']['url'][:50]}... ({data_length} chars)")
        # print()
        
        # Create the single user message with all content parts
        payload["messages"] = [{
            "role": "user",
            "content": contents
        }]
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"API call failed: {response.text}")
        
        response_json = response.json()
        
        if "choices" not in response_json:
            raise Exception(f"Unexpected response structure. Available keys: {list(response_json.keys())}. Response: {response_json}")
        
        content = response_json["choices"][0]["message"]["content"]
        
        # Parse response
        if response_schema:
            try:
                return response_schema.model_validate_json(content)
            except Exception as e:
                print(f"Error parsing structured output: {str(e)}")
                print(f"Response content: {content}")
                
                # Try to clean the JSON by removing problematic escape characters
                try:
                    import json
                    import re
                    
                    # Clean the content by removing problematic escape sequences
                    cleaned_content = content
                    
                    # First, try to fix common JSON issues
                    # Remove any characters that are not properly escaped
                    cleaned_content = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', '', cleaned_content)
                    
                    # Fix unescaped quotes in the reasoning field
                    # Look for the reasoning field and clean it
                    reasoning_pattern = r'"reasoning":\s*"([^"]*(?:\\.[^"]*)*)"'
                    reasoning_match = re.search(reasoning_pattern, cleaned_content)
                    
                    if reasoning_match:
                        # Extract the reasoning text and clean it
                        reasoning_text = reasoning_match.group(1)
                        # Remove all escape characters and the characters they were escaping
                        cleaned_reasoning = re.sub(r'\\.', '', reasoning_text)
                        # Replace the reasoning field with cleaned version
                        cleaned_content = re.sub(
                            reasoning_pattern, 
                            f'"reasoning": "{cleaned_reasoning}"', 
                            cleaned_content
                        )
                    
                    # Try to parse as JSON first to validate
                    json.loads(cleaned_content)
                    
                    # If JSON parsing succeeds, try with the schema
                    return response_schema.model_validate_json(cleaned_content)
                    
                except Exception as clean_error:
                    print(f"Failed to clean JSON: {str(clean_error)}")
                    # If cleaning fails, try to extract just the essential parts
                    try:
                        # Look for score and reasoning in the content
                        import re
                        score_match = re.search(r'"score":\s*(\d+)', content)
                        
                        # Try multiple patterns for reasoning
                        reasoning = ""
                        reasoning_patterns = [
                            r'"reasoning":\s*"([^"]*(?:\\.[^"]*)*)"',  # Original pattern
                            r'"reasoning":\s*"([^"]*)"',  # Simple pattern without escape handling
                            r'"reasoning":\s*"([^"]*?)(?="\s*[,}])',  # Non-greedy pattern
                        ]
                        
                        for pattern in reasoning_patterns:
                            reasoning_match = re.search(pattern, content)
                            if reasoning_match:
                                reasoning = reasoning_match.group(1)
                                # Clean the reasoning text by removing escape characters
                                reasoning = re.sub(r'\\.', '', reasoning)
                                break
                        
                        if score_match and reasoning:
                            score = int(score_match.group(1))
                            
                            # Create a minimal valid JSON
                            minimal_json = f'{{"score": {score}, "reasoning": "{reasoning}"}}'
                            return response_schema.model_validate_json(minimal_json)
                        else:
                            raise clean_error
                    except Exception as fallback_error:
                        print(f"Fallback parsing also failed: {str(fallback_error)}")
                        raise e  # Raise original error
        else:
            return content