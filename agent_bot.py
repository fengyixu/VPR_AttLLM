"""
Try guide create simple nodes (with only feature discription first) and link them first, then refine the nodes and edges properties.
"""

import os
import base64
from typing import List, Optional, Union, Dict, Any
from openai import OpenAI
from pathlib import Path
import time
import threading
import logging
from abc import ABC, abstractmethod
from PIL import Image
from io import BytesIO
import re
import random

logger = logging.getLogger(__name__)

class RateController:
    """
    Rate controller for managing API calls per minute (QPM).
    Ensures we don't exceed the specified QPM limit.
    """
    
    def __init__(self, max_qpm: int = 60):
        """
        Initialize the rate controller.
        
        Args:
            max_qpm: Maximum queries per minute allowed
        """
        self.max_qpm = max_qpm
        self.min_interval = 60.0 / max_qpm  # Minimum time between requests in seconds
        self.last_request_time = 0
        self.lock = threading.Lock()
        self.request_count = 0
        self.window_start = time.time()
    
    def wait_if_needed(self):
        """
        Wait if necessary to respect the QPM limit.
        Should be called before each API request.
        """
        with self.lock:
            current_time = time.time()
            
            # Reset counter if a minute has passed
            if current_time - self.window_start >= 60.0:
                self.request_count = 0
                self.window_start = current_time
            
            # Check if we've hit the limit for this window
            if self.request_count >= self.max_qpm:
                # Wait until the next minute window
                wait_time = 60.0 - (current_time - self.window_start)
                if wait_time > 0:
                    logger.info(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    # Reset for new window
                    self.request_count = 0
                    self.window_start = time.time()
            
            # Ensure minimum interval between requests
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()
            self.request_count += 1
    
    def get_status(self) -> Dict[str, Any]:
        """Get current rate controller status."""
        current_time = time.time()
        return {
            'max_qpm': self.max_qpm,
            'current_requests_in_window': self.request_count,
            'time_remaining_in_window': max(0, 60.0 - (current_time - self.window_start)),
            'min_interval_between_requests': self.min_interval
        }

class AgentBot(ABC):
    """
    Abstract base class for AI agent bots with unified interface.
    
    Provides a common interface for different AI models (Qwen, Gemini, etc.)
    while maintaining conversation memory and image support.
    """
    
    def __init__(self, model: str, system_prompt: str = "You are a helpful AI assistant."):
        """
        Initialize AgentBot.
        
        Args:
            model: Model name/identifier
            system_prompt: System message for conversation context
        """
        self.model = model
        self.system_prompt = system_prompt
        self.conversation_history = []
        self._token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "thinking_tokens": 0, "total_tokens": 0, "api_calls": 0}
        self._add_system_message()
    
    def _add_system_message(self) -> None:
        """Add system message to conversation history."""
        if self.system_prompt:
            self.conversation_history = [
                {"role": "system", "content": self.system_prompt}
            ]
    
    def _process_images(self, image_paths: List[Any], max_size: int = 1024, resize = False) -> List[Dict[str, Any]]:
        """
        Convert local image files or PIL images to base64 data URLs, resizing longer edge to max_size.
        
        Args:
            image_paths: List of local file paths or PIL Image objects
            max_size: Maximum size of the longer image edge (pixels)
            
        Returns:
            List of image content dictionaries
        """
        image_contents = []
        for path in image_paths:
            if isinstance(path, Image.Image):
                img = path.convert("RGB")
            else:
                p = Path(path)
                if not p.exists():
                    raise FileNotFoundError(f"Image file not found: {path}")
                with Image.open(p) as im:
                    img = im.convert("RGB")
            w, h = img.size
            longer = max(w, h)
            if longer > max_size and resize:
                scale = max_size / float(longer)
                new_size = (int(round(w * scale)), int(round(h * scale)))
                img = img.resize(new_size, Image.LANCZOS)
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=90)
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            image_contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}
            })
        return image_contents
    
    @abstractmethod
    def _make_api_call(self, messages: List[Dict[str, Any]], max_tokens: Optional[int] = None, temperature: float = 0.7, thinking_budget_tokens: Optional[int] = None) -> str:
        """
        Abstract method to make API call to the specific model.
        
        Args:
            messages: List of conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            thinking_budget_tokens: Thinking budget tokens for internal reasoning (optional, model-specific)
            
        Returns:
            Model response text
        """
        pass
    
    def count_tokens(self, text_prompt: str, image_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Count tokens for text and images using local tokenizer.
        
        Args:
            text_prompt: Text prompt
            image_paths: Optional list of image paths
            
        Returns:
            Dictionary containing token count information
        """
        try:
            from transformers import AutoTokenizer
            
            # Use appropriate tokenizer based on model
            if "qwen" in self.model.lower():
                # Qwen models typically use similar tokenization to GPT models
                tokenizer_name = "microsoft/DialoGPT-medium"
            elif "gemini" in self.model.lower():
                # Gemini models use Gemma tokenizer
                tokenizer_name = "microsoft/DialoGPT-medium"
            else:
                # Default to a general-purpose tokenizer
                tokenizer_name = "gpt2"
            
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            
            # Count text tokens
            text_tokens = len(tokenizer.encode(text_prompt))
            
            # Estimate image tokens (rough approximation)
            image_tokens = 0
            if image_paths:
                # Rough estimate: each image typically consumes ~1000-2000 tokens
                # This varies by image size and complexity
                image_tokens = len(image_paths) * 1500
            
            total_tokens = text_tokens + image_tokens
            
            return {
                "total_tokens": total_tokens,
                "text_tokens": text_tokens,
                "image_tokens": image_tokens,
                "model": self.model,
                "tokenizer_used": tokenizer_name,
                "note": "Approximate count using local tokenizer"
            }
            
        except ImportError:
            # Fallback to simple word count if transformers not available
            text_tokens = len(text_prompt.split())
            image_tokens = len(image_paths) * 1500 if image_paths else 0
            total_tokens = text_tokens + image_tokens
            
            return {
                "total_tokens": total_tokens,
                "text_tokens": text_tokens,
                "image_tokens": image_tokens,
                "model": self.model,
                "tokenizer_used": "word_count_fallback",
                "note": "Fallback count - install transformers for better accuracy"
            }
            
        except Exception as e:
            # Final fallback
            text_tokens = len(text_prompt.split())
            image_tokens = len(image_paths) * 1500 if image_paths else 0
            total_tokens = text_tokens + image_tokens
            
            return {
                "total_tokens": total_tokens,
                "text_tokens": text_tokens,
                "image_tokens": image_tokens,
                "model": self.model,
                "tokenizer_used": "simple_fallback",
                "error": str(e),
                "note": "Simple fallback count due to error"
            }

    def _accumulate_usage(self, usage_dict: Dict[str, int]) -> None:
        """Accumulate token usage from a single API call."""
        for key in ("prompt_tokens", "completion_tokens", "thinking_tokens", "total_tokens"):
            self._token_usage[key] += usage_dict.get(key, 0)
        self._token_usage["api_calls"] += 1

    def get_usage_summary(self) -> Dict[str, int]:
        """Return cumulative token usage across all API calls."""
        return dict(self._token_usage)

    def reset_usage(self) -> None:
        """Reset cumulative token usage counters."""
        self._token_usage = {k: 0 for k in self._token_usage}

    def chat(self, text_prompt: str, image_paths: Optional[List[str]] = None, preserve_image_order: bool = False, max_tokens: Optional[int] = None, temperature: float = 0.7,
        thinking_budget_tokens: Optional[int] = None) -> str:
        """
        Send a chat message with optional images.
        
        Args:
            text_prompt: Text prompt/question. When preserve_image_order=True, 
                        use {IMAGE_0}, {IMAGE_1}, etc. markers for image placement
            image_paths: Optional list of image file paths or URLs
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            thinking_budget_tokens: Thinking budget tokens for internal reasoning (Gemini-specific)
            preserve_image_order: If True, respects {IMAGE_X} markers in text for precise image placement
            
        Returns:
            Model response text
        """
        content = []
        
        if preserve_image_order and image_paths and "{IMAGE_" in text_prompt:
            # Enhanced mode: parse text and insert images at marker positions
            content = self._build_content_with_markers(text_prompt, image_paths)
        else:
            # Original mode: all images first, then text
            if image_paths:
                content.extend(self._process_images(image_paths))
            content.append({"type": "text", "text": text_prompt})
        
        # Add user message to history
        user_message = {"role": "user", "content": content}
        self.conversation_history.append(user_message)
        
        try:
            # Make API call using the specific implementation
            assistant_response = self._make_api_call(
                self.conversation_history, 
                max_tokens, 
                temperature,
                thinking_budget_tokens
            )
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant", 
                "content": assistant_response
            })
            
            return assistant_response
            
        except Exception as e:
            raise RuntimeError(f"API call failed: {str(e)}")

    def _build_content_with_markers(self, text_prompt: str, image_paths: List[str]) -> List[dict]:
        """
        Build content array with images inserted at marker positions.
        Supports both individual markers {IMAGE_5} and range markers {IMAGE_2-11}.
        
        Args:
            text_prompt: Text with {IMAGE_X} or {IMAGE_X-Y} markers
            image_paths: List of image paths corresponding to markers
            
        Returns:
            List of content dictionaries with proper image/text ordering
        """
        content = []
        
        def expand_ranges(text):
            """Expand range markers like {IMAGE_2-11} to individual markers"""
            def replace_range(match):
                start, end = int(match.group(1)), int(match.group(2))
                return ' '.join(f'{{IMAGE_{i}}}' for i in range(start, end + 1))
            
            return re.sub(r'\{IMAGE_(\d+)-(\d+)\}', replace_range, text)
        
        # First expand any range markers
        expanded_text = expand_ranges(text_prompt)
        
        # Split by individual image markers
        parts = re.split(r'(\{IMAGE_\d+\})', expanded_text)
        
        for part in parts:
            if not part:
                continue
                
            marker_match = re.match(r'\{IMAGE_(\d+)\}', part)
            if marker_match:
                image_index = int(marker_match.group(1))
                if 0 <= image_index < len(image_paths):
                    # debugging use: directly add image path
                    # content.extend([{"type": "image_url", "image_url": {"url": image_paths[image_index]}}])
                    content.extend(self._process_images([image_paths[image_index]]))
            else:
                if part.strip():
                    content.append({"type": "text", "text": part})
        
        return content

    # def chat_old(
    #     self, 
    #     text_prompt: str, 
    #     image_paths: Optional[List[str]] = None,
    #     max_tokens: Optional[int] = None,
    #     temperature: float = 0.7,
    #     thinking_budget_tokens: Optional[int] = None
    # ) -> str:
    #     """
    #     Send a chat message with optional images.
        
    #     Args:
    #         text_prompt: Text prompt/question
    #         image_paths: Optional list of image file paths or URLs
    #         max_tokens: Maximum tokens to generate
    #         temperature: Sampling temperature
    #         thinking_budget_tokens: Thinking budget tokens for internal reasoning (Gemini-specific)
            
    #     Returns:
    #         Model response text
    #     """
    #     # Prepare message content
    #     content = []
        
    #     # Add images if provided
    #     if image_paths:
    #         content.extend(self._process_images(image_paths))
        
    #     # Add text prompt
    #     content.append({"type": "text", "text": text_prompt})
        
    #     # Add user message to history
    #     user_message = {"role": "user", "content": content}
    #     self.conversation_history.append(user_message)
        
    #     try:
    #         # Make API call using the specific implementation
    #         assistant_response = self._make_api_call(
    #             self.conversation_history, 
    #             max_tokens, 
    #             temperature,
    #             thinking_budget_tokens
    #         )
            
    #         # Add assistant response to history
    #         self.conversation_history.append({
    #             "role": "assistant", 
    #             "content": assistant_response
    #         })
            
    #         return assistant_response
            
    #     except Exception as e:
    #         raise RuntimeError(f"API call failed: {str(e)}")

    def chat_image(self,image_path: Any, resize = False, prompt_list: List[str] = None,
        max_tokens: Optional[int] = None, temperature: float = 0.7, thinking_budget_tokens: Optional[int] = None) -> List[str]:
        """
        Process multiple prompts sequentially on a single image while maintaining conversation context.
        
        This function:
        1. Processes each prompt one by one
        2. After each response, removes the user message (current prompt) from chat history
        3. Keeps the assistant response in history for context
        4. Maintains the original image throughout all prompts
        5. Returns a list of responses corresponding to each prompt
        
        Args:
            image_path: Path to the image file, URL, or PIL Image
            prompt_list: List of text prompts to process sequentially
            max_tokens: Maximum tokens to generate per response
            temperature: Sampling temperature for responses
            thinking_budget_tokens: Thinking budget tokens for internal reasoning (Gemini-specific)
            
        Returns:
            List of responses, one for each prompt in prompt_list
            
        Raises:
            RuntimeError: If any API call fails during processing
        """
        responses = []
        
        # Process each prompt sequentially
        for i, prompt in enumerate(prompt_list):
            try:
                # Create message content with image and current prompt
                content = []
                
                # Add image (same image for all prompts)
                content.extend(self._process_images([image_path]))
                
                # Add current prompt
                content.append({"type": "text", "text": prompt})
                
                # Add user message to history
                user_message = {"role": "user", "content": content}
                self.conversation_history.append(user_message)
                
                # Make API call
                assistant_response = self._make_api_call(
                    self.conversation_history,
                    max_tokens,
                    temperature,
                    thinking_budget_tokens
                )
                
                # Remove the user message (current prompt) from history
                self.conversation_history.pop()
                
                # Add assistant response to history for context
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_response
                })
                
                responses.append(assistant_response)
                
            except Exception as e:
                # Clean up: remove the user message if it was added
                if self.conversation_history and self.conversation_history[-1]["role"] == "user":
                    self.conversation_history.pop()
                raise RuntimeError(f"API call failed for prompt {i+1}: {str(e)}")
        
        return responses

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get current conversation history.
        
        Returns:
            List of conversation messages
        """
        return self.conversation_history.copy()
    
    def clear_conversation(self) -> None:
        """Clear conversation history and restart with system message."""
        self.conversation_history = []
        self._add_system_message()
    
    def set_system_prompt(self, prompt: str) -> None:
        """
        Update system prompt and restart conversation.
        
        Args:
            prompt: New system prompt
        """
        self.system_prompt = prompt
        self.clear_conversation()
    
    def save_conversation(self, filepath: str) -> None:
        """
        Save conversation history to JSON file.
        
        Args:
            filepath: Path to save conversation
        """
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
    
    def load_conversation(self, filepath: str) -> None:
        """
        Load conversation history from JSON file.
        
        Args:
            filepath: Path to conversation file
        """
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            self.conversation_history = json.load(f)

class QwenAgent(AgentBot):
    """
    QwenAgent implementation for vision-language conversations.
    
    Supports local/remote images, conversation memory, and easy model switching.
    Requires DASHSCOPE_API_KEY environment variable to be set.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "qwen3.5-plus",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        system_prompt: str = "You are a helpful AI assistant."
    ):
        """
        Initialize QwenAgent.
        
        Args:
            api_key: Dashscope API key. If None, reads from DASHSCOPE_API_KEY env var.
            model: Model name (qwen-vl-max, qwen3.5-plus, etc.)
            base_url: API endpoint base URL
            system_prompt: System message for conversation context
        """
        super().__init__(model, system_prompt)
        resolved_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Dashscope API key not provided. Set the DASHSCOPE_API_KEY environment variable "
                "or pass api_key= explicitly."
            )
        self.client = OpenAI(
            api_key=resolved_key,
            base_url=base_url
        )
    
    def _make_api_call(self, messages: List[Dict[str, Any]], max_tokens: Optional[int] = None, temperature: float = 0.7, thinking_budget_tokens: Optional[int] = None) -> str:
        """
        Make API call to Qwen model.
        
        Args:
            messages: List of conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            thinking_budget_tokens: Thinking budget tokens for internal reasoning (not used by Qwen)
            
        Returns:
            Model response text
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        if hasattr(response, "usage") and response.usage:
            u = response.usage
            self._accumulate_usage({
                "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
                "total_tokens": getattr(u, "total_tokens", 0) or 0,
            })

        return response.choices[0].message.content

class GeminiAgent(AgentBot):
    """
    GeminiAgent implementation for vision-language conversations.
    Supports local/remote images, conversation memory, and easy model switching.
    Requires GEMINI_API_KEY environment variable to be set.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        system_prompt: str = "You are a helpful AI assistant.",
        max_output_tokens: int = 65536,
        top_p: float = 0.95,
        thinking_budget_tokens: int = 1024
    ):
        """
        Initialize GeminiAgent.
        
        Args:
            api_key: Google Gemini API key. If None, reads from GEMINI_API_KEY env var.
            model: Model name (gemini-2.0-flash, gemini-2.5-flash, etc.)
            system_prompt: System message for conversation context
            max_output_tokens: Maximum output tokens (default 65536)
            top_p: Nucleus sampling probability (default 0.95)
            thinking_budget_tokens: Thinking budget tokens for internal reasoning (default 1024)
        """
        super().__init__(model, system_prompt)
        try:
            from google import genai
            from google.genai import types
            self.genai = genai
            self.types = types
        except ImportError:
            raise ImportError("Google GenAI library not found. Install with: pip install google-genai")
        
        resolved_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Gemini API key not provided. Set the GEMINI_API_KEY environment variable "
                "or pass api_key= explicitly."
            )
        self.client = genai.Client(api_key=resolved_key)
        self.max_output_tokens = max_output_tokens
        self.top_p = top_p
        self.thinking_budget_tokens = thinking_budget_tokens
    
    def _convert_messages_to_gemini_format(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert OpenAI-style messages to Gemini format.
        
        Args:
            messages: List of OpenAI-style conversation messages
            
        Returns:
            List of Gemini-formatted messages
        """
        gemini_messages = []
        
        for message in messages:
            if message["role"] == "system":
                # Gemini doesn't have system messages, so we'll prepend to the first user message
                continue
            
            if message["role"] == "user":
                # Process user message content
                if isinstance(message["content"], list):
                    # Multi-modal content (text + images)
                    parts = []
                    for item in message["content"]:
                        if item["type"] == "text":
                            parts.append({"text": item["text"]})
                        elif item["type"] == "image_url":
                            if item["image_url"]["url"].startswith("data:"):
                                # Base64 image
                                parts.append({"inline_data": {"mime_type": "image/jpeg", "data": item["image_url"]["url"].split(",")[1]}})
                            else:
                                # URL image
                                parts.append({"file_data": {"mime_type": "image/jpeg", "file_uri": item["image_url"]["url"]}})
                    gemini_messages.append({"role": "user", "parts": parts})
                else:
                    # Text-only content
                    gemini_messages.append({"role": "user", "parts": [{"text": message["content"]}]})
            
            elif message["role"] == "assistant":
                # Assistant message
                gemini_messages.append({"role": "model", "parts": [{"text": message["content"]}]})
        
        return gemini_messages
    
    def _make_api_call(self, messages: List[Dict[str, Any]], max_tokens: Optional[int] = None, temperature: float = 0.7, thinking_budget_tokens: Optional[int] = None) -> str:
        """
        Make API call to Gemini model.
        
        Args:
            messages: List of conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            thinking_budget_tokens: Thinking budget tokens for internal reasoning
            
        Returns:
            Model response text
        """
        # Convert messages to Gemini format
        gemini_messages = self._convert_messages_to_gemini_format(messages)
        
        # Use provided thinking_budget_tokens or fall back to instance default
        budget_tokens = thinking_budget_tokens if thinking_budget_tokens is not None else self.thinking_budget_tokens
        
        # Prepare generation config with supported parameters
        config = self.types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens if max_tokens is not None else self.max_output_tokens,
            top_p=self.top_p, # Use class attribute
            thinking_config=self.types.ThinkingConfig(
                thinking_budget=budget_tokens
            )
        )
        
        # Make API call with simple retry for transient network/SSL errors
        last_err = None
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=gemini_messages,
                    config=config
                )
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    um = response.usage_metadata
                    self._accumulate_usage({
                        "prompt_tokens": getattr(um, "prompt_token_count", 0) or 0,
                        "completion_tokens": getattr(um, "candidates_token_count", 0) or 0,
                        "thinking_tokens": getattr(um, "thoughts_token_count", 0) or 0,
                        "total_tokens": getattr(um, "total_token_count", 0) or 0,
                    })
                return response.text
            except Exception as e:
                last_err = e
                if attempt < 2:
                    # Exponential backoff with small jitter
                    backoff = (0.5 * (2 ** attempt)) + random.uniform(0, 0.25)
                    time.sleep(backoff)
                    continue
                raise

    def _make_image_generation_call(self, text_prompt: str, image_paths: Optional[List[str]] = None, temperature: float = 0.7, max_size: int = 1024) -> Dict[str, Any]:
        """
        Make image generation API call to Gemini.
        
        Args:
            text_prompt: Text prompt for image generation
            image_paths: Optional list of reference image paths
            temperature: Sampling temperature
            max_size: Maximum size of the longer image edge (pixels)
        Returns:
            Dictionary containing generated images and metadata
        """
        # Prepare content for image generation
        contents = [text_prompt]
        
        # Add reference images if provided
        if image_paths:
            for image_path in image_paths:
                if Path(image_path).exists():
                    image = Image.open(image_path)
                    # resize image if longer edge is larger than max_size
                    w, h = image.size
                    longer = max(w, h)
                    if longer > max_size:
                        scale = max_size / float(longer)
                        new_size = (int(round(w * scale)), int(round(h * scale)))
                        image = image.resize(new_size, Image.LANCZOS)
                    contents.append(image)
                else:
                    logger.warning(f"Image file not found: {image_path}")
        
        # Prepare generation config for image generation
        config = self.types.GenerateContentConfig(
            temperature=temperature,
            response_modalities=['TEXT', 'IMAGE']
        )
        
        # Make API call to image generation model
        response = self.client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=contents,
            config=config
        )
        
        # Extract generated images and text
        generated_images = []
        generated_text = ""
        
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    generated_text += part.text
                elif part.inline_data is not None:
                    # Convert base64 data to PIL Image
                    image_data = part.inline_data.data
                    image = Image.open(BytesIO(image_data))
                    generated_images.append(image)
        
        return {
            "generated_images": generated_images,
            "generated_text": generated_text,
            "metadata": {
                "prompt": text_prompt,
                "reference_images": image_paths,
                "model": "gemini-2.0-flash-preview-image-generation"
            }
        }

    def image_gen(
        self,
        text_prompt: str,
        image_paths: Optional[List[str]] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate images based on text prompt and optional reference images.
        
        Args:
            text_prompt: Text prompt for image generation
            image_paths: Optional list of reference image paths
            temperature: Sampling temperature
            
        Returns:
            Dictionary containing generated images and metadata
        """
        try:
            return self._make_image_generation_call(text_prompt, image_paths, temperature)
        except Exception as e:
            raise RuntimeError(f"Image generation failed: {str(e)}")

    def save_generated_images(self, image_result: Dict[str, Any], base_path: str = None, prefix: str = "generated_image") -> List[str]:
        """
        Save generated images from image generation result.
        
        Args:
            image_result: Result from image_gen() method
            base_path: Directory to save images (defaults to current directory)
            prefix: Prefix for saved image filenames
            
        Returns:
            List of saved file paths
        """
        if base_path is None:
            base_path = os.path.dirname(__file__)
        
        saved_paths = []
        generated_images = image_result.get('generated_images', [])
        
        for i, image in enumerate(generated_images):
            if image:  # Check if image is not None
                filename = f"{prefix}_{i+1}.jpg"
                save_path = os.path.join(base_path, filename)
                image.save(save_path)
                saved_paths.append(save_path)
                print(f"Saved image {i+1} to: {save_path}")
        
        return saved_paths


def _timed(label):
    """Context manager that prints elapsed time for a labeled block."""
    from contextlib import contextmanager
    @contextmanager
    def _ctx():
        t0 = time.perf_counter()
        yield
        print(f"  [{label}] took {time.perf_counter() - t0:.3f}s")
    return _ctx()


def run_test_qwen_agent():
    test_image = os.path.join(os.path.dirname(__file__), "test_agent.jpg")
    total_start = time.perf_counter()

    qwen_agent = QwenAgent(model="qwen-vl-max")

    with _timed("single image query"):
        response = qwen_agent.chat(
            text_prompt="Describe the image in one sentence",
            image_paths=[test_image]
        )
        print("Qwen response:", response)

    with _timed("text-only follow-up"):
        response = qwen_agent.chat("Describe the spatial dimension of the urban space")
        print("Qwen follow-up:", response)

    qwen_agent.clear_conversation()

    # print("\n=== Testing chat_image function with QwenAgent ===")
    # prompt_list = [
    #     "Describe the overall scene in this image",
    #     "What are the main architectural features?",
    #     "How would you characterize the urban environment?"
    # ]

    # with _timed("chat_image (3 prompts)"):
    #     responses = qwen_agent.chat_image(
    #         image_path=test_image,
    #         prompt_list=prompt_list,
    #         temperature=0.7
    #     )

    # for i, (prompt, resp) in enumerate(zip(prompt_list, responses)):
    #     print(f"\nPrompt {i+1}: {prompt}")
    #     print(f"Response: {resp}")

    elapsed = time.perf_counter() - total_start
    usage = qwen_agent.get_usage_summary()
    print(f"\n=== Summary ===")
    print(f"  API calls:         {usage['api_calls']}")
    print(f"  Prompt tokens:     {usage['prompt_tokens']}")
    print(f"  Completion tokens: {usage['completion_tokens']}")
    if usage['thinking_tokens']:
        print(f"  Thinking tokens:   {usage['thinking_tokens']}")
    print(f"  Total tokens:      {usage['total_tokens']}")
    print(f"  Wall time:         {elapsed:.3f}s")

def run_test_gemini_agent():
    test_image = os.path.join(os.path.dirname(__file__), "test_agent.jpg")
    total_start = time.perf_counter()

    gemini_agent = GeminiAgent(model="gemini-2.5-flash")

    with _timed("single image query"):
        response = gemini_agent.chat(
            text_prompt="Describe the image in one sentence",
            image_paths=[test_image],
            temperature=0.7
        )
        print("Gemini response:", response)

    with _timed("text-only follow-up"):
        response = gemini_agent.chat("Describe the spatial dimension of the urban space in one sentence")
        print("Gemini follow-up:", response)

    gemini_agent.clear_conversation()

    print("\n=== Testing chat_image function ===")
    prompt_list = [
        "Describe the overall scene in this image in one sentence",
        "List the main architectural features in one sentence"
    ]

    with _timed("chat_image (2 prompts)"):
        responses = gemini_agent.chat_image(
            image_path=test_image,
            prompt_list=prompt_list,
            temperature=0.7
        )

    for i, (prompt, resp) in enumerate(zip(prompt_list, responses)):
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"Response: {resp}")

    with _timed("image generation"):
        try:
            image_result = gemini_agent.image_gen(
                text_prompt="A modern urban street with tall buildings",
                temperature=0.7
            )
            print(f"Generated {len(image_result['generated_images'])} images")
            saved_paths = gemini_agent.save_generated_images(image_result, prefix="test_generation")
            print(f"Successfully saved {len(saved_paths)} images")
        except Exception as e:
            print(f"Image generation test failed: {e}")

    elapsed = time.perf_counter() - total_start
    usage = gemini_agent.get_usage_summary()
    print(f"\n=== Summary ===")
    print(f"  API calls:         {usage['api_calls']}")
    print(f"  Prompt tokens:     {usage['prompt_tokens']}")
    print(f"  Completion tokens: {usage['completion_tokens']}")
    if usage['thinking_tokens']:
        print(f"  Thinking tokens:   {usage['thinking_tokens']}")
    print(f"  Total tokens:      {usage['total_tokens']}")
    print(f"  Wall time:         {elapsed:.3f}s")

# Example usage
if __name__ == "__main__":
    # run_test_qwen_agent() # succeed
    run_test_gemini_agent() # succeed