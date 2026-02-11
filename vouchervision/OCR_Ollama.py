"""
OCR using Ollama Vision Models
Supports vision-enabled models like llama3.2-vision, qwen2-vl, etc.
"""
import os
import base64
import io
import json
import requests
import threading
from PIL import Image
import numpy as np

try:
    from vouchervision.model_cache import ModelCache
except:
    from model_cache import ModelCache


class OllamaVisionOCR:
    """OCR using Ollama vision-enabled models"""
    
    # Class-level cache for model connections (though Ollama manages models server-side)
    _model_cache = ModelCache()
    
    # Semaphore to limit concurrent Ollama requests (vision models are resource-intensive)
    # Only allow 2 concurrent requests to prevent overwhelming Ollama
    _request_semaphore = threading.Semaphore(2)
    
    def __init__(self, logger=None, model_name='llama3.2-vision', base_url=None):
        """
        Initialize Ollama Vision OCR
        
        Args:
            logger: Logger instance
            model_name: Ollama model name (e.g., 'llama3.2-vision', 'qwen3-vl:4b')
            base_url: Ollama API base URL (default: http://localhost:11434)
        """
        self.logger = logger
        self.model_name = model_name
        self.base_url = base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        
        # Timeout for vision models (they're slower than text-only models)
        self.timeout = int(os.getenv('OLLAMA_OCR_TIMEOUT', '600'))  # 10 minutes default (Qwen3-VL is slow)
        
        # OCR-specific prompts optimized for Qwen vision models
        self.PROMPT_OCR_ONLY = """Extract and transcribe all text visible in this image. Include all labels, printed text, handwritten text, numbers, dates, and location information. Return only the raw text without any explanation."""

        self.PROMPT_OCR_STRUCTURED = """Transcribe all text from this image. List each text element you see, including:
- Labels and printed text
- Handwritten notes
- Numbers and codes
- Dates and locations
Just list the text you see, one item per line."""
        
        # Verify Ollama is running and model is available
        self._verify_ollama_connection()
    
    def _verify_ollama_connection(self):
        """Verify Ollama server is accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                
                if self.logger:
                    self.logger.info(f"Connected to Ollama at {self.base_url}")
                    self.logger.info(f"Available models: {model_names}")
                
                # Check if requested model is available
                if not any(self.model_name in name for name in model_names):
                    warning_msg = f"Warning: Model '{self.model_name}' not found. Available: {model_names}"
                    if self.logger:
                        self.logger.warning(warning_msg)
                    print(warning_msg)
            else:
                error_msg = f"Ollama server responded with status {response.status_code}"
                if self.logger:
                    self.logger.error(error_msg)
                raise ConnectionError(error_msg)
        except requests.exceptions.RequestException as e:
            error_msg = f"Cannot connect to Ollama at {self.base_url}: {e}"
            if self.logger:
                self.logger.error(error_msg)
            raise ConnectionError(error_msg)
    
    def encode_image_base64(self, image):
        """Encode PIL Image to base64 string"""
        buffered = io.BytesIO()
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def ocr_with_ollama(self, image_input, prompt=None, structured=False):
        """
        Perform OCR using Ollama vision model
        
        Args:
            image_input: Path to image, PIL Image, or numpy array
            prompt: Custom prompt (optional, uses default if None)
            structured: Whether to request structured output
            
        Returns:
            tuple: (ocr_text, full_response, usage_report)
        """
        # Use semaphore to limit concurrent requests to Ollama
        # This prevents overwhelming the server with parallel vision model requests
        with self._request_semaphore:
            if self.logger:
                self.logger.info(f"Acquired Ollama request slot (max 2 concurrent)")
            
            # Load and prepare image
            if isinstance(image_input, str):
                image = Image.open(image_input)
            elif isinstance(image_input, Image.Image):
                image = image_input
            elif isinstance(image_input, np.ndarray):
                image = Image.fromarray(image_input)
            else:
                raise ValueError("Unsupported input type. Provide path, PIL Image, or numpy array.")
            
            # Encode image
            image_b64 = self.encode_image_base64(image)
            
            # Select prompt
            if prompt is None:
                prompt = self.PROMPT_OCR_STRUCTURED if structured else self.PROMPT_OCR_ONLY
            
            # Prepare Ollama API request
            api_url = f"{self.base_url}/api/generate"
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for consistent OCR
                    "top_p": 0.9,
                }
            }
            
            if self.logger:
                self.logger.info(f"Sending OCR request to Ollama model: {self.model_name} (timeout: {self.timeout}s)")
            
            try:
                response = requests.post(api_url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                
                result = response.json()
                ocr_text = result.get('response', '').strip()
                
                # Create usage report
                usage_report = {
                    'ocr_method': f'Ollama-{self.model_name}',
                    'model': self.model_name,
                    'total_tokens': result.get('eval_count', 0) + result.get('prompt_eval_count', 0),
                    'prompt_tokens': result.get('prompt_eval_count', 0),
                    'completion_tokens': result.get('eval_count', 0),
                    'total_time_seconds': result.get('total_duration', 0) / 1_000_000_000,  # Convert nanoseconds
                    'cost': 0.0,  # Ollama is free/local
                }
                
                if self.logger:
                    self.logger.info(f"OCR completed. Extracted {len(ocr_text)} characters")
                    self.logger.info(f"Tokens: {usage_report['total_tokens']}, Time: {usage_report['total_time_seconds']:.2f}s")
                
                return ocr_text, result, usage_report
                
            except requests.exceptions.Timeout:
                error_msg = f"Ollama OCR timed out after {self.timeout}s. Vision models can be slow - try increasing OLLAMA_OCR_TIMEOUT env var or reducing num_workers in config."
                if self.logger:
                    self.logger.error(error_msg)
                raise RuntimeError(error_msg)
            except requests.exceptions.RequestException as e:
                error_msg = f"Ollama API request failed: {e}"
                if self.logger:
                    self.logger.error(error_msg)
                raise RuntimeError(error_msg)


def main():
    """Test the Ollama OCR with an example image"""
    import sys
    
    # Example usage
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        print("Usage: python OCR_Ollama.py <path_to_image>")
        print("Using default test mode...")
        img_path = None
    
    # Initialize OCR with your model
    # Common Ollama vision models: llama3.2-vision, qwen2-vl:4b, bakllava
    ocr = OllamaVisionOCR(model_name='llama3.2-vision')  
    
    if img_path:
        # Perform OCR
        text, full_response, usage = ocr.ocr_with_ollama(img_path)
        
        print("\n" + "="*50)
        print("EXTRACTED TEXT:")
        print("="*50)
        print(text)
        print("\n" + "="*50)
        print("USAGE STATS:")
        print("="*50)
        print(json.dumps(usage, indent=2))
    else:
        print("Ollama OCR initialized successfully!")
        print(f"Using model: {ocr.model_name}")
        print(f"Ollama base URL: {ocr.base_url}")


if __name__ == '__main__':
    main()
