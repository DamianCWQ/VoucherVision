import torch
from PIL import Image
import warnings
from transformers import DonutProcessor, VisionEncoderDecoderModel

try:
    from vouchervision.utils_LLM import SystemLoadMonitor
    from vouchervision.model_cache import ModelCache
except:
    from utils_LLM import SystemLoadMonitor
    from model_cache import ModelCache

warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

class DonutOCR:
    """
    Donut OCR class for Visual Document Understanding (VDU)
    Uses the Donut model from Hugging Face for OCR and document understanding
    
    Reference:
    @article{kim2021donut,
      title={OCR-free Document Understanding Transformer},
      author={Kim, Geewook and Hong, Teakgyu and Yim, Moonbin and Nam, JeongYeon and Park, Jinyoung and Yim, Jinyeong and Hwang, Wonseok and Yun, Sangdoo and Han, Dongyoon and Park, Seunghyun},
      journal={arXiv preprint arXiv:2111.15664},
      year={2021}
    }
    """
    
    # Class-level cache instance for model persistence
    _model_cache = ModelCache()
    
    @staticmethod
    def _load_donut_model(model_id, device):
        """Static method to load Donut model - called only once per model_id"""
        print(f"Loading Donut model from HuggingFace: {model_id}")
        model = VisionEncoderDecoderModel.from_pretrained(model_id)
        model.to(device)
        model.eval()
        return model
    
    @staticmethod
    def _load_donut_processor(model_id):
        """Static method to load Donut processor - called only once per model_id"""
        print(f"Loading Donut processor from HuggingFace: {model_id}")
        processor = DonutProcessor.from_pretrained(model_id)
        return processor
    
    def __init__(self, logger, model_id='naver-clova-ix/donut-base', device=None):
        """
        Initialize Donut OCR
        
        Args:
            logger: Logger instance
            model_id: Hugging Face model ID (default: 'naver-clova-ix/donut-base')
            device: Device to run model on (cuda/cpu), auto-detected if None
        """
        self.logger = logger
        self.model_id = model_id
        
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.logger.info(f"Initializing Donut OCR with model: {model_id} on device: {self.device}")
        
        # System monitoring
        self.monitor = SystemLoadMonitor(logger)
        
        # Use cached models instead of loading fresh each time
        self.model = self._model_cache.get_model(
            f"donut_model_{model_id}",
            self._load_donut_model,
            model_id,
            self.device
        )
        
        self.processor = self._model_cache.get_model(
            f"donut_processor_{model_id}",
            self._load_donut_processor,
            model_id
        )
        
        # Default task prompt for OCR
        # Different fine-tuned models use different task prompts
        # HeR-T is specifically fine-tuned for herbarium specimens
        
        if 'her-t' in model_id.lower() or 'hert' in model_id.lower():
            # HeR-T: Herbarium specimen transcription model
            # Uses <s_herbarium> task prompt (token ID 57537)
            self.task_prompt = "<s_herbarium>"
            self.logger.info(f"Using HeR-T model - optimized for herbarium specimens!")
        elif 'cord' in model_id.lower():
            self.task_prompt = "<s_cord-v2>"
        elif 'docvqa' in model_id.lower():
            self.task_prompt = "<s_docvqa>"
        elif 'rvl' in model_id.lower():
            self.task_prompt = "<s_rvlcdip>"
        else:
            # For base model or unknown models
            self.task_prompt = "<s>"
            if 'base' in model_id.lower() and 'finetuned' not in model_id.lower():
                self.logger.warning(
                    f"Using base Donut model without fine-tuning may produce poor results. "
                    f"Consider using 'elderprince/HeR-T' for herbarium specimens or other fine-tuned models."
                )
        
    def transcribe_document(self, image_path, task_prompt=None, max_length=1024):
        """
        Perform OCR on document image using Donut
        
        Args:
            image_path: Path to image file or PIL Image object
            task_prompt: Task-specific prompt (uses default if None)
            max_length: Maximum length of generated text (increased to 1024 for longer documents)
            
        Returns:
            tuple: (transcribed_text, usage_report)
        """
        self.monitor.start_monitoring_usage()
        
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path.convert("RGB")
            
        # Use default task prompt if not provided
        if task_prompt is None:
            task_prompt = self.task_prompt
            
        self.logger.info(f"Using task prompt: {task_prompt}")
            
        # Prepare inputs
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        
        # Prepare decoder inputs
        decoder_input_ids = self.processor.tokenizer(
            task_prompt, 
            add_special_tokens=False, 
            return_tensors="pt"
        ).input_ids
        decoder_input_ids = decoder_input_ids.to(self.device)
        
        self.logger.info(f"Generating text with max_length={max_length}")
        
        # Generate text with adjusted parameters for better output
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=max_length,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,  # Use greedy decoding for faster inference
                early_stopping=True,
                return_dict_in_generate=True,
            )
        
        # Decode the generated text
        sequence = outputs.sequences[0]
        # Replace pad tokens with eos tokens (tensor operation, not string replace)
        sequence[sequence == self.processor.tokenizer.pad_token_id] = self.processor.tokenizer.eos_token_id
        
        # Decode with skip_special_tokens=False to see what was generated
        transcribed_text_raw = self.processor.batch_decode([sequence], skip_special_tokens=False)[0]
        self.logger.info(f"Raw Donut output (with special tokens): {transcribed_text_raw[:500]}")
        
        # Decode again with skip_special_tokens=True for clean text
        transcribed_text = self.processor.batch_decode([sequence], skip_special_tokens=True)[0]
        self.logger.info(f"Cleaned Donut output: {transcribed_text[:500]}")
        
        # Clean up the text (remove task prompt if present)
        if task_prompt in transcribed_text:
            transcribed_text = transcribed_text.replace(task_prompt, "").strip()
        
        # For the base Donut model, we just return the text directly
        # Fine-tuned models (cord-v2, docvqa) return structured JSON
        parsed_result = transcribed_text
        
        # If the output looks like JSON, try to parse it
        if transcribed_text.strip().startswith("{") or transcribed_text.strip().startswith("["):
            try:
                import json
                parsed_result = json.loads(transcribed_text)
                self.logger.info(f"Successfully parsed JSON output")
            except:
                self.logger.info(f"Output looks like JSON but couldn't parse, keeping as text")
                pass
        
        # Create usage report
        usage_report = {
            "model": self.model_id,
            "device": str(self.device),
            "image_size": image.size,
            "max_length": max_length
        }
        
        self.monitor.stop_inference_timer()  # Must be called before stop_monitoring_report_usage
        monitor_report = self.monitor.stop_monitoring_report_usage()
        usage_report.update(monitor_report)
        
        return parsed_result, usage_report
    
    def ocr_donut(self, image_path, task_prompt=None, max_length=1024, flatten_json=True):
        """
        Perform OCR with Donut and return formatted text
        
        Args:
            image_path: Path to image or PIL Image
            task_prompt: Custom task prompt (optional)
            max_length: Maximum generation length
            flatten_json: If True, flattens JSON output to string
            
        Returns:
            tuple: (formatted_text, raw_json, usage_report)
        """
        self.logger.info(f"Running Donut OCR on image")
        
        # Get transcription
        result, usage_report = self.transcribe_document(
            image_path, 
            task_prompt=task_prompt,
            max_length=max_length
        )
        
        # Format output
        if isinstance(result, dict):
            raw_json = result
            if flatten_json:
                # Convert dict to readable text
                formatted_text = self._format_json_to_text(result)
            else:
                formatted_text = str(result)
        else:
            formatted_text = str(result)
            raw_json = {}
            
        self.logger.info(f"Donut OCR completed successfully")
        
        return formatted_text, raw_json, usage_report
    
    def _format_json_to_text(self, json_dict, level=0):
        """
        Recursively format JSON dictionary to readable text
        
        Args:
            json_dict: Dictionary to format
            level: Current indentation level
            
        Returns:
            str: Formatted text
        """
        text_lines = []
        indent = "  " * level
        
        for key, value in json_dict.items():
            if isinstance(value, dict):
                text_lines.append(f"{indent}{key}:")
                text_lines.append(self._format_json_to_text(value, level + 1))
            elif isinstance(value, list):
                text_lines.append(f"{indent}{key}:")
                for item in value:
                    if isinstance(item, dict):
                        text_lines.append(self._format_json_to_text(item, level + 1))
                    else:
                        text_lines.append(f"{indent}  - {item}")
            else:
                text_lines.append(f"{indent}{key}: {value}")
                
        return "\n".join(text_lines)


if __name__ == "__main__":
    # Example usage
    import logging
    
    # Setup logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize Donut OCR
    donut_ocr = DonutOCR(logger, model_id='naver-clova-ix/donut-base')
    
    # Example image path (replace with your actual image)
    # image_path = "path/to/your/document.jpg"
    
    # Perform OCR
    # formatted_text, raw_json, usage_report = donut_ocr.ocr_donut(image_path)
    # print("Transcribed Text:")
    # print(formatted_text)
    # print("\nUsage Report:")
    # print(usage_report)
