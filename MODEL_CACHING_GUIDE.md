# Model Caching Guide for VoucherVision

## Problem Solved
Previously, VoucherVision was loading AI models (Florence, Qwen, Phi-3.5) fresh from disk **for every worker thread**, causing:
- Out-of-memory (OOM) errors
- Excessive GPU memory usage
- Slow processing times
- Repeated model loading overhead

## Solution: Model Cache
A singleton-based model cache now ensures models are loaded **once** and **shared** across all worker threads.

## How It Works

### Architecture
1. **ModelCache** (singleton class) manages all model instances
2. Each OCR class (FlorenceOCR, Qwen2VLOCR, Phi35VisionOCR) uses the cache
3. First thread to request a model → loads it
4. Subsequent threads → get cached reference (instant)

### Benefits
- ✅ **Reduced Memory**: Models loaded once, not per-thread
- ✅ **Faster Processing**: No repeated model loading
- ✅ **Thread-Safe**: Proper locking prevents race conditions
- ✅ **Automatic**: No configuration needed

## Usage

### Normal Operation
Everything is automatic! Just use VoucherVision as normal:

```bash
python run_VoucherVision_CLI.py
```

### Cache Management (Advanced)

If you need to clear the cache (rarely needed):

```python
from vouchervision.model_cache import ModelCache

# Clear all cached models and free GPU memory
cache = ModelCache()
cache.clear_cache()

# Remove a specific model
cache.remove_model("florence_model_microsoft/Florence-2-large")
```

## Environment Variables (Optional)

For better memory management with PyTorch:

```bash
# PowerShell
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Linux/Mac
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

## What Changed

### Modified Files
1. **vouchervision/model_cache.py** (NEW) - Singleton cache manager
2. **vouchervision/OCR_Florence_2.py** - Uses cache for Florence models
3. **vouchervision/OCR_Qwen.py** - Uses cache for Qwen models  
4. **vouchervision/OCR_Phi_35_Vision.py** - Uses cache for Phi models

### Code Changes
Before (each thread loads fresh):
```python
def __init__(self, logger, model_id):
    self.model = AutoModelForCausalLM.from_pretrained(model_id).cuda()
```

After (shared cache):
```python
def __init__(self, logger, model_id):
    self.model = self._model_cache.get_model(
        f"model_{model_id}",
        self._load_model,
        model_id
    )
```

## Monitoring

You'll see these messages in the console:

**First load (normal):**
```
Loading Florence model from HuggingFace: microsoft/Florence-2-large
```

**Using cache (shows it's working):**
```
Using cached model: florence_model_microsoft/Florence-2-large
```

## Troubleshooting

### Still getting OOM errors?
1. Reduce number of workers in config
2. Use smaller model variants (e.g., Florence-2-base instead of Florence-2-large)
3. Set environment variable: `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`

### Models not being cached?
- Check console output for "Using cached model" messages
- If not appearing, check for import errors in model_cache.py

### Want to force reload?
```python
from vouchervision.model_cache import ModelCache
cache = ModelCache()
cache.clear_cache()  # Next run will reload all models
```

## Technical Details

### Thread Safety
- Uses `threading.Lock()` for singleton creation
- Model-specific locks prevent concurrent loading
- Safe for multi-threaded processing

### Memory Management
- Cached models persist for entire Python process
- Cleared only on process exit or manual clear
- GPU memory freed when cache cleared

### Cache Keys
Models are cached using unique keys:
- Florence: `florence_model_{model_id}` and `florence_processor_{model_id}`
- Qwen: `qwen_model_{model_id}` and `qwen_processor_{model_id}`
- Phi: `phi_model_{model_id}` and `phi_processor_{model_id}`
- Mistral (cleaning): `mistral_model_{model_id}` and `mistral_tokenizer_{model_id}`

## Questions?
Check the console output for cache-related messages or create an issue on GitHub.
