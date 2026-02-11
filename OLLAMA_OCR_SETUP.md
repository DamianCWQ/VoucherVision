# Using Ollama Vision Models for OCR in VoucherVision

## Quick Start

Your VoucherVision is now configured to use Ollama's Qwen2-VL 4B model for OCR!

### 1. Verify Your Ollama Model

First, check which vision model you have installed:

```powershell
ollama list
```

Look for vision-capable models like:
- `qwen2-vl:4b` or `qwen2-vl:7b`
- `llama3.2-vision`
- `bakllava`
- Or any other vision model you've downloaded

### 2. Pull the Model (if needed)

If you don't have a Qwen2-VL model yet:

```powershell
# For the 4B parameter version (faster, less memory)
ollama pull qwen2-vl:4b

# OR for the 7B parameter version (more accurate, more memory)
ollama pull qwen2-vl:7b
```

**Note:** There's currently no Qwen3-VL available in Ollama. If you have a custom Qwen3 model, you'll need to verify its exact name with `ollama list`.

### 3. Update Configuration

Your configuration has been updated to:

**In `.env` file:**
```env
OLLAMA_OCR_MODEL=qwen2-vl:4b
OLLAMA_BASE_URL=http://localhost:11434
```

**In `custom_VV_config.yaml`:**
```yaml
OCR_option:
  - Ollama

ollama_ocr_model: qwen2-vl:4b
```

### 4. Customize Model (Optional)

To use a different Ollama vision model:

**Option A:** Edit `.env` file
```env
OLLAMA_OCR_MODEL=llama3.2-vision
```

**Option B:** Edit `custom_VV_config.yaml`
```yaml
ollama_ocr_model: llama3.2-vision
```

Config file takes precedence over .env file.

### 5. Run VoucherVision

Ensure Ollama is running (it usually starts automatically):

```powershell
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it
ollama serve
```

Then run VoucherVision:

```powershell
python run_VoucherVision_CLI.py
```

## Supported Vision Models

Any Ollama model with vision capabilities will work. Popular options:

| Model | Size | Best For |
|-------|------|----------|
| `qwen2-vl:4b` | ~2.5GB | Fast OCR, lower memory |
| `qwen2-vl:7b` | ~4.4GB | Better accuracy, more memory |
| `llama3.2-vision` | ~7.9GB | General vision tasks |
| `bakllava` | ~4.7GB | Alternative vision model |

## Troubleshooting

### Model Not Found Error

```
Warning: Model 'qwen2-vl:4b' not found
```

**Solution:** Pull the model first:
```powershell
ollama pull qwen2-vl:4b
```

### Connection Error

```
Cannot connect to Ollama at http://localhost:11434
```

**Solution:** Start Ollama:
```powershell
ollama serve
```

### Wrong Model Name

Check your exact model name:
```powershell
ollama list
```

Then update either `.env` or `custom_VV_config.yaml` with the exact name shown.

### Using Custom Qwen3 Model

If you have a custom Qwen3-VL model:

1. Check its exact name: `ollama list`
2. Update config with that exact name:
   ```yaml
   ollama_ocr_model: your-exact-model-name-here
   ```

## Performance Notes

- **Speed:** Local Ollama OCR is faster than cloud APIs (no network latency)
- **Cost:** Free! No API charges
- **Privacy:** All processing happens locally
- **Memory:** Models stay loaded in memory (use model caching benefits)

## Testing OCR

Test your Ollama OCR directly:

```powershell
cd vouchervision
python OCR_Ollama.py path/to/your/test/image.jpg
```

This will show you the extracted text and performance stats.

## Combining with LLM

Your current setup:
- **OCR:** Ollama (qwen2-vl:4b) - Extracts text from images
- **LLM:** Ollama Custom (llama3.1:8b) - Parses extracted text into structured data

Both are now using local Ollama models = completely free and private!
