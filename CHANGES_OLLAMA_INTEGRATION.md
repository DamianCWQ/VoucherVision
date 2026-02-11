# VoucherVision Ollama & Custom Model Integration - Summary

## What Was Changed

This modification adds support for Ollama and other custom LLM providers to VoucherVision, allowing you to use local models and custom API endpoints beyond the pre-configured options.

## Files Modified

### 1. Core Model Configuration
- **[vouchervision/model_maps.py](vouchervision/model_maps.py)**
  - Added `MODELS_OLLAMA` list with pre-configured Ollama models
  - Added "Ollama" to `MODEL_FAMILY` dictionary
  - Added color mappings for Ollama models in expense reports
  - Added version mappings for all Ollama models
  - Updated `get_API_name()` method to handle Ollama model names
  - Updated `get_version_has_key()` and `get_version_mapping_is_azure()` methods

### 2. Ollama Handler
- **[vouchervision/LLM_Ollama.py](vouchervision/LLM_Ollama.py)** (NEW FILE)
  - Complete handler for Ollama API integration
  - Supports both chat and vision models
  - Configurable via environment variables
  - Implements retry logic and error handling
  - Compatible with VoucherVision's JSON validation system

### 3. Model Initialization
Updated `initialize_llm_model()` method in:
- **[vouchervision/utils_VoucherVision.py](vouchervision/utils_VoucherVision.py)**
- **[vouchervision/utils_VoucherVision_batch.py](vouchervision/utils_VoucherVision_batch.py)**
- **[vouchervision/utils_VoucherVision_parallel.py](vouchervision/utils_VoucherVision_parallel.py)**

All three files now check for 'Ollama' in name_parts and instantiate the OllamaHandler accordingly.

### 4. Documentation
- **[OLLAMA_SETUP.md](OLLAMA_SETUP.md)** (NEW FILE)
  - Comprehensive setup guide
  - Configuration options
  - Troubleshooting tips
  - Instructions for adding other custom providers

- **[OLLAMA_README.md](OLLAMA_README.md)** (NEW FILE)
  - Quick start guide
  - Benefits overview
  - Requirements list

- **[.env.example](.env.example)** (NEW FILE)
  - Template for environment variable configuration
  - Documents all required and optional variables
  - Includes examples for multiple providers

## Available Ollama Models

The following models are pre-configured and ready to use:

1. **Ollama llama3.2** - Latest Llama 3.2 model
2. **Ollama llama3.2-vision** - Llama 3.2 with vision capabilities
3. **Ollama llama3.1** - Llama 3.1 model
4. **Ollama mistral** - Mistral model
5. **Ollama mixtral** - Mixtral mixture-of-experts model
6. **Ollama qwen2.5** - Qwen 2.5 model
7. **Ollama Custom** - Use any Ollama model by name

## How to Use

### Quick Start (Using Pre-configured Models)

1. Install Ollama: https://ollama.ai
2. Pull a model: `ollama pull llama3.2`
3. Start Ollama: `ollama serve`
4. In VoucherVision UI:
   - Select "Ollama" from Model Family dropdown
   - Select your model (e.g., "Ollama llama3.2")
   - Run transcription

### Using Custom Models

1. Create a `.env` file in the VoucherVision root directory:
   ```bash
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_CUSTOM_MODEL_NAME=your-model-name
   ```

2. Pull your custom model in Ollama:
   ```bash
   ollama pull your-model-name
   ```

3. In VoucherVision, select:
   - Model Family: "Ollama"
   - Model: "Ollama Custom"

### Remote Ollama Instance

To use Ollama running on a different machine:

```bash
# In .env file
OLLAMA_BASE_URL=http://your-server-ip:11434
```

## Configuration Options

Environment variables in `.env`:

| Variable | Purpose | Default |
|----------|---------|---------|
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_CUSTOM_MODEL_NAME` | Model name for "Ollama Custom" | None |

Model parameters (in VoucherVision config YAML):

```yaml
ollama:
  temperature: 0.5    # Randomness (0.0-1.0)
  max_tokens: 1024    # Max response length
  top_p: 1.0         # Nucleus sampling
  top_k: 40          # Top-k sampling
```

## Architecture

The implementation follows VoucherVision's existing patterns:

```
Model Selection (UI)
    ↓
ModelMaps.MODEL_FAMILY['Ollama']
    ↓
initialize_llm_model() detects 'Ollama' in name_parts
    ↓
OllamaHandler initialized
    ↓
call_llm_api_Ollama() sends request
    ↓
Response parsed, validated, and returned
```

## Extending to Other Providers

The same pattern can be used to add support for other LLM providers:

1. Create `vouchervision/LLM_YourProvider.py` (use LLM_Ollama.py as template)
2. Add provider to `model_maps.py`
3. Update `initialize_llm_model()` in utils files
4. Create documentation

See [OLLAMA_SETUP.md](OLLAMA_SETUP.md) section "Adding Other Custom Model Providers" for details.

## Benefits

✅ **Privacy**: Run models locally, data never leaves your machine  
✅ **Cost**: No API fees for unlimited usage  
✅ **Offline**: Works without internet connection  
✅ **Flexibility**: Use any Ollama-compatible model  
✅ **Customization**: Fine-tune models for your specific use case  
✅ **Control**: Full control over model parameters and behavior  

## Requirements

- **Ollama**: Installed and running
- **RAM**: 8GB minimum (16GB+ recommended for larger models)
- **GPU**: Optional but recommended for faster inference
- **Dependencies**: Already included in requirements.txt (langchain-community)

## Testing

To test the integration:

1. Start Ollama with a small model:
   ```bash
   ollama pull llama3.2
   ollama serve
   ```

2. In VoucherVision:
   - Select "Ollama" → "Ollama llama3.2"
   - Process a test image
   - Verify JSON output is correct

3. Check logs for any errors or warnings

## Troubleshooting

Common issues and solutions are documented in [OLLAMA_SETUP.md](OLLAMA_SETUP.md#troubleshooting)

## Future Enhancements

Possible future improvements:

- [ ] Auto-detect available Ollama models
- [ ] Support for custom prompting strategies per model
- [ ] Fine-tuning workflow integration
- [ ] Batch processing optimizations for local models
- [ ] Model performance benchmarking tools
- [ ] Support for other local inference engines (LM Studio, GPT4All, etc.)

## Migration Notes

**No breaking changes** - All existing functionality remains unchanged. The Ollama integration is purely additive.

Existing users can continue using their current models without any modifications.

## Support

- **Ollama-specific issues**: https://github.com/ollama/ollama
- **VoucherVision integration**: Open an issue on VoucherVision repository
- **Documentation**: See [OLLAMA_SETUP.md](OLLAMA_SETUP.md)

---

**Last Updated**: February 10, 2026  
**Author**: VoucherVision AI Assistant  
**Version**: 1.0
