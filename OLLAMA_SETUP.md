# Using Ollama and Custom Models with VoucherVision

This guide explains how to use Ollama models and other custom LLMs with VoucherVision.

## Table of Contents
1. [Using Pre-configured Ollama Models](#using-pre-configured-ollama-models)
2. [Using Custom Ollama Models](#using-custom-ollama-models)
3. [Configuration Options](#configuration-options)
4. [Troubleshooting](#troubleshooting)

## Using Pre-configured Ollama Models

VoucherVision now supports several pre-configured Ollama models:

- **Ollama llama3.2** - Latest Llama 3.2 model
- **Ollama llama3.2-vision** - Llama 3.2 with vision capabilities
- **Ollama llama3.1** - Llama 3.1 model
- **Ollama mistral** - Mistral model
- **Ollama mixtral** - Mixtral model
- **Ollama qwen2.5** - Qwen 2.5 model
- **Ollama Custom** - Use any custom Ollama model

### Prerequisites

1. **Install Ollama**: Follow the instructions at [ollama.ai](https://ollama.ai)
2. **Pull the model you want to use**:
   ```bash
   ollama pull llama3.2
   # or any other model
   ```

### Steps to Use

1. **Start Ollama** (if not already running):
   ```bash
   ollama serve
   ```

2. **Launch VoucherVision** and select the Ollama model family from the dropdown

3. **Choose your model** from the available Ollama models

4. **Run your transcription** as normal

## Using Custom Ollama Models

If you want to use an Ollama model that's not in the pre-configured list, use the "Ollama Custom" option.

### Step 1: Set Environment Variables

Create or edit a `.env` file in your VoucherVision root directory:

```bash
# Ollama base URL (default: http://localhost:11434)
OLLAMA_BASE_URL=http://localhost:11434

# Custom model name (for "Ollama Custom" option)
OLLAMA_CUSTOM_MODEL_NAME=your-model-name
```

### Step 2: Pull Your Custom Model

```bash
ollama pull your-model-name
```

### Step 3: Select "Ollama Custom" in VoucherVision

In the VoucherVision interface:
1. Select **Ollama** as the model family
2. Choose **Ollama Custom** from the model dropdown
3. Run your transcription

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_BASE_URL` | URL where Ollama is running | `http://localhost:11434` |
| `OLLAMA_CUSTOM_MODEL_NAME` | Model name when using "Ollama Custom" | None |

### Model Parameters

You can configure model parameters in your VoucherVision config YAML file:

```yaml
ollama:
  temperature: 0.5        # Controls randomness (0.0 - 1.0)
  max_tokens: 1024        # Maximum tokens to generate
  top_p: 1.0             # Nucleus sampling parameter
  top_k: 40              # Top-k sampling parameter
```

## Advanced: Using Remote Ollama Instances

If you're running Ollama on a different machine or port:

```bash
# In your .env file
OLLAMA_BASE_URL=http://your-server:11434
```

## Vision Models

For models with vision capabilities (like llama3.2-vision), VoucherVision will automatically detect and use the vision features when processing images.

## Troubleshooting

### "Connection refused" error

**Problem**: Cannot connect to Ollama

**Solution**: 
- Ensure Ollama is running: `ollama serve`
- Check the URL in `OLLAMA_BASE_URL` is correct
- Verify firewall settings if using a remote instance

### Model not found

**Problem**: Ollama reports the model doesn't exist

**Solution**:
- Pull the model first: `ollama pull model-name`
- Check spelling of model name in environment variables
- List available models: `ollama list`

### Slow performance

**Problem**: Model runs very slowly

**Solution**:
- Ensure you have sufficient RAM/VRAM
- Try a smaller model
- Check CPU/GPU usage with system monitoring tools
- Consider using quantized models (GGUF format)

### JSON parsing errors

**Problem**: VoucherVision fails to parse model output

**Solution**:
- Increase temperature slightly (try 0.3-0.7)
- Try a different model that better supports structured output
- Check Ollama logs for errors

## Adding Other Custom Model Providers

The same pattern used for Ollama can be adapted for other API-compatible LLM providers:

1. Create a new handler in `vouchervision/LLM_YourProvider.py`
2. Add the provider to `model_maps.py`:
   - Add to `MODELS_*` list
   - Add to `MODEL_FAMILY` dict
   - Add version mappings
   - Add to `get_API_name()` method
3. Update `initialize_llm_model()` in:
   - `utils_VoucherVision.py`
   - `utils_VoucherVision_batch.py`
   - `utils_VoucherVision_parallel.py`

See `LLM_Ollama.py` as a reference implementation.

## Example: Complete Setup

Here's a complete example for setting up llama3.2-vision:

```bash
# 1. Install Ollama (if not already installed)
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Pull the model
ollama pull llama3.2-vision

# 3. Create .env file in VoucherVision directory
echo "OLLAMA_BASE_URL=http://localhost:11434" > .env

# 4. Start Ollama
ollama serve

# 5. Launch VoucherVision
# - Select "Ollama" as model family
# - Select "Ollama llama3.2-vision" as model
# - Run your transcription
```

## Support

For issues specific to:
- **Ollama**: Check [Ollama documentation](https://github.com/ollama/ollama)
- **VoucherVision**: Open an issue on the VoucherVision GitHub repository
