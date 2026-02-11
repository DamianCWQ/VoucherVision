# Ollama Integration for VoucherVision

## Quick Start

VoucherVision now supports Ollama for running local LLMs! This allows you to use powerful open-source models completely offline.

### 1. Install Ollama

Visit [ollama.ai](https://ollama.ai) and follow the installation instructions for your platform.

### 2. Pull a Model

```bash
ollama pull llama3.2
```

### 3. Start Ollama

```bash
ollama serve
```

### 4. Use in VoucherVision

1. Launch VoucherVision
2. Select **Ollama** from the Model Family dropdown
3. Choose your model (e.g., "Ollama llama3.2")
4. Run transcription!

## Available Pre-configured Models

- Ollama llama3.2
- Ollama llama3.2-vision
- Ollama llama3.1
- Ollama mistral
- Ollama mixtral
- Ollama qwen2.5
- Ollama Custom (use any model you have installed)

## Custom Models

To use a model not in the list:

1. Create a `.env` file with:
   ```bash
   OLLAMA_CUSTOM_MODEL_NAME=your-model-name
   ```

2. Select "Ollama Custom" in VoucherVision

For detailed setup instructions, see [OLLAMA_SETUP.md](OLLAMA_SETUP.md)

## Benefits

✅ **Privacy**: All data stays local  
✅ **No API costs**: Run unlimited inferences  
✅ **Offline capable**: Works without internet  
✅ **Model variety**: Use any Ollama-compatible model  

## Requirements

- Sufficient RAM (8GB minimum, 16GB+ recommended)
- GPU recommended for faster inference (but not required)
- Ollama installed and running

---

For complete documentation and troubleshooting, see [OLLAMA_SETUP.md](OLLAMA_SETUP.md)
