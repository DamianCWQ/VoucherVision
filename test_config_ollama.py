"""
Test script to verify Ollama LLM configuration can be set via config file
"""
import os
import yaml

# Load config
config_path = os.path.join(os.path.dirname(__file__), 'custom_VV_config.yaml')

with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

# Check if ollama_llm_model is in config
ollama_llm_model = cfg['leafmachine']['project'].get('ollama_llm_model', None)

print("="*60)
print("Ollama LLM Configuration Test")
print("="*60)
print(f"\n✅ Config file: {config_path}")
print(f"\n📋 Configuration values:")
print(f"   LLM_version: {cfg['leafmachine']['LLM_version']}")
print(f"   ollama_llm_model (from config): {ollama_llm_model}")
print(f"   OLLAMA_CUSTOM_MODEL_NAME (from env): {os.getenv('OLLAMA_CUSTOM_MODEL_NAME', '(not set)')}")

print("\n🔍 Priority order (as implemented):")
print("   1. Config file: ollama_llm_model")
print("   2. Environment: OLLAMA_CUSTOM_MODEL_NAME")

if ollama_llm_model:
    print(f"\n✅ SUCCESS! Will use model from config: {ollama_llm_model}")
else:
    env_model = os.getenv('OLLAMA_CUSTOM_MODEL_NAME', None)
    if env_model:
        print(f"\n⚠️  Will fall back to env variable: {env_model}")
    else:
        print("\n❌ No Ollama LLM model configured!")

print("\n" + "="*60)
print("To change the model, edit custom_VV_config.yaml:")
print("   ollama_llm_model: your-model-name")
print("="*60)
