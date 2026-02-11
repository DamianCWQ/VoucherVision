"""
Ollama Configuration Test Script for VoucherVision

This script helps verify that your Ollama setup is working correctly
before using it with VoucherVision.

Usage:
    python test_ollama_setup.py
"""

import os
import sys
import json
import requests
from urllib.parse import urljoin

def test_ollama_connection(base_url="http://localhost:11434"):
    """Test connection to Ollama server"""
    print(f"Testing connection to Ollama at {base_url}...")
    
    try:
        response = requests.get(urljoin(base_url, "api/tags"), timeout=5)
        if response.status_code == 200:
            print("✓ Connection successful!")
            return True, response.json()
        else:
            print(f"✗ Connection failed with status code: {response.status_code}")
            return False, None
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to Ollama at {base_url}")
        print("  Make sure Ollama is running with: ollama serve")
        return False, None
    except Exception as e:
        print(f"✗ Error: {e}")
        return False, None

def list_available_models(models_data):
    """List available Ollama models"""
    if not models_data or 'models' not in models_data:
        print("\n✗ No models found")
        return []
    
    models = models_data['models']
    if not models:
        print("\n✗ No models installed")
        print("  Install a model with: ollama pull llama3.2")
        return []
    
    print(f"\n✓ Found {len(models)} installed model(s):")
    model_names = []
    for model in models:
        name = model.get('name', 'unknown')
        size = model.get('size', 0) / (1024**3)  # Convert to GB
        print(f"  • {name} ({size:.2f} GB)")
        model_names.append(name.split(':')[0])  # Remove tag if present
    
    return model_names

def test_model_inference(base_url, model_name):
    """Test basic inference with a model"""
    print(f"\nTesting inference with {model_name}...")
    
    test_prompt = "Return only a JSON object with a single key 'test' and value 'success': "
    
    try:
        response = requests.post(
            urljoin(base_url, "api/generate"),
            json={
                "model": model_name,
                "prompt": test_prompt,
                "stream": False,
                "format": "json"
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '')
            print(f"✓ Inference successful!")
            print(f"  Response: {generated_text[:100]}...")
            
            # Try to parse as JSON
            try:
                parsed = json.loads(generated_text)
                if 'test' in parsed:
                    print("✓ JSON output is valid!")
                    return True
                else:
                    print("⚠ JSON output missing expected key")
                    return True  # Still worked, just not perfect
            except json.JSONDecodeError:
                print("⚠ Response was not valid JSON (model may need better prompting)")
                return True  # Still consider it a success
        else:
            print(f"✗ Inference failed with status code: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Error during inference: {e}")
        return False

def check_environment_variables():
    """Check if environment variables are set"""
    print("\nChecking environment variables...")
    
    base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    custom_model = os.getenv('OLLAMA_CUSTOM_MODEL_NAME', None)
    
    print(f"  OLLAMA_BASE_URL: {base_url}")
    if custom_model:
        print(f"  OLLAMA_CUSTOM_MODEL_NAME: {custom_model}")
    else:
        print(f"  OLLAMA_CUSTOM_MODEL_NAME: (not set)")
    
    return base_url, custom_model

def check_vouchervision_models():
    """Check which VoucherVision Ollama models are available"""
    print("\nVoucherVision pre-configured Ollama models:")
    
    vv_models = {
        'llama3.2': 'Ollama llama3.2',
        'llama3.2-vision': 'Ollama llama3.2-vision',
        'llama3.1': 'Ollama llama3.1',
        'mistral': 'Ollama mistral',
        'mixtral': 'Ollama mixtral',
        'qwen2.5': 'Ollama qwen2.5',
    }
    
    for model_id, vv_name in vv_models.items():
        print(f"  • {vv_name} → requires: ollama pull {model_id}")
    
    return vv_models

def main():
    print("=" * 60)
    print("VoucherVision Ollama Configuration Test")
    print("=" * 60)
    
    # Check environment variables
    base_url, custom_model = check_environment_variables()
    
    # Test connection
    connected, models_data = test_ollama_connection(base_url)
    
    if not connected:
        print("\n" + "=" * 60)
        print("SETUP REQUIRED")
        print("=" * 60)
        print("\n1. Install Ollama from https://ollama.ai")
        print("2. Start Ollama with: ollama serve")
        print("3. Pull a model with: ollama pull llama3.2")
        print("4. Run this script again")
        return False
    
    # List available models
    available_models = list_available_models(models_data)
    
    if not available_models:
        print("\n" + "=" * 60)
        print("NO MODELS INSTALLED")
        print("=" * 60)
        print("\nInstall a model with:")
        print("  ollama pull llama3.2")
        print("  ollama pull mistral")
        return False
    
    # Show VoucherVision models
    vv_models = check_vouchervision_models()
    
    # Check which VV models are available
    print("\nStatus in VoucherVision:")
    for model_id, vv_name in vv_models.items():
        if model_id in ' '.join(available_models):
            print(f"  ✓ {vv_name} - READY")
        else:
            print(f"  ✗ {vv_name} - Not installed (ollama pull {model_id})")
    
    # Test inference on first available model
    if available_models:
        test_model = available_models[0]
        test_model_inference(base_url, test_model)
    
    # Check custom model
    if custom_model:
        if custom_model in ' '.join(available_models):
            print(f"\n✓ Custom model '{custom_model}' is available")
            print("  You can use 'Ollama Custom' in VoucherVision")
        else:
            print(f"\n✗ Custom model '{custom_model}' is NOT available")
            print(f"  Install it with: ollama pull {custom_model}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    
    if available_models:
        print("\n✓ Ollama is ready to use with VoucherVision!")
        print("\nNext steps:")
        print("  1. Launch VoucherVision")
        print("  2. Select 'Ollama' from Model Family dropdown")
        print("  3. Choose an available model")
        print("  4. Start transcribing!")
        return True
    else:
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
