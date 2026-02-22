import requests
import json
import base64
from PIL import Image
import io

# Load a test image from the dataset
img_path = r"C:\Users\User\Documents\Damian\Swinburne\small dataset\5005.jpg"
image = Image.open(img_path)

# Encode image to base64
buffered = io.BytesIO()
if image.mode != 'RGB':
    image = image.convert('RGB')
image.save(buffered, format="JPEG", quality=95)
image_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

# Make request to Ollama
base_url = "http://localhost:11434"
api_url = f"{base_url}/api/generate"

payload = {
    "model": "qwen3-vl:4b",
    "prompt": "Extract and transcribe all text visible in this image. Include all labels, printed text, handwritten text, numbers, dates, and location information. Return only the raw text without any explanation.",
    "images": [image_b64],
    "stream": False,
    "options": {
        "temperature": 0.1,
        "top_p": 0.9,
        "num_predict": 2048,
    }
}

print("Sending request to Ollama...")
response = requests.post(api_url, json=payload, timeout=120)
print(f"Status code: {response.status_code}")
print("\nFull response JSON:")
result = response.json()
print(json.dumps(result, indent=2))

print("\n" + "="*60)
print("EXTRACTED TEXT FROM 'response' FIELD:")
print("="*60)
text = result.get('response', '').strip()
print(f"Length: {len(text)} characters")
print(text)
