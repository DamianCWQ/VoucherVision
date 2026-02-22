"""Quick test to verify qwen3-vl OCR now works"""
import sys
sys.path.insert(0, r'C:\Users\User\Documents\GitHub\VoucherVision')

from vouchervision.OCR_Ollama import OllamaVisionOCR

# Initialize OCR
print("Initializing Ollama OCR with qwen3-vl:4b...")
ocr = OllamaVisionOCR(model_name='qwen3-vl:4b')

# Test with an image
img_path = r"C:\Users\User\Documents\Damian\Swinburne\small dataset\5005.jpg"
print(f"\nProcessing image: {img_path}")

# Perform OCR
text, full_response, usage = ocr.ocr_with_ollama(img_path)

print("\n" + "="*60)
print("OCR RESULT:")
print("="*60)
print(f"Characters extracted: {len(text)}")
print(f"Tokens used: {usage['total_tokens']}")
print(f"Time: {usage['total_time_seconds']:.2f}s")
print("\nExtracted text (first 500 chars):")
print(text[:500] if len(text) > 500 else text)
print("\n" + "="*60)

if len(text) > 0:
    print("✅ SUCCESS! OCR is now working!")
else:
    print("❌ FAILED! OCR still returning empty text.")
