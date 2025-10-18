#!/usr/bin/env python3
"""
Model optimization script to reduce file size for Railway deployment
"""

import torch
import os
import gzip
from pathlib import Path

def compress_model(input_path, output_path):
    """Compress model using gzip"""
    print(f"Compressing {input_path}...")
    
    # Load model
    model = torch.load(input_path, map_location='cpu')
    
    # Compress and save
    with gzip.open(output_path, 'wb') as f:
        torch.save(model, f)
    
    # Check sizes
    original_size = os.path.getsize(input_path)
    compressed_size = os.path.getsize(output_path)
    compression_ratio = (1 - compressed_size / original_size) * 100
    
    print(f"Original size: {original_size / (1024*1024):.1f} MB")
    print(f"Compressed size: {compressed_size / (1024*1024):.1f} MB")
    print(f"Compression: {compression_ratio:.1f}%")
    
    return compressed_size < 100 * 1024 * 1024  # Check if under 100MB

def quantize_model(input_path, output_path):
    """Quantize model to reduce size"""
    print(f"Quantizing {input_path}...")
    
    # Load model
    model = torch.load(input_path, map_location='cpu')
    
    # Quantize (if it's a PyTorch model)
    if hasattr(model, 'eval'):
        model.eval()
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        torch.save(quantized_model, output_path)
        
        # Check sizes
        original_size = os.path.getsize(input_path)
        quantized_size = os.path.getsize(output_path)
        reduction = (1 - quantized_size / original_size) * 100
        
        print(f"Original size: {original_size / (1024*1024):.1f} MB")
        print(f"Quantized size: {quantized_size / (1024*1024):.1f} MB")
        print(f"Size reduction: {reduction:.1f}%")
        
        return quantized_size < 100 * 1024 * 1024

def main():
    model_path = "assets/transformer_chatbot_gpu_deco_2.pth"
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    print("🔧 Model Optimization Options:")
    print("1. Compress with gzip")
    print("2. Quantize model")
    print("3. Both")
    
    choice = input("Choose option (1-3): ").strip()
    
    if choice == "1":
        compressed_path = "assets/transformer_chatbot_gpu_deco_2_compressed.pth.gz"
        success = compress_model(model_path, compressed_path)
        if success:
            print("✅ Model compressed successfully!")
            print(f"Use {compressed_path} in your deployment")
        else:
            print("❌ Model still too large after compression")
    
    elif choice == "2":
        quantized_path = "assets/transformer_chatbot_gpu_deco_2_quantized.pth"
        success = quantize_model(model_path, quantized_path)
        if success:
            print("✅ Model quantized successfully!")
            print(f"Use {quantized_path} in your deployment")
        else:
            print("❌ Model still too large after quantization")
    
    elif choice == "3":
        # Try compression first
        compressed_path = "assets/transformer_chatbot_gpu_deco_2_compressed.pth.gz"
        if compress_model(model_path, compressed_path):
            print("✅ Compression successful!")
        else:
            # Try quantization
            quantized_path = "assets/transformer_chatbot_gpu_deco_2_quantized.pth"
            if quantize_model(model_path, quantized_path):
                print("✅ Quantization successful!")
            else:
                print("❌ Model still too large. Consider using Vast.ai instead.")
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
