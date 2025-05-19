#!/usr/bin/env python3
"""
Script to generate summaries using small instruction-tuned causal language models (LLMs).
These are decoder-only models rather than encoder-decoder models like BART/T5.
"""

import os
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Ensure data directory exists
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Define small instruction-tuned models
SMALL_LLMS = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 1.1B
    "phi2": "microsoft/phi-2",                          # 2.7B
    "phi3": "microsoft/phi-3-mini-4k-instruct",         # 3.8B
    "gemma": "google/gemma-2b-it",                      # 2.5B
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",    # 8B (if GPU memory allows)
}

# Define prompts for different model types
PROMPT_TEMPLATES = {
    "tinyllama": "<|system|>\nYou are a helpful assistant that summarizes news articles accurately and concisely.\n<|user|>\nSummarize the following news article in a few sentences:\n\n{text}\n<|assistant|>",
    
    "phi2": "Instruction: Summarize the following news article in a few sentences.\n\nInput: {text}\n\nOutput:",
    
    "phi3": "<|system|>\nYou are a helpful assistant that summarizes news articles accurately and concisely.\n<|user|>\nSummarize the following news article in a few sentences:\n\n{text}\n<|assistant|>",
    
    "gemma": "<start_of_turn>user\nSummarize the following news article in a few sentences:\n\n{text}<end_of_turn>\n<start_of_turn>model\n",
    
    "llama3": "<|system|>\nYou are a helpful assistant that summarizes news articles accurately and concisely.\n<|user|>\nSummarize the following news article in a few sentences:\n\n{text}\n<|assistant|>",
    
    "default": "Summarize the following news article in a few sentences:\n\n{text}\n\nSummary:"
}

def load_articles(input_file: str) -> List[Dict[str, Any]]:
    """
    Load processed articles from a JSON file.
    
    Args:
        input_file: Path to the JSON file containing processed articles
        
    Returns:
        List of article dictionaries
    """
    with open(input_file, 'r') as f:
        articles = json.load(f)
    
    print(f"Loaded {len(articles)} articles from {input_file}")
    return articles


def get_best_device(device_param=-1):
    """
    Get the best available device.
    
    Args:
        device_param: User-specified device (-1 for auto-detect, 0+ for specific GPU)
        
    Returns:
        Device string and numeric device id
    """
    # If user specified a specific GPU device, try to use it
    if device_param >= 0 and torch.cuda.is_available() and device_param < torch.cuda.device_count():
        device_str = f"cuda:{device_param}"
        print(f"Using specified GPU device: {device_str} ({torch.cuda.get_device_name(device_param)})")
        return device_str, device_param
    
    # Auto-detection: Check for CUDA GPUs
    if torch.cuda.is_available():
        device_str = "cuda:0"
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return device_str, 0
    
    # Check for MPS (Apple Silicon)
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device_str = "mps"
        print("Using Apple Silicon MPS (Metal Performance Shaders)")
        return device_str, "mps"
    
    # Fallback to CPU
    print("No GPU detected, using CPU for inference (this will be slower)")
    return "cpu", -1


def setup_llm_pipeline(model_name: str, device: int = -1) -> pipeline:
    """
    Set up the Hugging Face text generation pipeline for causal LLM.
    
    Args:
        model_name: Name of the LLM to use
        device: Device to use (-1 for CPU, 0+ for specific GPU)
        
    Returns:
        Hugging Face pipeline for text generation
    """
    print(f"Loading LLM: {model_name}")
    
    # Determine device
    device_str, device_id = get_best_device(device)
    
    # Load tokenizer and model with low-memory options
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set up low-memory options based on available resources
    model_kwargs = {}
    
    # If using GPU, try to use BF16 or 8-bit precision if available
    if device_str.startswith("cuda"):
        try:
            from transformers import BitsAndBytesConfig
            model_kwargs = {
                "quantization_config": BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True
                ),
                "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            }
            print(f"Using 8-bit quantization with {'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'}")
        except ImportError:
            model_kwargs = {"torch_dtype": torch.float16}
            print("Using float16 precision (install bitsandbytes for 8-bit quantization)")
    
    # Set padding side right for causal language models
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_str if device_str.startswith("cuda") else None,
            **model_kwargs
        )
        
        # If not using GPU device mapping, manually move to device
        if not device_str.startswith("cuda"):
            model = model.to(device_str)
        
        # Create text generation pipeline
        text_generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device_id
        )
        
        return text_generator
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise


def generate_summaries(
    articles: List[Dict[str, Any]],
    text_generator,
    model_key: str,
    max_length: int = 150,
    min_length: int = 40
) -> List[Dict[str, Any]]:
    """
    Generate summaries for each article using the causal LLM.
    
    Args:
        articles: List of article dictionaries
        text_generator: Hugging Face text generation pipeline
        model_key: Key identifier for the model (to select the right prompt)
        max_length: Maximum length of the summary in tokens
        min_length: Minimum length of the summary in tokens
        
    Returns:
        List of article dictionaries with summaries added
    """
    articles_with_summaries = []
    
    # Get the appropriate prompt template
    prompt_template = PROMPT_TEMPLATES.get(model_key, PROMPT_TEMPLATES["default"])
    
    for article in tqdm(articles, desc="Generating summaries with LLM"):
        # Get the text to summarize (prefer cleaned_text if available)
        text = article.get("cleaned_text", article.get("text", ""))
        
        # Skip articles without text
        if not text:
            continue
        
        # Create prompt from template
        prompt = prompt_template.format(text=text[:3000])  # Truncate to avoid exceeding context limits
            
        # Generate summary
        try:
            generation_kwargs = {
                "max_new_tokens": max_length,
                "min_new_tokens": min_length,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "pad_token_id": text_generator.tokenizer.pad_token_id
            }
            
            outputs = text_generator(
                prompt,
                **generation_kwargs
            )
            
            # Extract generated text
            generated_text = outputs[0]['generated_text']
            
            # Remove the prompt from the generated text
            summary = generated_text[len(prompt):].strip()
            
            # Clean up the summary
            summary = summary.split("<|")[0].strip()  # Remove any special tokens that might appear
            
            # Add summary to article
            article_with_summary = article.copy()
            article_with_summary["llm_summary"] = summary
            
            articles_with_summaries.append(article_with_summary)
            
            # Slight delay to avoid overloading GPU
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error generating summary for article '{article.get('title', 'Unknown')}': {str(e)}")
        
    return articles_with_summaries


def save_articles_with_summaries(articles: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save articles with summaries to a JSON file.
    
    Args:
        articles: List of article dictionaries with summaries
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        json.dump(articles, f, indent=2)
        
    print(f"Saved {len(articles)} articles with summaries to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate summaries using causal language models (LLMs)")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file containing processed articles")
    parser.add_argument("--output", type=str, help="Output JSON file for articles with summaries")
    parser.add_argument("--model-name", type=str, help="Hugging Face model name for custom LLM")
    parser.add_argument("--small-llm", type=str, choices=list(SMALL_LLMS.keys()), 
                       help=f"Small LLM to use: {', '.join(SMALL_LLMS.keys())}")
    parser.add_argument("--device", type=int, default=-1, help="Device to use (-1 for auto, 0+ for specific GPU)")
    parser.add_argument("--max-length", type=int, default=150, help="Maximum length of the summary in tokens")
    parser.add_argument("--min-length", type=int, default=40, help="Minimum length of the summary in tokens")
    args = parser.parse_args()
    
    # Set default output file if not provided
    if not args.output:
        input_path = Path(args.input)
        args.output = str(DATA_DIR / f"{input_path.stem}_llm_summaries.json")
    
    # Determine model to use
    if args.small_llm:
        model_name = SMALL_LLMS[args.small_llm]
        model_key = args.small_llm
        print(f"Using small LLM: {args.small_llm} ({model_name})")
    elif args.model_name:
        model_name = args.model_name
        model_key = "default"
    else:
        model_name = SMALL_LLMS["phi2"]  # Default to Phi-2 as a good small model
        model_key = "phi2"
        print(f"Using default small LLM: phi2 ({model_name})")
    
    # Load articles
    articles = load_articles(args.input)
    
    # Set up LLM pipeline
    text_generator = setup_llm_pipeline(model_name, args.device)
    
    # Generate summaries
    articles_with_summaries = generate_summaries(
        articles,
        text_generator,
        model_key,
        max_length=args.max_length,
        min_length=args.min_length
    )
    
    # Save articles with summaries
    save_articles_with_summaries(articles_with_summaries, args.output)


if __name__ == "__main__":
    main() 