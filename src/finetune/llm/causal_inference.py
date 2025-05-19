#!/usr/bin/env python3
"""
Script to generate summaries using fine-tuned causal language models (LLMs).
Supports loading PEFT/LoRA adapters for efficient inference.
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
from peft import PeftModel, PeftConfig, AutoPeftModelForCausalLM

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


def setup_model_and_tokenizer(model_path: str, device_str: str, use_8bit=False, use_4bit=False):
    """
    Set up the model and tokenizer for inference.
    
    Args:
        model_path: Path to the model or adapter
        device_str: Device to use
        use_8bit: Whether to use 8-bit quantization
        use_4bit: Whether to use 4-bit quantization
        
    Returns:
        Tokenizer and model
    """
    model_path = Path(model_path)
    
    # Check if it's a PEFT/LoRA model
    adapter_config_path = model_path / "adapter_config.json"
    
    if adapter_config_path.exists():
        print(f"Loading PEFT adapter from {model_path}")
        
        # Load the adapter config to get the base model
        config = PeftConfig.from_pretrained(model_path)
        base_model_name = config.base_model_name_or_path
        
        print(f"Base model: {base_model_name}")
        
        # Set quantization if running on CUDA
        model_kwargs = {}
        if device_str.startswith("cuda"):
            if use_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                    model_kwargs = {
                        "quantization_config": BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                        ),
                        "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                    }
                    print("Using 4-bit quantization")
                except ImportError:
                    print("Failed to load BitsAndBytesConfig, falling back to 8-bit")
                    model_kwargs = {"load_in_8bit": True}
            elif use_8bit:
                model_kwargs = {"load_in_8bit": True}
                print("Using 8-bit quantization")
            else:
                model_kwargs = {"torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
                print(f"Using {'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'} precision")
        
        # Load the base model and tokenizer
        try:
            # Try the simpler loading method first
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto" if device_str.startswith("cuda") else None,
                **model_kwargs
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            print(f"Error with direct loading: {e}")
            print("Trying alternative loading method...")
            
            # Load base model and tokenizer first, then adapter
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map="auto" if device_str.startswith("cuda") else None,
                **model_kwargs
            )
            
            # Load PEFT adapter
            model = PeftModel.from_pretrained(
                base_model,
                model_path,
                device_map="auto" if device_str.startswith("cuda") else None
            )
        
        # Ensure padding token is set
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        
        # If not using device_map="auto", manually move to device
        if not device_str.startswith("cuda") and model.device.type != device_str:
            model = model.to(device_str)
        
    else:
        # Regular model (not PEFT)
        print(f"Loading regular model from {model_path}")
        
        # Set quantization if running on CUDA
        model_kwargs = {}
        if device_str.startswith("cuda"):
            if use_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                    model_kwargs = {
                        "quantization_config": BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                        ),
                        "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                    }
                    print("Using 4-bit quantization")
                except ImportError:
                    print("Failed to load BitsAndBytesConfig, falling back to 8-bit")
                    model_kwargs = {"load_in_8bit": True}
            elif use_8bit:
                model_kwargs = {"load_in_8bit": True}
                print("Using 8-bit quantization")
            else:
                model_kwargs = {"torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
                print(f"Using {'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'} precision")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Ensure padding token is set
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto" if device_str.startswith("cuda") else None,
            **model_kwargs
        )
        
        # If not using device_map="auto", manually move to device
        if not device_str.startswith("cuda") and model.device.type != device_str:
            model = model.to(device_str)
    
    return tokenizer, model


def setup_generation_pipeline(model, tokenizer, device_id):
    """
    Set up the text generation pipeline.
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer to use
        device_id: The device ID to use
        
    Returns:
        Text generation pipeline
    """
    # Print some info about the model
    print(f"Model type: {model.__class__.__name__}")
    print(f"Model device: {model.device}")
    
    # Create text generation pipeline
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device_id
    )
    
    return text_generator


def generate_summaries(
    articles: List[Dict[str, Any]],
    text_generator,
    model_key: str,
    max_length: int = 150,
    min_length: int = 40,
    temperature: float = 0.7,
    num_beams: int = 1
) -> List[Dict[str, Any]]:
    """
    Generate summaries for each article using the causal LLM.
    
    Args:
        articles: List of article dictionaries
        text_generator: Text generation pipeline
        model_key: Key to identify the model type for prompt formatting
        max_length: Maximum length of the summary in tokens
        min_length: Minimum length of the summary in tokens
        temperature: Generation temperature (higher = more random)
        num_beams: Number of beams for beam search (1 = greedy)
        
    Returns:
        List of article dictionaries with summaries added
    """
    articles_with_summaries = []
    
    # Get the appropriate prompt template
    prompt_template = PROMPT_TEMPLATES.get(model_key, PROMPT_TEMPLATES["default"])
    
    for article in tqdm(articles, desc="Generating summaries with fine-tuned LLM"):
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
                "min_new_tokens": min_length if num_beams == 1 else None,  # min_length doesn't work well with beam search
                "do_sample": temperature > 0,
                "temperature": temperature,
                "num_beams": num_beams,
                "top_p": 0.9,
                "top_k": 50,
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
            # Remove special tokens at the end that might appear
            for token in ["<|", "</s>", "<end_of_turn>", "<|endoftext|>"]:
                if token in summary:
                    summary = summary.split(token)[0].strip()
            
            # Add summary to article
            article_with_summary = article.copy()
            article_with_summary["finetuned_llm_summary"] = summary
            
            # Keep existing summaries if present
            if "baseline_summary" in article:
                article_with_summary["baseline_summary"] = article["baseline_summary"]
            if "llm_summary" in article:
                article_with_summary["llm_summary"] = article["llm_summary"]
                
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
    parser = argparse.ArgumentParser(description="Generate summaries using fine-tuned causal LLMs")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file containing processed articles")
    parser.add_argument("--output", type=str, help="Output JSON file for articles with summaries")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the fine-tuned model or adapter")
    parser.add_argument("--model-key", type=str, choices=list(SMALL_LLMS.keys()) + ["default"], default="default",
                        help="Model type to determine the prompt template")
    parser.add_argument("--device", type=int, default=-1, help="Device to use (-1 for auto, 0+ for specific GPU)")
    parser.add_argument("--max-length", type=int, default=150, help="Maximum length of the summary in tokens")
    parser.add_argument("--min-length", type=int, default=40, help="Minimum length of the summary in tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature (0 = deterministic)")
    parser.add_argument("--num-beams", type=int, default=1, help="Number of beams for beam search (1 = greedy)")
    parser.add_argument("--use-8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--use-4bit", action="store_true", help="Use 4-bit quantization")
    args = parser.parse_args()
    
    # Set default output file if not provided
    if not args.output:
        input_path = Path(args.input)
        args.output = str(DATA_DIR / f"{input_path.stem}_finetuned_llm_summaries.json")
    
    # Get best device
    device_str, device_id = get_best_device(args.device)
    
    # Load articles
    articles = load_articles(args.input)
    
    # Set up model and tokenizer
    tokenizer, model = setup_model_and_tokenizer(
        args.model_path, 
        device_str, 
        use_8bit=args.use_8bit, 
        use_4bit=args.use_4bit
    )
    
    # Set up generation pipeline
    generator = setup_generation_pipeline(model, tokenizer, device_id)
    
    # Generate summaries
    articles_with_summaries = generate_summaries(
        articles,
        generator,
        args.model_key,
        max_length=args.max_length,
        min_length=args.min_length,
        temperature=args.temperature,
        num_beams=args.num_beams
    )
    
    # Save articles with summaries
    save_articles_with_summaries(articles_with_summaries, args.output)


if __name__ == "__main__":
    main() 