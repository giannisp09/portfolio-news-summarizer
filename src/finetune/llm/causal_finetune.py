#!/usr/bin/env python3
"""
Script to fine-tune a small instruction-tuned causal language model (LLM) for summarization.
Uses Parameter-Efficient Fine-Tuning (PEFT) with LoRA or QLoRA for memory efficiency.
"""

import os
import json
import argparse
import random
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List

import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import wandb

# Check for required packages and install if missing
required_packages = {
    "peft": "0.7.0",
    "transformers": "4.37.0", 
    "bitsandbytes": "0.41.1",
    "accelerate": "0.25.0",
    "datasets": "2.16.0",
}

def check_install_package(package, version=None):
    """Check if package is installed, install if not."""
    try:
        __import__(package)
        print(f"Package {package} is already installed.")
    except ImportError:
        print(f"Package {package} not found. Installing...")
        if version:
            subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}=={version}"])
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Package {package} has been installed.")

# Check and install required packages
for package, version in required_packages.items():
    check_install_package(package, version)

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    PeftConfig,
    prepare_model_for_kbit_training
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Ensure models directory exists
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Define small instruction-tuned models
SMALL_LLMS = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 1.1B
    "phi2": "microsoft/phi-2",                          # 2.7B
    "phi3": "microsoft/phi-3-mini-4k-instruct",         # 3.8B
    "gemma": "google/gemma-2b-it",                      # 2.5B
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",  
    "smollm1.7" : "HuggingFaceTB/SmolLM2-1.7B-Instruct" # 8B (if GPU memory allows)
}

# Default model
DEFAULT_MODEL = "microsoft/phi-2"

# Define prompts for different model types
PROMPT_TEMPLATES = {
    "tinyllama": "<|system|>\nYou are a helpful assistant that summarizes news articles accurately and concisely.\n<|user|>\nSummarize the following news article in a few sentences:\n\n{text}\n<|assistant|>\n{summary}",
    
    "phi2": "Instruction: Summarize the following news article in a few sentences.\n\nInput: {text}\n\nOutput: {summary}",
    
    "phi3": "<|system|>\nYou are a helpful assistant that summarizes news articles accurately and concisely.\n<|user|>\nSummarize the following news article in a few sentences:\n\n{text}\n<|assistant|>\n{summary}",
    
    "gemma": "<start_of_turn>user\nSummarize the following news article in a few sentences:\n\n{text}<end_of_turn>\n<start_of_turn>model\n{summary}<end_of_turn>",
    
    "llama3": "<|system|>\nYou are a helpful assistant that summarizes news articles accurately and concisely.\n<|user|>\nSummarize the following news article in a few sentences:\n\n{text}\n<|assistant|>\n{summary}",
    
    "default": "Summarize the following news article in a few sentences:\n\n{text}\n\nSummary: {summary}"
}

def get_best_device(device_param=-1):
    """
    Get the best available device for training.
    
    Args:
        device_param: User-specified device (-1 for auto-detect, 0+ for specific GPU/device)
        
    Returns:
        Device string for torch and HF
    """
    # If user specified a specific GPU device, try to use it
    if device_param >= 0 and torch.cuda.is_available() and device_param < torch.cuda.device_count():
        device_str = f"cuda:{device_param}"
        logger.info(f"Using specified GPU device: {device_str} ({torch.cuda.get_device_name(device_param)})")
        return device_str
    
    # Auto-detection: Check for CUDA GPUs
    if torch.cuda.is_available():
        device_str = "cuda:0"
        logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return device_str
    
    # Check for MPS (Apple Silicon)
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device_str = "mps"
        logger.info("Using Apple Silicon MPS (Metal Performance Shaders)")
        return device_str
    
    # Fallback to CPU
    logger.warning("No GPU detected, using CPU for training (this will be very slow)")
    return "cpu"


def load_data_from_json(json_file: str, split_ratio: Dict[str, float] = None, max_samples: int = None) -> Dict[str, Dataset]:
    """
    Load data for fine-tuning from a JSON file.
    
    Args:
        json_file: Path to JSON file containing the dataset
        split_ratio: Dictionary with split ratios for train, val, and test
                     (defaults to 0.8, 0.1, 0.1)
        max_samples: Maximum number of samples to use
        
    Returns:
        Dictionary of datasets for training, validation, and test
    """
    if split_ratio is None:
        split_ratio = {"train": 0.8, "val": 0.1, "test": 0.1}
    
    # Check that split ratios sum to 1
    total_ratio = sum(split_ratio.values())
    if abs(total_ratio - 1.0) > 1e-6:
        logger.warning(f"Split ratios sum to {total_ratio}, not 1.0")
        # Normalize ratios
        split_ratio = {k: v / total_ratio for k, v in split_ratio.items()}
    
    logger.info(f"Loading data from JSON file: {json_file}")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON formats
    if isinstance(data, list):
        # Direct list of examples
        examples = data
    elif isinstance(data, dict) and "articles" in data:
        # Dictionary with "articles" key
        examples = data["articles"]
    elif isinstance(data, dict) and all(isinstance(data[k], list) for k in data):
        # Dictionary with split keys
        datasets = {}
        for split, items in data.items():
            df = pd.DataFrame(items)
            
            # Ensure required columns exist
            required_cols = {"text", "summary"}
            if not all(col in df.columns for col in required_cols):
                # Try to map columns if possible
                if "article" in df.columns and "text" not in df.columns:
                    df["text"] = df["article"]
                if "title" in df.columns and "summary" not in df.columns:
                    df["summary"] = df["title"]
                
                # Check for other possible column mappings
                text_candidates = ["text", "article", "content", "body", "cleaned_text"]
                summary_candidates = ["summary", "headline", "title", "labeled_summary", "reference_summary", "ground_truth"]
                
                for candidate in text_candidates:
                    if candidate in df.columns and "text" not in df.columns:
                        df["text"] = df[candidate]
                        break
                        
                for candidate in summary_candidates:
                    if candidate in df.columns and "summary" not in df.columns:
                        df["summary"] = df[candidate]
                        break
            
            # Ensure required columns exist after mapping
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"JSON data must contain 'text' and 'summary' fields for split {split}")
            
            if max_samples is not None and len(df) > max_samples and split == "train":
                df = df.sample(max_samples, random_state=42)
                
            dataset = Dataset.from_pandas(df)
            datasets[split] = dataset
            logger.info(f"Loaded {len(dataset)} examples for {split}")
        
        return datasets
    else:
        raise ValueError("Unrecognized JSON format. Expected a list of examples or a dictionary with 'articles' key.")
    
    # Convert to DataFrame
    df = pd.DataFrame(examples)
    
    # Ensure required columns exist
    required_cols = {"text", "summary"}
    if not all(col in df.columns for col in required_cols):
        # Try to map columns if possible
        if "article" in df.columns and "text" not in df.columns:
            df["text"] = df["article"]
        if "title" in df.columns and "summary" not in df.columns:
            df["summary"] = df["title"]
        
        # Check for other possible column mappings
        text_candidates = ["text", "article", "content", "body", "cleaned_text"]
        summary_candidates = ["summary", "headline", "title", "labeled_summary", "reference_summary", "ground_truth"]
        
        for candidate in text_candidates:
            if candidate in df.columns and "text" not in df.columns:
                df["text"] = df[candidate]
                break
                
        for candidate in summary_candidates:
            if candidate in df.columns and "summary" not in df.columns:
                df["summary"] = df[candidate]
                break
    
    # Ensure required columns exist after mapping
    if not all(col in df.columns for col in required_cols):
        raise ValueError("JSON data must contain 'text' and 'summary' fields")
    
    # Apply max_samples
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(max_samples, random_state=42)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split the data
    datasets = {}
    n = len(df)
    train_end = int(n * split_ratio["train"])
    val_end = train_end + int(n * split_ratio["val"])
    
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    # Convert to Hugging Face datasets
    datasets["train"] = Dataset.from_pandas(train_df)
    datasets["val"] = Dataset.from_pandas(val_df)
    datasets["test"] = Dataset.from_pandas(test_df)
    
    logger.info(f"Loaded {len(datasets['train'])} examples for train")
    logger.info(f"Loaded {len(datasets['val'])} examples for val")
    logger.info(f"Loaded {len(datasets['test'])} examples for test")
    
    return datasets


def preprocess_data(
    examples: Dict[str, List], 
    tokenizer, 
    max_length: int, 
    prompt_template: str, 
    is_llama_model: bool = False
):
    """
    Preprocess data by formatting and tokenizing it.
    """
    # Create prompts from template
    prompts = []
    for i in range(len(examples["text"])):
        text = examples["text"][i]
        summary = examples["summary"][i]
        
        # Skip examples with missing text or summary
        if not text or not summary:
            continue
            
        # Format the prompt
        prompt = prompt_template.format(
            text=text[:3000],  # Truncate text to fit model context
            summary=summary
        )
        prompts.append(prompt)
    
    # Tokenize the prompts
    tokenized_inputs = tokenizer(
        prompts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # For causal language models, we need labels to be the same as input_ids
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
    
    # If it's a LLaMA-like model with specific padding handling
    if is_llama_model:
        # Set labels for padding tokens to -100 so they're ignored in loss calculation
        tokenized_inputs["labels"][tokenized_inputs["attention_mask"] == 0] = -100
    
    return tokenized_inputs


def setup_model_and_tokenizer(model_name, use_4bit=False, use_8bit=False, device="cpu"):
    """
    Load and set up the model and tokenizer with efficient settings.
    """
    logger.info(f"Loading model and tokenizer: {model_name}")
    
    # Set up quantization config if requested
    quantization_config = None
    
    # Only apply quantization for CUDA devices
    if device.startswith("cuda"):
        if use_4bit:
            logger.info("Using 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif use_8bit:
            logger.info("Using 8-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=["lm_head"]
            )
    
    # Set appropriate model loading parameters
    model_kwargs = {}
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    
    # Set appropriate torch dtype based on device
    if device.startswith("cuda"):
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if needed
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device.startswith("cuda") else None,
        **model_kwargs
    )
    
    # If using quantization, prepare model for k-bit training
    if (use_4bit or use_8bit) and device.startswith("cuda"):
        model = prepare_model_for_kbit_training(model)
    
    # If not using automatic device mapping, move to device manually
    if not device.startswith("cuda") and device != "auto":
        model = model.to(device)
    
    return model, tokenizer


def setup_lora_config(
    r=8, 
    lora_alpha=16, 
    lora_dropout=0.1, 
    target_modules=None,
    use_8bit=False, 
    use_4bit=False,
    bias="none"
):
    """
    Set up the LoRA configuration.
    """
    # Set default target modules if not provided
    if target_modules is None:
        # Different models have different naming conventions for attention modules
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    logger.info(f"Setting up LoRA with r={r}, alpha={lora_alpha}, dropout={lora_dropout}")
    logger.info(f"Target modules: {target_modules}")
    
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    
    return lora_config


def compute_metrics(eval_preds):
    """
    Compute basic metrics for evaluation.
    
    This is a simple implementation - in practice you might want to use ROUGE, etc.
    for more comprehensive evaluation.
    """
    # Simple loss-based metric for tracking
    # A more comprehensive evaluation would use actual summarization metrics
    return {"loss": eval_preds.loss}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a causal LLM for summarization")
    
    # Data arguments
    parser.add_argument("--data-file", type=str, help="Path to JSON file containing training data")
    parser.add_argument("--hf-dataset", type=str, help="Hugging Face dataset name")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to use for training")
    
    # Model arguments
    parser.add_argument("--model", type=str, help="Full model name/path")
    parser.add_argument("--small-llm", type=str, choices=list(SMALL_LLMS.keys()), 
                        help=f"Small LLM variant to use: {', '.join(SMALL_LLMS.keys())}")
    parser.add_argument("--output-dir", type=str, default="models/finetuned_llm", 
                        help="Directory to save fine-tuned model")
    
    # LoRA configuration
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA attention dimension")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha parameter")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout probability")
    
    # Quantization options
    parser.add_argument("--use-8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--use-4bit", action="store_true", help="Use 4-bit quantization (QLoRA)")
    
    # Training hyperparameters
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--device", type=int, default=-1, help="Device to use (-1 for auto)")
    
    # Misc arguments
    parser.add_argument("--use-wandb", action="store_true", help="Log with Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="news-llm-summarizer", help="W&B project name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Determine device
    device = get_best_device(args.device)
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save arguments
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Determine which model to use
    model_name = DEFAULT_MODEL
    model_key = "phi2"  # Default
    
    if args.small_llm:
        model_name = SMALL_LLMS[args.small_llm]
        model_key = args.small_llm
        logger.info(f"Using small LLM: {args.small_llm} ({model_name})")
    elif args.model:
        model_name = args.model
        model_key = "default"
        logger.info(f"Using custom model: {model_name}")
    
    # Get prompt template
    prompt_template = PROMPT_TEMPLATES.get(model_key, PROMPT_TEMPLATES["default"])
    
    # Load dataset
    if args.data_file:
        datasets = load_data_from_json(args.data_file, max_samples=args.max_samples)
    elif args.hf_dataset:
        from datasets import load_dataset
        
        # Load from Hugging Face datasets hub
        try:
            raw_datasets = load_dataset(args.hf_dataset)
            
            # Map to expected format
            for split in raw_datasets:
                if "text" not in raw_datasets[split].column_names or "summary" not in raw_datasets[split].column_names:
                    # Try to find appropriate columns
                    cols = raw_datasets[split].column_names
                    
                    if "document" in cols and "text" not in cols:
                        raw_datasets[split] = raw_datasets[split].rename_column("document", "text")
                    elif "article" in cols and "text" not in cols:
                        raw_datasets[split] = raw_datasets[split].rename_column("article", "text")
                        
                    if "highlights" in cols and "summary" not in cols:
                        raw_datasets[split] = raw_datasets[split].rename_column("highlights", "summary")
                    elif "headline" in cols and "summary" not in cols:
                        raw_datasets[split] = raw_datasets[split].rename_column("headline", "summary")
            
            # Apply max_samples if needed
            if args.max_samples and "train" in raw_datasets:
                raw_datasets["train"] = raw_datasets["train"].select(range(min(len(raw_datasets["train"]), args.max_samples)))
                
            datasets = raw_datasets
            
        except Exception as e:
            logger.error(f"Error loading HF dataset: {e}")
            sys.exit(1)
    else:
        logger.error("No dataset specified. Use --data-file or --hf-dataset")
        sys.exit(1)
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"finetune-{model_key}"
        )
    
    # Setup model and tokenizer
    is_llama_model = "llama" in model_name.lower()
    model, tokenizer = setup_model_and_tokenizer(
        model_name=model_name,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit,
        device=device
    )
    
    # Setup LoRA configuration
    # Target modules vary by model architecture
    if "llama" in model_name.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif "phi" in model_name.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "dense"]
    elif "gemma" in model_name.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        # Default modules that commonly work
        target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    
    lora_config = setup_lora_config(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        use_8bit=args.use_8bit,
        use_4bit=args.use_4bit
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Preprocess training and validation datasets
    def preprocess_function(examples):
        return preprocess_data(
            examples=examples,
            tokenizer=tokenizer,
            max_length=args.max_length,
            prompt_template=prompt_template,
            is_llama_model=is_llama_model
        )
    
    # Apply preprocessing to datasets
    tokenized_datasets = {}
    for split in datasets:
        tokenized_datasets[split] = datasets[split].map(
            preprocess_function,
            batched=True,
            remove_columns=datasets[split].column_names
        )
        logger.info(f"Processed {len(tokenized_datasets[split])} examples for {split}")
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir=str(output_dir / "logs"),
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        report_to="wandb" if args.use_wandb else "none",
        fp16=device.startswith("cuda"),  # Use mixed precision on CUDA
        optim="paged_adamw_8bit" if args.use_8bit or args.use_4bit else "adamw_torch",
    )
    
    # Set up data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're using causal language modeling, not masked LM
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"] if "val" in tokenized_datasets else None,
        data_collator=data_collator,
        compute_metrics=compute_metrics if "val" in tokenized_datasets else None
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the fine-tuned model
    logger.info("Saving fine-tuned model...")
    model.save_pretrained(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")
    
    # Also save the adapter separately for easier loading
    model.save_pretrained(output_dir / "adapter_only", adapter_only=True)
    
    # Finish wandb run
    if args.use_wandb:
        wandb.finish()
    
    logger.info(f"Fine-tuning complete! Model saved to {output_dir}")
    
    # Test the fine-tuned model
    if "test" in tokenized_datasets:
        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(tokenized_datasets["test"])
        logger.info(f"Test results: {test_results}")
        
        # Save test results
        with open(output_dir / "test_results.json", "w") as f:
            json.dump(test_results, f, indent=2)


if __name__ == "__main__":
    main() 