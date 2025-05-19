import os
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset, load_dataset
from typing import Dict, List, Union, Tuple, Optional

def get_best_device(device_param=-1):
    """
    Get the best available device for inference.
    
    Args:
        device_param: User-specified device (-1 for auto-detect, 0+ for specific GPU/device)
        
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

class RewardModelEvaluator:
    def __init__(
        self, 
        model_path: str,
        device: Union[str, int] = None,
        max_length: int = 512
    ):
        """Initialize the reward model evaluator.
        
        Args:
            model_path (str): Path to the reward model.
            device (str or int, optional): Device to run the model on. 
                If int: -1 for auto-detect, 0+ for specific GPU
                If str: Explicit device specification (e.g., 'cuda:0', 'cpu', 'mps')
                Defaults to auto-detection.
            max_length (int, optional): Maximum sequence length for tokenization. Defaults to 512.
        """
        # Setup device
        if device is None or (isinstance(device, int) and device < 0):
            # Auto-detect device
            self.device_str, _ = get_best_device()
        elif isinstance(device, int):
            # Use specified GPU index
            self.device_str, _ = get_best_device(device)
        else:
            # Use explicitly provided device string
            self.device_str = device
            print(f"Using specified device: {device}")
        
        self.model_path = model_path
        self.max_length = max_length
        
        print(f"Loading model from {model_path}")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device_str)
        self.model.eval()
    
    def prepare_inputs(self, text: str, summary: str) -> Dict[str, torch.Tensor]:
        """Prepare inputs for the reward model.
        
        Args:
            text (str): The source text.
            summary (str): The generated summary.
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary with tokenized inputs.
        """
        # Format: [text] + [SEP] + [summary]
        inputs = self.tokenizer(
            text, 
            summary,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True
        )
        
        return {k: v.to(self.device_str) for k, v in inputs.items()}
    
    @torch.no_grad()
    def score(self, text: str, summary: str) -> float:
        """Get a reward score for a single text-summary pair.
        
        Args:
            text (str): The source text.
            summary (str): The generated summary.
            
        Returns:
            float: The reward score (higher is better).
        """
        inputs = self.prepare_inputs(text, summary)
        outputs = self.model(**inputs)
        
        # For regression models, use the predicted value directly
        if self.model.config.num_labels == 1:
            return outputs.logits.item()
        
        # For classification models, use the probability of the positive class
        else:
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # Assume last class is the positive class
            return probs[0][-1].item()
    
    @torch.no_grad()
    def batch_score(self, texts: List[str], summaries: List[str]) -> List[float]:
        """Get reward scores for a batch of text-summary pairs.
        
        Args:
            texts (List[str]): The source texts.
            summaries (List[str]): The generated summaries.
            
        Returns:
            List[float]: The reward scores (higher is better).
        """
        scores = []
        for text, summary in zip(texts, summaries):
            scores.append(self.score(text, summary))
        return scores
    
    def evaluate_dataset(
        self, 
        dataset: Union[Dataset, pd.DataFrame], 
        text_column: str = "text", 
        summary_column: str = "summary", 
        batch_size: int = 16
    ) -> Dict[str, float]:
        """Evaluate a dataset of text-summary pairs.
        
        Args:
            dataset (Union[Dataset, pd.DataFrame]): Dataset containing text and summaries.
            text_column (str, optional): Column name for source texts. Defaults to "text".
            summary_column (str, optional): Column name for summaries. Defaults to "summary".
            batch_size (int, optional): Batch size for evaluation. Defaults to 16.
            
        Returns:
            Dict[str, float]: Dictionary with evaluation results.
        """
        if isinstance(dataset, pd.DataFrame):
            texts = dataset[text_column].tolist()
            summaries = dataset[summary_column].tolist()
        else:
            texts = dataset[text_column]
            summaries = dataset[summary_column]
        
        all_scores = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating"):
            batch_texts = texts[i:i+batch_size]
            batch_summaries = summaries[i:i+batch_size]
            batch_scores = self.batch_score(batch_texts, batch_summaries)
            all_scores.extend(batch_scores)
        
        results = {
            "mean_score": np.mean(all_scores),
            "std_score": np.std(all_scores),
            "min_score": np.min(all_scores),
            "max_score": np.max(all_scores),
            "median_score": np.median(all_scores),
            "scores": all_scores
        }
        
        return results

def load_huggingface_dataset(
    dataset_name: str,
    split: str = "test",
    subset: Optional[str] = None,
    text_column: str = "text",
    summary_column: str = "summary"
) -> Dataset:
    """Load a dataset from Hugging Face datasets.
    
    Args:
        dataset_name (str): The name of the dataset on Hugging Face Hub.
        split (str, optional): The split to use. Defaults to "test".
        subset (str, optional): The subset/config to use. Defaults to None.
        text_column (str, optional): The column containing source texts. Defaults to "text".
        summary_column (str, optional): The column containing summaries. Defaults to "summary".
        
    Returns:
        Dataset: The loaded dataset.
    """
    print(f"Loading Hugging Face dataset: {dataset_name}" + (f", subset: {subset}" if subset else ""))
    
    # Load the dataset
    if subset:
        dataset = load_dataset(dataset_name, subset, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    
    # Verify columns exist
    if text_column not in dataset.column_names:
        raise ValueError(f"Text column '{text_column}' not found in dataset. Available columns: {dataset.column_names}")
    
    if summary_column not in dataset.column_names:
        raise ValueError(f"Summary column '{summary_column}' not found in dataset. Available columns: {dataset.column_names}")
    
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Evaluate summaries using a reward model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the reward model")
    
    # Dataset loading options
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("--dataset_path", type=str, help="Path to the dataset file (.csv, .json, .jsonl)")
    dataset_group.add_argument("--hf_dataset", type=str, help="Name of a Hugging Face dataset")
    
    # Hugging Face dataset options
    parser.add_argument("--hf_subset", type=str, help="Subset/config of the Hugging Face dataset")
    parser.add_argument("--hf_split", type=str, default="test", help="Split to use for the Hugging Face dataset")
    
    # Common options
    parser.add_argument("--text_column", type=str, default="text", help="Column name for source texts")
    parser.add_argument("--summary_column", type=str, default="summary", help="Column name for summaries")
    parser.add_argument("--output_path", type=str, default="reward_model_scores.json", help="Path to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--device", type=int, default=-1, help="Device to use (-1 for auto-detect, 0+ for specific GPU)")
    
    args = parser.parse_args()
    
    # Load the dataset
    if args.hf_dataset:
        # Load from Hugging Face
        dataset = load_huggingface_dataset(
            dataset_name=args.hf_dataset,
            split=args.hf_split,
            subset=args.hf_subset,
            text_column=args.text_column,
            summary_column=args.summary_column
        )
    else:
        # Load from local file
        if args.dataset_path.endswith('.csv'):
            dataset = pd.read_csv(args.dataset_path)
        elif args.dataset_path.endswith('.json'):
            dataset = pd.read_json(args.dataset_path)
        elif args.dataset_path.endswith('.jsonl'):
            dataset = pd.read_json(args.dataset_path, lines=True)
        else:
            raise ValueError(f"Unsupported file format for {args.dataset_path}. Use .csv, .json, or .jsonl")
    
    # Initialize reward model evaluator
    evaluator = RewardModelEvaluator(
        model_path=args.model_path,
        device=args.device,
        max_length=args.max_length
    )
    
    # Evaluate the dataset
    results = evaluator.evaluate_dataset(
        dataset=dataset,
        text_column=args.text_column,
        summary_column=args.summary_column,
        batch_size=args.batch_size
    )
    
    # Save results
    import json
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output_path}")
    print(f"Average reward score: {results['mean_score']:.4f}")

if __name__ == "__main__":
    main()
