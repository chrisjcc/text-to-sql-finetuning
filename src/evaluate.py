"""
Script to evaluate a fine-tuned text-to-SQL model.

Usage:
    python scripts/evaluate.py
"""

import sys
from pathlib import Path
from random import randint
from typing import Dict, Any

import torch
from transformers import pipeline
from datasets import Dataset
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_preparation import DatasetProcessor
from src.model_setup import ModelSetup
from src.utils import (
    setup_logging,
    validate_file_exists,
    check_gpu_availability,
)
from config.config import Config


class ModelEvaluator:
    """Handles model evaluation for text-to-SQL tasks."""
    
    def __init__(self, model_path: str, test_dataset: Dataset):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the trained model
            test_dataset: Test dataset for evaluation
        """
        self.model_path = model_path
        self.test_dataset = test_dataset
        self.pipe = None
        
    def load_model(self) -> None:
        """Load the trained model and create a pipeline."""
        print("Loading trained model...")
        model, tokenizer = ModelSetup.load_trained_model(self.model_path)
        
        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )
        print("Model loaded successfully!")
    
    def generate_sql(
        self,
        messages: list,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> str:
        """
        Generate SQL query from messages.
        
        Args:
            messages: Conversation messages
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            Generated SQL query
        """
        prompt = self.pipe.tokenizer.apply_chat_template(
            messages[:2],  # System + user messages only
            tokenize=False,
            add_generation_prompt=True
        )
        
        outputs = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=self.pipe.tokenizer.eos_token_id,
            pad_token_id=self.pipe.tokenizer.pad_token_id
        )
        
        generated_text = outputs[0]['generated_text'][len(prompt):].strip()
        return generated_text
    
    def evaluate_sample(self, sample: Dict[str, Any]) -> int:
        """
        Evaluate a single sample.
        
        Args:
            sample: Dataset sample with messages
            
        Returns:
            1 if prediction matches ground truth, 0 otherwise
        """
        predicted_answer = self.generate_sql(sample["messages"])
        ground_truth = sample["messages"][2]["content"]
        
        return 1 if predicted_answer == ground_truth else 0
    
    def run_evaluation(
        self,
        num_samples: int = 1000,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Run evaluation on multiple samples.
        
        Args:
            num_samples: Number of samples to evaluate
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with evaluation results
        """
        if self.pipe is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print(f"\nEvaluating on {num_samples} samples...")
        
        # Select samples
        eval_samples = self.test_dataset.shuffle(seed=seed).select(range(num_samples))
        
        # Evaluate
        success_rate = []
        for sample in tqdm(eval_samples, desc="Evaluating"):
            success_rate.append(self.evaluate_sample(sample))
        
        # Calculate metrics
        accuracy = sum(success_rate) / len(success_rate)
        
        results = {
            "accuracy": accuracy,
            "num_samples": num_samples,
            "num_correct": sum(success_rate),
            "num_incorrect": len(success_rate) - sum(success_rate),
        }
        
        return results
    
    def show_example_predictions(self, num_examples: int = 3) -> None:
        """
        Show example predictions from the model.
        
        Args:
            num_examples: Number of examples to show
        """
        if self.pipe is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print("\n" + "="*80)
        print("Example Predictions")
        print("="*80)
        
        for i in range(num_examples):
            rand_idx = randint(0, len(self.test_dataset) - 1)
            sample = self.test_dataset[rand_idx]
            
            predicted = self.generate_sql(sample["messages"])
            ground_truth = sample["messages"][2]["content"]
            
            print(f"\n--- Example {i+1} ---")
            print(f"Question: {sample['messages'][1]['content']}")
            print(f"\nGround Truth SQL:\n{ground_truth}")
            print(f"\nPredicted SQL:\n{predicted}")
            print(f"\nMatch: {'✓' if predicted == ground_truth else '✗'}")
            print("-" * 80)


def main():
    """Main evaluation function."""
    # Setup logging
    setup_logging(log_file=Path("logs/evaluation.log"))
    
    # Load configuration
    config = Config.load()
    
    # Check GPU availability
    check_gpu_availability()
    
    # Validate test dataset exists
    validate_file_exists(config.evaluation.test_dataset_path, "Test dataset")
    
    # Load test dataset
    print("\n" + "="*80)
    print("Loading test dataset...")
    print("="*80)
    processor = DatasetProcessor(config.dataset.dataset_name)
    test_dataset = processor.load_prepared_dataset(config.evaluation.test_dataset_path)
    print(f"Test dataset loaded: {len(test_dataset)} samples")
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=config.evaluation.model_path,
        test_dataset=test_dataset
    )
    
    # Load model
    print("\n" + "="*80)
    evaluator.load_model()
    print("="*80)
    
    # Show example predictions
    evaluator.show_example_predictions(num_examples=3)
    
    # Run full evaluation
    print("\n" + "="*80)
    print("Running full evaluation...")
    print("="*80)
    
    results = evaluator.run_evaluation(
        num_samples=config.evaluation.num_eval_samples
    )
    
    # Print results
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)
    print(f"Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Correct predictions: {results['num_correct']}/{results['num_samples']}")
    print(f"Incorrect predictions: {results['num_incorrect']}/{results['num_samples']}")
    print("="*80)
    
    print("\nNote: This evaluation uses exact string matching.")
    print("Alternative evaluation methods could include:")
    print("  - Executing queries and comparing results")
    print("  - Semantic similarity of SQL queries")
    print("  - Human evaluation of query correctness")


if __name__ == "__main__":
    main()
