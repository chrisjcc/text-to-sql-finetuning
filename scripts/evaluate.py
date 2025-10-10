"""
Script to evaluate a fine-tuned text-to-SQL model using Hydra hierarchical configs.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py evaluation.num_eval_samples=500
"""

import sys
import os
from pathlib import Path
from random import randint
from typing import Dict, Any

import torch
from transformers import pipeline
from datasets import Dataset
from tqdm import tqdm
from dotenv import load_dotenv
import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_preparation import DatasetProcessor
from src.model_setup import ModelSetup
from src.utils import (
    setup_logging,
    validate_file_exists,
    check_gpu_availability,
)


class ModelEvaluator:
    """Handles model evaluation for text-to-SQL tasks."""

    def __init__(self, model_path: Path, test_dataset: Dataset):
        """
        Initialize the evaluator.

        Args:
            model_path: Path to the trained model
            test_dataset: Test dataset for evaluation
        """
        self.model_path = Path(model_path).resolve()
        self.test_dataset = test_dataset
        self.pipe = None

    def load_model(self) -> None:
        """Load the trained model and create a generation pipeline."""
        print(f"Loading trained model from {self.model_path}...")
        model, tokenizer = ModelSetup.load_trained_model(str(self.model_path))

        # Try to create pipeline with device specification
        # If it fails (accelerate loaded model), retry without device
        try:
            device = 0 if torch.cuda.is_available() else -1

            self.pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device=device
            )
            print("Model loaded successfully!")
        except ValueError as e:
            if "accelerate" in str(e).lower():
                # Model was loaded with accelerate, retry without device parameter
                print("Retrying pipeline creation without device parameter (accelerate detected)...")
                self.pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                )
                print("Model loaded successfully!")
            else:
                # Different error, re-raise
                raise

    def generate_sql(
        self,
        messages: list,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> str:
        """Generate SQL query from messages."""
        prompt = self.pipe.tokenizer.apply_chat_template(
            messages[:2],  # System + user messages only
            tokenize=False,
            add_generation_prompt=True,
        )

        outputs = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=self.pipe.tokenizer.eos_token_id,
            pad_token_id=self.pipe.tokenizer.pad_token_id,
        )

        generated_text = outputs[0]["generated_text"][len(prompt) :].strip()
        return generated_text

    def evaluate_sample(self, sample: Dict[str, Any]) -> int:
        """Evaluate a single sample by comparing generated SQL with ground truth."""
        predicted_answer = self.generate_sql(sample["messages"])
        ground_truth = sample["messages"][2]["content"]
        return int(predicted_answer == ground_truth)

    def run_evaluation(self, num_samples: int = 1000, seed: int = 42) -> Dict[str, Any]:
        """Run evaluation on multiple samples."""
        if self.pipe is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        print(f"\nEvaluating on {num_samples} samples...")
        eval_samples = self.test_dataset.shuffle(seed=seed).select(range(num_samples))

        success = [self.evaluate_sample(sample) for sample in tqdm(eval_samples, desc="Evaluating")]

        accuracy = sum(success) / len(success)
        return {
            "accuracy": accuracy,
            "num_samples": len(success),
            "num_correct": sum(success),
            "num_incorrect": len(success) - sum(success),
        }

    def show_examples(self, num_examples: int = 3):
        """Display example predictions."""
        if self.pipe is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        print("\n" + "=" * 80)
        print("Example Predictions")
        print("=" * 80)
        for i in range(num_examples):
            sample = self.test_dataset[randint(0, len(self.test_dataset) - 1)]
            predicted = self.generate_sql(sample["messages"])
            ground_truth = sample["messages"][2]["content"]
            print(f"\n--- Example {i + 1} ---")
            print(f"Question: {sample['messages'][1]['content']}")
            print(f"Ground Truth SQL:\n{ground_truth}")
            print(f"Predicted SQL:\n{predicted}")
            print(f"Match: {'✓' if predicted == ground_truth else '✗'}")
            print("-" * 80)


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main evaluation function using Hydra hierarchical configs."""
    # Load environment variables (for secrets)
    load_dotenv()

    # Log Hydra config
    print("\nConfiguration:\n", OmegaConf.to_yaml(cfg))

    # Setup logging
    setup_logging(Path("logs") / "evaluation.log")

    # Resolve paths relative to Hydra's working directory
    model_path = Path(cfg.evaluation.model_path).resolve()
    test_dataset_path = Path(cfg.evaluation.test_dataset_path).resolve()

    # Ensure files exist
    validate_file_exists(model_path, "Model directory")
    validate_file_exists(test_dataset_path, "Test dataset")

    # Check GPU availability
    check_gpu_availability()

    # Load test dataset
    print("\n" + "=" * 80)
    print("\nLoading test dataset...")
    print("=" * 80)
    processor = DatasetProcessor(cfg.dataset.name)
    test_dataset = processor.load_prepared_dataset(test_dataset_path)
    print(f"Test dataset loaded: {len(test_dataset)} samples")

    # Create evaluator
    evaluator = ModelEvaluator(model_path=model_path, test_dataset=test_dataset)
    evaluator.load_model()
    # Show example predictions
    evaluator.show_examples(num_examples=3)

    # Run full evaluation
    print("\n" + "=" * 80)
    print("Running full evaluation...")
    print("=" * 80)

    # Run evaluation
    results = evaluator.run_evaluation(num_samples=cfg.evaluation.num_eval_samples)

    # Report results
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Correct: {results['num_correct']} / {results['num_samples']}")
    print(f"Incorrect: {results['num_incorrect']} / {results['num_samples']}")
    print("=" * 80)

    print("\nNote: This evaluation uses exact string matching.")
    print("Alternative evaluation methods could include:")
    print("  - Executing queries and comparing results")
    print("  - Semantic similarity of SQL queries")
    print("  - Human evaluation of query correctness")


if __name__ == "__main__":
    main()
