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
from transformers.pipelines.pt_utils import KeyDataset
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

    def __init__(self, model_path: Path, test_dataset: Dataset, batch_size: int = 8):
        """
        Initialize the evaluator.

        Args:
            model_path: Path to the trained model
            test_dataset: Test dataset for evaluation
            batch_size: Batch size for inference
        """
        self.model_path = Path(model_path).resolve()
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.pipe = None

    def load_model(self) -> None:
        """Load the trained model and create a generation pipeline."""
        print(f"Loading trained model from {self.model_path}...")
        model, tokenizer = ModelSetup.load_trained_model(str(self.model_path))

        # CRITICAL: Set padding side to 'left' for decoder-only models
        tokenizer.padding_side = 'left'

        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Try to create pipeline with device specification
        # If it fails (accelerate loaded model), retry without device
        try:
            device = 0 if torch.cuda.is_available() else -1

            self.pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device=device,
                batch_size=self.batch_size,  # Enable batching
            )
            print(f"Model loaded successfully with batch_size={self.batch_size}!")
        except ValueError as e:
            if "accelerate" in str(e).lower():
                # Model was loaded with accelerate, retry without device parameter
                print("Retrying pipeline creation without device parameter (accelerate detected)...")
                self.pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    batch_size=self.batch_size,  # Enable batching
                )
                print(f"Model loaded successfully with batch_size={self.batch_size}!")
            else:
                # Different error, re-raise
                raise

    def generate_sql_batch(
        self,
        prompts: list,
        max_new_tokens: int = 256,
        temperature: float = 0.1,  # Lower temperature for more focused generation
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> list:
        """Generate SQL queries for a batch of prompts."""
        outputs = self.pipe(
            prompts,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Use greedy decoding for SQL (deterministic)
            # temperature=temperature,  # Not used with do_sample=False
            # top_k=top_k,  # Not used with do_sample=False
            # top_p=top_p,  # Not used with do_sample=False
            eos_token_id=self.pipe.tokenizer.eos_token_id,
            pad_token_id=self.pipe.tokenizer.pad_token_id,
            batch_size=self.batch_size,  # Process in batches
            return_full_text=False,  # Only return generated text, not the prompt
        )

        # Extract generated text (remove prompt)
        # Pipeline returns: [[{"generated_text": "..."}], [{"generated_text": "..."}], ...]
        # For each prompt, we get a list with one dict (since num_return_sequences=1 by default)
        generated_sqls = []
        for i, output in enumerate(outputs):
            # output is a list of dicts, take the first one
            generated_text = output[0]["generated_text"].strip()

            # Extract just the SQL query (stop at first complete SQL statement)
            # Find the first semicolon and take everything before it
            if ';' in generated_text:
                sql = generated_text.split(';')[0] + ';'
            else:
                # If no semicolon, take first line only
                sql = generated_text.split('\n')[0].strip()

            generated_sqls.append(sql)

        return generated_sqls

    def run_evaluation(self, num_samples: int = 1000, seed: int = 42) -> Dict[str, Any]:
        """Run evaluation on multiple samples using efficient dataset batching."""
        if self.pipe is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        print(f"\nEvaluating on {num_samples} samples with batch_size={self.batch_size}...")
        eval_samples = self.test_dataset.shuffle(seed=seed).select(range(num_samples))

        # Prepare dataset with prompts
        print("Preparing dataset...")
        def prepare_prompt(example):
            prompt = self.pipe.tokenizer.apply_chat_template(
                example["messages"][:2],  # System + user only
                tokenize=False,
                add_generation_prompt=True,
            )
            return {
                "prompt": prompt,
                "ground_truth": example["messages"][2]["content"]
            }

        eval_dataset = eval_samples.map(prepare_prompt)

        # Use pipeline with dataset for true batching
        print(f"Generating predictions with dataset batching (batch_size={self.batch_size})...")

        predictions = []
        ground_truths = []

        # Process using KeyDataset for efficient batching
        for i, output in enumerate(tqdm(
            self.pipe(
                KeyDataset(eval_dataset, "prompt"),
                max_new_tokens=256,
                do_sample=False,  # Use greedy decoding for SQL (deterministic)
                # temperature=0.7,  # Not used with do_sample=False
                # top_k=50,  # Not used with do_sample=False
                # top_p=0.95,  # Not used with do_sample=False
                batch_size=self.batch_size,
                return_full_text=False,
            ),
            total=len(eval_dataset),
            desc="Evaluating"
        )):

            generated = output[0]["generated_text"].strip()
            
            # Extract just SQL (stop at semicolon or first newline)
            if ';' in generated:
                sql = generated.split(';')[0] + ';'
            else:
                sql = generated.split('\n')[0].strip()
            
            predictions.append(sql)
            ground_truths.append(eval_dataset[i]["ground_truth"])

        # Calculate accuracy
        print("Computing accuracy...")
        correct = sum(1 for pred, truth in zip(predictions, ground_truths) if pred.strip() == truth.strip())

        return {
            "accuracy": correct / len(predictions),
            "num_samples": len(predictions),
            "num_correct": correct,
            "num_incorrect": len(predictions) - correct,
            "predictions": predictions[:10],  # Save first 10 for inspection
            "ground_truths": ground_truths[:10],
        }

    def show_examples(self, num_examples: int = 3):
        """Display example predictions."""
        if self.pipe is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        print("\n" + "=" * 80)
        print("Example Predictions")
        print("=" * 80)

        # Process examples in batch for efficiency
        samples = [self.test_dataset[randint(0, len(self.test_dataset) - 1)]
                   for _ in range(num_examples)]

        prompts = []
        for sample in samples:
            prompt = self.pipe.tokenizer.apply_chat_template(
                sample["messages"][:2],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)

        # Generate all predictions at once
        predictions = self.generate_sql_batch(prompts)

        # Display results
        for i, (sample, predicted) in enumerate(zip(samples, predictions)):
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

    # Create evaluator with batch size
    batch_size = cfg.evaluation.get('batch_size', 8)  # Default to 8
    evaluator = ModelEvaluator(
        model_path=model_path,
        test_dataset=test_dataset,
        batch_size=batch_size
    )
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
