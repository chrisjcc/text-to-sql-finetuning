"""
Script to evaluate a fine-tuned text-to-SQL model using Hydra hierarchical configs.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py evaluation.num_eval_samples=500
"""

import sys
import re
from random import choice
from pathlib import Path
from typing import Dict, Any
import json

import torch
from datasets import Dataset
from tqdm import tqdm
from dotenv import load_dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_preparation import DatasetProcessor
from src.utils import setup_logging, validate_file_exists, check_gpu_availability, extract_sql, load_model_and_tokenizer


# ----------------------------------------
# Utility
# ----------------------------------------

def normalize_sql(sql: str) -> str:
    """Normalize SQL for fair comparison."""
    return re.sub(r'\s+', ' ', sql.strip().rstrip(';').lower())


# ----------------------------------------
# ModelEvaluator
# ----------------------------------------

class ModelEvaluator:
    """Handles model evaluation for text-to-SQL tasks."""

    def __init__(self, model_name_or_path: str, adapter_path: str | None, test_dataset: Dataset, batch_size: int = 8, temperature: float = 0.0):
        """
        Initialize evaluator.

        Args:
            model_name_or_path: Base model name or path
            adapter_path: Optional PEFT adapter path or HF ID
            test_dataset: Hugging Face Dataset
            batch_size: Batch size for generation
            temperature: Sampling temperature for generation (0.0 = greedy)
        """
        self.model_name_or_path = model_name_or_path
        self.adapter_path = adapter_path
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.temperature = temperature
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self) -> None:
        """Load model and tokenizer with optional adapter."""
        print(f"Loading model (base: {self.model_name_or_path}, adapter: {self.adapter_path})...")
        self.model, self.tokenizer = load_model_and_tokenizer(self.model_name_or_path, self.adapter_path)
        self.model.to(self.device)
        # Ensure padding side and pad token
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("✅ Model and tokenizer loaded.")

    # ----------------------------------------
    # Generation
    # ----------------------------------------

    def generate_sql_batch(self, prompts: list, max_new_tokens: int = 128) -> list:
        """Generate SQL queries for a batch of prompts."""
        self.model.eval()
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=self.temperature > 0.0,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [extract_sql(sql) for sql in decoded]

    # ----------------------------------------
    # Evaluation
    # ----------------------------------------

    def run_evaluation(self, num_samples: int = 1000) -> Dict[str, Any]:
        """Evaluate the model on a subset of the dataset."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        print(f"\nEvaluating on {num_samples} samples with batch_size={self.batch_size}...")
        eval_samples = self.test_dataset.shuffle(seed=42).select(range(num_samples))

        # Prepare prompts and ground truths safely
        prompts = []
        ground_truths = []
        for ex in eval_samples:
            # Use system + last user message for safety
            sys_msg = ex["messages"][0]["content"]
            user_msg = ex["messages"][-2]["content"]
            assistant_msg = ex["messages"][-1]["content"]
            prompt = self.tokenizer.apply_chat_template([{"role":"system","content":sys_msg},{"role":"user","content":user_msg}], tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
            ground_truths.append(assistant_msg)

        # Process in batches
        predictions = []
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i:i+self.batch_size]
            batch_sql = self.generate_sql_batch(batch_prompts)
            predictions.extend(batch_sql)

        # Compute accuracy with normalized comparison
        correct = sum(normalize_sql(p) == normalize_sql(t) for p, t in zip(predictions, ground_truths))

        return {
            "accuracy": correct / len(predictions),
            "num_samples": len(predictions),
            "num_correct": correct,
            "num_incorrect": len(predictions) - correct,
            "predictions": predictions[:10],
            "ground_truths": ground_truths[:10],
        }

    # ----------------------------------------
    # Examples
    # ----------------------------------------

    def show_examples(self, num_examples: int = 3):
        """Show a few example predictions."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        samples = [choice(self.test_dataset) for _ in range(num_examples)]
        prompts = []
        for sample in samples:
            sys_msg = sample["messages"][0]["content"]
            user_msg = sample["messages"][-2]["content"]
            prompt = self.tokenizer.apply_chat_template([{"role":"system","content":sys_msg},{"role":"user","content":user_msg}], tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        predictions = self.generate_sql_batch(prompts)
        for i, (sample, pred) in enumerate(zip(samples, predictions)):
            ground_truth = sample["messages"][-1]["content"]
            print(f"\n--- Example {i+1} ---")
            print(f"Question: {sample['messages'][-2]['content']}")
            print(f"Ground Truth SQL:\n{ground_truth}")
            print(f"Predicted SQL:\n{pred}")
            print(f"Match: {'✓' if normalize_sql(pred)==normalize_sql(ground_truth) else '✗'}")
            print("-"*80)


# ----------------------------------------
# Main
# ----------------------------------------

@hydra.main(config_path="../config", config_name="evaluation", version_base=None)
def main(cfg: DictConfig):
    """Hydra-based evaluation entry point."""
    load_dotenv()
    print("\nConfiguration:\n", OmegaConf.to_yaml(cfg))

    # Setup logging
    log_file = Path(get_original_cwd()) / "logs" / "evaluation.log"
    setup_logging(log_file)

    # Resolve paths
    model_path = Path(cfg.evaluation.model_path).resolve()
    test_dataset_path = Path(cfg.evaluation.test_dataset_path).resolve()
    validate_file_exists(model_path, "Model directory")
    validate_file_exists(test_dataset_path, "Test dataset")

    # GPU
    check_gpu_availability()

    # Load dataset
    print("\nLoading test dataset...")
    processor = DatasetProcessor(cfg.dataset.name)
    test_dataset = processor.load_prepared_dataset(test_dataset_path)
    print(f"Loaded {len(test_dataset)} samples.")

    # Create evaluator
    evaluator = ModelEvaluator(
        model_name_or_path=cfg.evaluation.model_path,
        adapter_path=getattr(cfg.evaluation, "adapter_path", None),
        test_dataset=test_dataset,
        batch_size=cfg.evaluation.get("batch_size", 8),
        temperature=cfg.evaluation.get("temperature", 0.0),
    )
    evaluator.load_model()
    evaluator.show_examples(num_examples=3)

    # Run evaluation
    print("\nRunning full evaluation...")
    results = evaluator.run_evaluation(num_samples=cfg.evaluation.num_eval_samples)

    # Save results
    results_path = Path(get_original_cwd()) / "results" / "evaluation_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Evaluation results saved to {results_path}")

    # Report
    print("\nEvaluation Summary")
    print("="*80)
    print(f"Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Correct: {results['num_correct']} / {results['num_samples']}")
    print(f"Incorrect: {results['num_incorrect']} / {results['num_samples']}")
    print("="*80)
    print("\nNote: This evaluation uses exact string matching with whitespace normalization.")
    print("Alternative evaluation methods could include:")
    print("  - Executing-based queries and comparing results")
    print("  - Semantic similarity of SQL queries")
    print("  - Human evaluation of query correctness")


if __name__ == "__main__":
    main()
