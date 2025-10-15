"""
Enhanced script to evaluate text-to-SQL models with baseline comparison.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py evaluation.num_eval_samples=500
    python scripts/evaluate.py evaluation.skip_baseline=true  # Skip baseline comparison
"""

import sys
import re
from random import choice
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime

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
from src.model_setup import ModelSetup
from src.utils import (
    setup_logging,
    validate_file_exists,
    check_gpu_availability,
    extract_sql,
)

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
    """Handles model evaluation for text-to-SQL tasks with baseline comparison."""

    def __init__(
        self,
        model_name_or_path: str,
        adapter_path: Optional[str],
        test_dataset: Dataset,
        batch_size: int = 8,
        temperature: float = 0.0,
        skip_baseline: bool = False
    ):
        """
        Initialize evaluator.

        Args:
            model_name_or_path: Base model name or path
            adapter_path: Optional PEFT adapter path or HF ID
            test_dataset: Hugging Face Dataset
            batch_size: Batch size for generation
            temperature: Sampling temperature for generation (0.0 = greedy)
            skip_baseline: If True, skip baseline evaluation
        """
        self.base_model_path = model_name_or_path
        self.adapter_path = adapter_path
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.temperature = temperature
        self.skip_baseline = skip_baseline
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, adapter_path: Optional[str] = None) -> None:
        """
        Load model and tokenizer with optional adapter.

        Args:
            adapter_path: Optional adapter path to load. If None, loads base model only.
        """
        model_desc = "base model" if adapter_path is None else f"fine-tuned model (adapter: {adapter_path})"
        print(f"\n{'='*80}")
        print(f"Loading {model_desc}...")
        print(f"{'='*80}")

        # Unload previous model to free memory
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()

        self.model, self.tokenizer = ModelSetup.load_trained_model(
            model_path=self.base_model_path,
            adapter_path=adapter_path
        )

        # Merge adapter into base model if it's a PEFT model
        if hasattr(self.model, "merge_and_unload"):
            print("🔄 Merging adapter weights into base model...")
            self.model = self.model.merge_and_unload()
            print("✅ Adapter merged successfully")

        # Set model to evaluation mode
        self.model.eval()
        self.model.to(self.device)

        # Ensure padding side and pad token
        # FIXED: Use 'left' padding for generation with decoder-only models
        # Training uses 'right' padding, but generation requires 'left' padding
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"✅ {model_desc.capitalize()} loaded successfully.")

    # ----------------------------------------
    # Generation
    # ----------------------------------------

    def generate_sql_batch(self, prompts: list, max_new_tokens: int = 128) -> list:
        """
        Generate SQL queries for a batch of prompts.

        IMPORTANT FIX: Improved generation config to prevent hallucinations
        and extra text generation after SQL queries.
        """
        self.model.eval()
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=self.temperature > 0.0,
                temperature=self.temperature if self.temperature > 0.0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,  # Prevent repetitive hallucinations
            )

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [extract_sql(sql) for sql in decoded]

    # ----------------------------------------
    # Single Model Evaluation
    # ----------------------------------------

    def _evaluate_single_model(
        self,
        num_samples: int,
        eval_samples: Dataset,
        prompts: list,
        ground_truths: list,
        model_type: str = "model"
    ) -> Dict[str, Any]:
        """
        Evaluate a single model configuration.

        Args:
            num_samples: Number of samples to evaluate
            eval_samples: Dataset samples
            prompts: List of prompts
            ground_truths: List of ground truth SQL queries
            model_type: Description of model type (for logging)

        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        print(f"\nEvaluating {model_type} on {num_samples} samples (batch_size={self.batch_size})...")

        # Process in batches with progress bar
        predictions = []
        for i in tqdm(range(0, len(prompts), self.batch_size), desc="Generating predictions"):
            batch_prompts = prompts[i:i+self.batch_size]
            batch_sql = self.generate_sql_batch(batch_prompts)
            predictions.extend(batch_sql)

        # Compute accuracy with normalized comparison
        matches = [(normalize_sql(p) == normalize_sql(t)) for p, t in zip(predictions, ground_truths)]
        correct = sum(matches)

        # Collect error examples
        error_examples = []
        for idx, (pred, truth, match) in enumerate(zip(predictions, ground_truths, matches)):
            if not match and len(error_examples) < 5:  # Store up to 5 error examples
                error_examples.append({
                    "question": eval_samples[idx]["messages"][-2]["content"],
                    "predicted": pred,
                    "ground_truth": truth
                })

        return {
            "accuracy": correct / len(predictions),
            "num_samples": len(predictions),
            "num_correct": correct,
            "num_incorrect": len(predictions) - correct,
            "sample_predictions": predictions[:10],
            "sample_ground_truths": ground_truths[:10],
            "error_examples": error_examples,
            "timestamp": datetime.now().isoformat(),
        }

    # ----------------------------------------
    # Comparative Evaluation
    # ----------------------------------------

    def run_comparative_evaluation(self, num_samples: int = 1000) -> Dict[str, Any]:
        """
        Run evaluation on both baseline and fine-tuned models (if applicable).

        Args:
            num_samples: Number of samples to evaluate

        Returns:
            Dictionary containing comparative evaluation results
        """
        print(f"\n{'='*80}")
        print("STARTING COMPARATIVE EVALUATION")
        print(f"{'='*80}")

        # Prepare evaluation data once
        print(f"\nPreparing {num_samples} evaluation samples...")
        eval_samples = self.test_dataset.shuffle(seed=42).select(range(num_samples))

        # Prepare prompts and ground truths
        prompts = []
        ground_truths = []
        for ex in eval_samples:
            sys_msg = ex["messages"][0]["content"]
            user_msg = ex["messages"][-2]["content"]
            assistant_msg = ex["messages"][-1]["content"]

            prompt = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": sys_msg},
                 {"role": "user", "content": user_msg}],
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(prompt)
            ground_truths.append(assistant_msg)

        results = {
            "evaluation_config": {
                "num_samples": num_samples,
                "batch_size": self.batch_size,
                "temperature": self.temperature,
                "base_model": self.base_model_path,
                "adapter_path": self.adapter_path,
            }
        }

        # Evaluate baseline model (if not skipped)
        if not self.skip_baseline:
            print(f"\n{'='*80}")
            print("BASELINE EVALUATION (Base Model Without Fine-tuning)")
            print(f"{'='*80}")

            self.load_model(adapter_path=None)
            baseline_results = self._evaluate_single_model(
                num_samples, eval_samples, prompts, ground_truths,
                model_type="baseline model"
            )
            results["baseline"] = baseline_results

            print(f"\n📊 Baseline Accuracy: {baseline_results['accuracy']*100:.2f}%")

        # Evaluate fine-tuned model (if adapter exists)
        if self.adapter_path:
            print(f"\n{'='*80}")
            print("FINE-TUNED EVALUATION (Model With Adapter)")
            print(f"{'='*80}")

            self.load_model(adapter_path=self.adapter_path)
            finetuned_results = self._evaluate_single_model(
                num_samples, eval_samples, prompts, ground_truths,
                model_type="fine-tuned model"
            )
            results["fine_tuned"] = finetuned_results

            print(f"\n📊 Fine-tuned Accuracy: {finetuned_results['accuracy']*100:.2f}%")

            # Calculate improvement metrics (if baseline was evaluated)
            if "baseline" in results:
                baseline_acc = results["baseline"]["accuracy"]
                finetuned_acc = finetuned_results["accuracy"]

                improvement = {
                    "absolute_improvement": finetuned_acc - baseline_acc,
                    "relative_improvement_pct": ((finetuned_acc - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else 0,
                    "correct_gain": finetuned_results["num_correct"] - results["baseline"]["num_correct"],
                }
                results["improvement"] = improvement

        elif not self.skip_baseline:
            # Only baseline was evaluated
            print("\n⚠️  No adapter path provided. Only baseline evaluation was performed.")

        return results

    # ----------------------------------------
    # Examples
    # ----------------------------------------

    def show_examples(self, num_examples: int = 3, model_type: str = "current model"):
        """Show a few example predictions."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        print(f"\n{'='*80}")
        print(f"SAMPLE PREDICTIONS ({model_type})")
        print(f"{'='*80}")

        samples = [choice(self.test_dataset) for _ in range(num_examples)]
        prompts = []
        for sample in samples:
            sys_msg = sample["messages"][0]["content"]
            user_msg = sample["messages"][-2]["content"]
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": sys_msg},
                 {"role": "user", "content": user_msg}],
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(prompt)

        predictions = self.generate_sql_batch(prompts)

        for i, (sample, pred) in enumerate(zip(samples, predictions)):
            ground_truth = sample["messages"][-1]["content"]
            match = normalize_sql(pred) == normalize_sql(ground_truth)

            print(f"\n--- Example {i+1} ---")
            print(f"Question: {sample['messages'][-2]['content']}")
            print(f"\nGround Truth SQL:\n{ground_truth}")
            print(f"\nPredicted SQL:\n{pred}")
            print(f"\nMatch: {'✅ Correct' if match else '❌ Incorrect'}")
            print("-"*80)


# ----------------------------------------
# Result Reporting
# ----------------------------------------

def print_comparative_summary(results: Dict[str, Any]):
    """Print a detailed comparative summary of evaluation results."""
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")

    # Configuration
    config = results["evaluation_config"]
    print(f"\nConfiguration:")
    print(f"  • Samples evaluated: {config['num_samples']}")
    print(f"  • Batch size: {config['batch_size']}")
    print(f"  • Temperature: {config['temperature']}")
    print(f"  • Base model: {config['base_model']}")
    if config['adapter_path']:
        print(f"  • Adapter: {config['adapter_path']}")

    # Results table
    print(f"\n{'-'*80}")
    if "baseline" in results and "fine_tuned" in results:
        baseline = results["baseline"]
        finetuned = results["fine_tuned"]
        improvement = results["improvement"]

        print(f"{'Metric':<30} {'Baseline':<20} {'Fine-tuned':<20} {'Δ':<10}")
        print(f"{'-'*80}")
        print(f"{'Accuracy':<30} {baseline['accuracy']*100:>6.2f}% {'':<13} {finetuned['accuracy']*100:>6.2f}% {'':<13} {improvement['absolute_improvement']*100:>+6.2f}%")
        print(f"{'Correct predictions':<30} {baseline['num_correct']:>6} / {baseline['num_samples']:<6} {finetuned['num_correct']:>6} / {finetuned['num_samples']:<6} {improvement['correct_gain']:>+6}")
        print(f"{'Incorrect predictions':<30} {baseline['num_incorrect']:>6} {'':<13} {finetuned['num_incorrect']:>6} {'':<13} {finetuned['num_incorrect'] - baseline['num_incorrect']:>+6}")
        print(f"{'-'*80}")
        print(f"\n💡 Relative Improvement: {improvement['relative_improvement_pct']:+.2f}%")

        if improvement['absolute_improvement'] > 0:
            print(f"✅ Fine-tuning improved accuracy by {improvement['absolute_improvement']*100:.2f} percentage points!")
        elif improvement['absolute_improvement'] < 0:
            print(f"⚠️  Fine-tuned model performed worse than baseline by {abs(improvement['absolute_improvement'])*100:.2f} percentage points.")
        else:
            print(f"➖ Fine-tuning showed no change in accuracy.")

    elif "baseline" in results:
        baseline = results["baseline"]
        print(f"{'Metric':<30} {'Baseline':<20}")
        print(f"{'-'*80}")
        print(f"{'Accuracy':<30} {baseline['accuracy']*100:>6.2f}%")
        print(f"{'Correct predictions':<30} {baseline['num_correct']:>6} / {baseline['num_samples']}")
        print(f"{'Incorrect predictions':<30} {baseline['num_incorrect']:>6}")

    elif "fine_tuned" in results:
        finetuned = results["fine_tuned"]
        print(f"{'Metric':<30} {'Fine-tuned':<20}")
        print(f"{'-'*80}")
        print(f"{'Accuracy':<30} {finetuned['accuracy']*100:>6.2f}%")
        print(f"{'Correct predictions':<30} {finetuned['num_correct']:>6} / {finetuned['num_samples']}")
        print(f"{'Incorrect predictions':<30} {finetuned['num_incorrect']:>6}")

    print(f"\n{'='*80}")
    print("\nNote: This evaluation uses exact string matching with whitespace normalization.")
    print("Alternative evaluation methods could include:")
    print("  - Execution-based: Running queries and comparing results")
    print("  - Semantic similarity: Comparing SQL query semantics")
    print("  - Human evaluation: Manual assessment of query correctness")


# ----------------------------------------
# Main
# ----------------------------------------

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Hydra-based evaluation entry point."""
    load_dotenv()
    print("\nConfiguration:\n", OmegaConf.to_yaml(cfg))

    # Setup logging
    log_file = Path(get_original_cwd()) / "logs" / "evaluation.log"
    setup_logging(log_file)

    # Resolve paths using Hydra interpolation
    model_path = cfg.evaluation.model_path
    validate_file_exists(model_path, "Model directory")

    # Use dataset config for test path (avoid duplication)
    test_dataset_path = Path(cfg.dataset.test_dataset_path).resolve()
    validate_file_exists(test_dataset_path, "Test dataset")

    # GPU
    check_gpu_availability()

    # Load dataset
    print("\nLoading test dataset...")
    processor = DatasetProcessor(cfg.dataset.name)
    test_dataset = processor.load_prepared_dataset(test_dataset_path)
    print(f"✅ Loaded {len(test_dataset)} samples.")

    # Create evaluator
    evaluator = ModelEvaluator(
        model_name_or_path=model_path,
        adapter_path=cfg.evaluation.adapter_path,
        test_dataset=test_dataset,
        batch_size=cfg.evaluation.batch_size,
        temperature=cfg.evaluation.temperature,
        skip_baseline=cfg.evaluation.skip_baseline,
    )

    # Show examples (load adapter if available, otherwise base model)
    adapter_for_examples = cfg.evaluation.adapter_path
    evaluator.load_model(adapter_path=adapter_for_examples)
    model_desc = "fine-tuned model" if adapter_for_examples else "base model"
    evaluator.show_examples(num_examples=cfg.evaluation.num_examples, model_type=model_desc)

    # Run comparative evaluation
    print("\n" + "="*80)
    print("RUNNING FULL COMPARATIVE EVALUATION")
    print("="*80)
    results = evaluator.run_comparative_evaluation(
        num_samples=cfg.evaluation.num_eval_samples
    )

    # Save results
    results_dir = Path(get_original_cwd()) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"evaluation_results_{timestamp}.json"

    # Add config to results for reproducibility
    results["config"] = OmegaConf.to_container(cfg, resolve=True)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Detailed results saved to: {results_path}")

    # Print comparative summary
    print_comparative_summary(results)

    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
