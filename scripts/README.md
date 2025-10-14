# Text-to-SQL Evaluation: Configuration & Usage Guide

## Table of Contents

- [Configuration Structure](#configuration-structure)
- [Configuration Files](#configuration-files)
- [Usage Examples](#usage-examples)
- [Complete Workflow](#complete-workflow)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Result Interpretation](#result-interpretation)
- [Quick Reference](#quick-reference)
- [Best Practices](#best-practices)
- [Additional Resources](#additional-resources)

---

## Configuration Structure

### Directory Layout

```
config/
├── config.yaml              # Main config (composes all others)
├── hf/
│   └── hf.yaml             # HuggingFace settings
├── dataset/
│   └── dataset.yaml        # Dataset paths and settings
├── wandb/
│   └── wandb.yaml          # Weights & Biases config
├── training/
│   └── training.yaml       # Training hyperparameters
└── evaluation/
    └── evaluation.yaml     # Evaluation settings (clean, no duplication)
```

### Design Principles

1. **No Duplication**: Dataset paths live in `dataset.yaml`, not `evaluation.yaml`
2. **Interpolation**: Use `${dataset.test_dataset_path}` to reference other configs
3. **Single Responsibility**: Each config file handles one concern
4. **Composition**: Main `config.yaml` composes all configs via defaults

---

## Configuration Files

### config/evaluation/evaluation.yaml

```yaml
# ==========================================================================
# Evaluation-specific configuration (clean, no duplication)
# ==========================================================================

# Model configuration
model_path: "meta-llama/Llama-3.1-8B-Instruct"  # Base model (override if needed)
adapter_path: null  # Set to checkpoint path for fine-tuned evaluation

# Examples:
# adapter_path: "${training.output_dir}/checkpoint-500"
# adapter_path: "./outputs/code-llama-3-1-8b-text-to-sql/checkpoint-1000"
# adapter_path: "your-username/text2sql-adapter"  # HuggingFace Hub

# Evaluation dataset (references dataset config via interpolation)
num_eval_samples: 1000  # Number of samples to evaluate (use smaller for quick tests)

# Generation parameters
batch_size: 8              # Adjust based on GPU memory
temperature: 0.0           # 0.0 = greedy (deterministic), >0.0 = sampling
max_new_tokens: 128        # Maximum tokens to generate per query

# Evaluation behavior
skip_baseline: false       # Set to true to skip baseline evaluation (faster)
num_examples: 3           # Number of example predictions to display
save_predictions: true    # Save detailed predictions to JSON
```

### config/dataset/dataset.yaml

```yaml
# ==========================================================================
# Dataset configuration (keep your existing config)
# ==========================================================================

name: b-mc2/sql-create-context
train_samples: 10000
test_samples: 2500
train_dataset_path: data/train_dataset.json
test_dataset_path: data/test_dataset.json
```

### config/training/training.yaml

```yaml
# ==========================================================================
# Training hyperparameters (keep your existing config)
# ==========================================================================

resume_from_checkpoint: false
output_dir: code-llama-3-1-8b-text-to-sql
num_train_epochs: 3
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 2e-4
max_seq_length: 2048
max_grad_norm: 1.0
warmup_ratio: 0.03
logging_steps: 10
lora_alpha: 16
lora_dropout: 0.05
lora_r: 8
```

### config/config.yaml

```yaml
# ==========================================================================
# Main config with defaults (keep your existing structure)
# ==========================================================================

defaults:
  - hf: hf
  - dataset: dataset
  - wandb: wandb
  - training: training
  - evaluation: evaluation
  - _self_  # This ensures config.yaml overrides take precedence

# Optional: Add any global settings here
logging:
  level: INFO
  log_dir: logs
```

---

## Usage Examples

### 1. Evaluate Baseline Model (No Fine-tuning)

```bash
# Uses base model with no adapter
python scripts/evaluate.py \
  evaluation.model_path="meta-llama/Llama-3.1-8B-Instruct" \
  evaluation.adapter_path=null
```

**What this does:**
- Evaluates the base model only
- No fine-tuned comparison
- Good for establishing baseline metrics

**Output Example:**

```
================================================================================
EVALUATION SUMMARY
================================================================================

Configuration:
 • Samples evaluated: 1000
 • Base model: meta-llama/Llama-3.1-8B-Instruct

Metric              Baseline
--------------------------------------------------------------------------------
Accuracy            52.40%
Correct predictions 524 / 1000
Incorrect predictions 476
```

### 2. Full Comparative Evaluation (Baseline + Fine-tuned)

```bash
# Compares base model vs fine-tuned checkpoint
python scripts/evaluate.py \
  evaluation.model_path="meta-llama/Llama-3.1-8B-Instruct" \
  evaluation.adapter_path="./code-llama-3-1-8b-text-to-sql/checkpoint-1000"
```

**What this does:**
- Evaluates baseline (base model alone)
- Evaluates fine-tuned (base model + adapter)
- Shows improvement metrics
- **This is the default recommended approach**

**Output Example:**

```
================================================================================
EVALUATION SUMMARY
================================================================================

Configuration:
 • Samples evaluated: 1000
 • Base model: meta-llama/Llama-3.1-8B-Instruct
 • Adapter: ./code-llama-3-1-8b-text-to-sql/checkpoint-1000

--------------------------------------------------------------------------------
Metric              Baseline    Fine-tuned    Δ
--------------------------------------------------------------------------------
Accuracy            52.40%      78.20%        +25.80%
Correct predictions 524 / 1000  782 / 1000    +258
Incorrect predictions 476        218           -258
--------------------------------------------------------------------------------

💡 Relative Improvement: +49.24%
✅ Fine-tuning improved accuracy by 25.80 percentage points!
```

### 3. Skip Baseline (Faster, When You Already Know Baseline)

```bash
# Only evaluates fine-tuned model
python scripts/evaluate.py \
  evaluation.adapter_path="./code-llama-3-1-8b-text-to-sql/checkpoint-1000" \
  evaluation.skip_baseline=true
```

**What this does:**
- Skips baseline evaluation (saves time)
- Only evaluates fine-tuned model
- Use when you've already established baseline

### 4. Quick Test with Smaller Sample

```bash
# Test with 100 samples instead of 1000
python scripts/evaluate.py \
  evaluation.num_eval_samples=100 \
  evaluation.adapter_path="./code-llama-3-1-8b-text-to-sql/checkpoint-500"
```

**What this does:**
- Faster evaluation for testing
- Good for iterating on evaluation code
- Not for final results

### 5. Evaluate Multiple Checkpoints

```bash
#!/bin/bash
# Script to evaluate all checkpoints

for checkpoint in checkpoint-{500,1000,1500,2000}; do
  echo "Evaluating $checkpoint..."
  python scripts/evaluate.py \
    evaluation.adapter_path="./code-llama-3-1-8b-text-to-sql/$checkpoint" \
    evaluation.skip_baseline=true \
    evaluation.num_eval_samples=500
done
```

**What this does:**
- Compares multiple training checkpoints
- Skips baseline each time (only need once)
- Helps find best checkpoint

### 6. Use Different Base Model

```bash
# Evaluate with a different base model
python scripts/evaluate.py \
  evaluation.model_path="codellama/CodeLlama-7b-Instruct-hf" \
  evaluation.adapter_path=null
```

**What this does:**
- Changes the base model
- Useful for comparing different model families

### 7. Adjust for GPU Memory Constraints

```bash
# Reduce batch size for limited memory
python scripts/evaluate.py \
  evaluation.batch_size=2 \
  evaluation.adapter_path="./code-llama-3-1-8b-text-to-sql/checkpoint-1000"
```

**What this does:**
- Reduces memory usage
- Slower but fits in smaller GPUs

### 8. Use HuggingFace Hub Adapter

```bash
# Evaluate adapter from HuggingFace Hub
python scripts/evaluate.py \
  evaluation.model_path="meta-llama/Llama-3.1-8B-Instruct" \
  evaluation.adapter_path="your-username/llama-text2sql-adapter"
```

**What this does:**
- Downloads adapter from HuggingFace Hub
- Useful for sharing and reproducing results

---

## Complete Workflow

### Step 1: Train Your Model

```bash
python scripts/train.py
```

**Output:** Checkpoints saved to `code-llama-3-1-8b-text-to-sql/checkpoint-{500,1000,1500,...}`

### Step 2: Quick Evaluation (Development)

```bash
# Quick test on checkpoint-500
python scripts/evaluate.py \
  evaluation.num_eval_samples=100 \
  evaluation.adapter_path="./code-llama-3-1-8b-text-to-sql/checkpoint-500"
```

### Step 3: Full Evaluation (Final)

```bash
# Full evaluation on best checkpoint
python scripts/evaluate.py \
  evaluation.adapter_path="./code-llama-3-1-8b-text-to-sql/checkpoint-1000"
```

### Step 4: Analyze Results

```bash
# Results saved to: results/evaluation_results_YYYYMMDD_HHMMSS.json
cat results/evaluation_results_*.json | jq '.improvement'
```

---

## Advanced Features

### Multiple Config Overrides

```bash
# Override multiple values
python scripts/evaluate.py \
  evaluation.model_path="codellama/CodeLlama-13b-Instruct-hf" \
  evaluation.adapter_path="./outputs/checkpoint-2000" \
  evaluation.num_eval_samples=2000 \
  evaluation.batch_size=4 \
  evaluation.temperature=0.1 \
  evaluation.skip_baseline=false \
  dataset.test_dataset_path="data/custom_test.json"
```

### Validate Config Without Running

```bash
# Print resolved config without executing evaluation
python scripts/evaluate.py --cfg job
```

This shows exactly how Hydra resolves your configuration.

### Environment Variables

```bash
# Use environment variables for sensitive values
export HF_TOKEN="your_token_here"
python scripts/evaluate.py
```

---

## Troubleshooting

### Issue: "Model path not found"

**Problem:** Cannot find model or checkpoint

**Solution:**

```bash
# Use absolute path
python scripts/evaluate.py \
  evaluation.adapter_path="$(pwd)/code-llama-3-1-8b-text-to-sql/checkpoint-1000"

# Or verify path exists
ls -la ./code-llama-3-1-8b-text-to-sql/checkpoint-1000/
```

### Issue: "Out of memory"

**Problem:** GPU memory exhausted

**Solution 1: Reduce batch size**

```bash
python scripts/evaluate.py evaluation.batch_size=1
```

**Solution 2: Reduce samples**

```bash
python scripts/evaluate.py evaluation.num_eval_samples=100
```

**Solution 3: Monitor GPU**

```bash
watch -n 1 nvidia-smi
```

### Issue: "Test dataset not found"

**Problem:** Cannot locate test dataset file

**Solution 1: Check dataset config**

```bash
cat config/dataset/dataset.yaml
```

**Solution 2: Override path**

```bash
python scripts/evaluate.py dataset.test_dataset_path="data/my_test.json"
```

**Solution 3: Verify file exists**

```bash
ls -la data/test_dataset.json
```

### Issue: "Evaluation too slow"

**Problem:** Takes too long to evaluate

**Solution 1: Skip baseline**

```bash
python scripts/evaluate.py evaluation.skip_baseline=true
```

**Solution 2: Reduce samples**

```bash
python scripts/evaluate.py evaluation.num_eval_samples=100
```

**Solution 3: Increase batch size (if memory allows)**

```bash
python scripts/evaluate.py evaluation.batch_size=16
```

### Issue: "Results not reproducible"

**Problem:** Different results on repeated runs

**Solution: Use deterministic generation**

```bash
# Ensure temperature is 0.0 (greedy decoding)
python scripts/evaluate.py evaluation.temperature=0.0
```

---

## Result Interpretation

### Output File Structure

**Location:** `results/evaluation_results_YYYYMMDD_HHMMSS.json`

```json
{
  "evaluation_config": {
    "num_samples": 1000,
    "batch_size": 8,
    "temperature": 0.0,
    "base_model": "meta-llama/Llama-3.1-8B-Instruct",
    "adapter_path": "./code-llama-3-1-8b-text-to-sql/checkpoint-1000"
  },
  "baseline": {
    "accuracy": 0.524,
    "num_samples": 1000,
    "num_correct": 524,
    "num_incorrect": 476,
    "sample_predictions": ["SELECT ...", "..."],
    "error_examples": [
      {
        "question": "What are all employee names?",
        "predicted": "SELECT name FROM employee",
        "ground_truth": "SELECT first_name, last_name FROM employees"
      }
    ],
    "timestamp": "2025-01-15T10:30:00"
  },
  "fine_tuned": {
    "accuracy": 0.782,
    "num_samples": 1000,
    "num_correct": 782,
    "num_incorrect": 218,
    "sample_predictions": ["SELECT ...", "..."],
    "error_examples": [...],
    "timestamp": "2025-01-15T10:35:00"
  },
  "improvement": {
    "absolute_improvement": 0.258,
    "relative_improvement_pct": 49.24,
    "correct_gain": 258
  },
  "config": {
    "evaluation": {...},
    "dataset": {...},
    "training": {...}
  }
}
```

### Key Metrics Explained

#### Absolute Improvement

- **Definition:** Percentage point difference between fine-tuned and baseline
- **Example:** 78.2% - 52.4% = 25.8 percentage points
- **Interpretation:** How many more percentage points of accuracy gained

#### Relative Improvement

- **Definition:** Percent increase over baseline
- **Formula:** `(fine_tuned - baseline) / baseline * 100`
- **Example:** `(78.2 - 52.4) / 52.4 * 100 = 49.24%`
- **Interpretation:** Fine-tuned model is 49.24% better than baseline

#### Correct Gain

- **Definition:** Additional number of correct predictions
- **Example:** 782 - 524 = 258 more correct predictions
- **Interpretation:** Fine-tuning fixed 258 queries that baseline got wrong

### Performance Benchmarks

| Dataset | Good | Excellent | State-of-the-art |
|---------|------|-----------|------------------|
| Spider | 70%+ | 80%+ | 85%+ |
| WikiSQL | 85%+ | 90%+ | 95%+ |
| BIRD | 50%+ | 65%+ | 75%+ |
| Custom | Baseline + 10% | Baseline + 20% | Baseline + 30% |

### When Fine-tuning Helps Most

#### Large Improvement (>20% absolute):
- ✅ Fine-tuning is very effective
- ✅ Model successfully adapted to your domain
- ✅ Training data quality is good

#### Moderate Improvement (10-20% absolute):
- ⚠️ Fine-tuning is moderately effective
- ⚠️ Consider more training data or epochs
- ⚠️ May need hyperparameter tuning

#### Small Improvement (<10% absolute):
- ❌ Limited benefit from fine-tuning
- ❌ Base model may already be strong
- ❌ Need better training data or different approach

#### Negative Improvement:
- 🚨 Fine-tuning made model worse
- 🚨 Check for overfitting
- 🚨 Review training data quality
- 🚨 Reduce learning rate or epochs

---

## Quick Reference

### Common Commands

| Goal | Command |
|------|---------|
| Baseline only | `evaluation.adapter_path=null` |
| Full comparison | `evaluation.adapter_path="./path/to/checkpoint"` |
| Skip baseline | `evaluation.skip_baseline=true` |
| Quick test | `evaluation.num_eval_samples=100` |
| Low memory | `evaluation.batch_size=2` |
| Multiple checkpoints | Loop with `skip_baseline=true` |
| Different base model | `evaluation.model_path="model-name"` |
| Custom test data | `dataset.test_dataset_path="path/to/data"` |

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | required | Base model path or HF ID |
| `adapter_path` | str/null | null | PEFT adapter path |
| `num_eval_samples` | int | 1000 | Number of samples |
| `batch_size` | int | 8 | Inference batch size |
| `temperature` | float | 0.0 | Sampling temperature |
| `max_new_tokens` | int | 128 | Max tokens to generate |
| `skip_baseline` | bool | false | Skip baseline eval |
| `num_examples` | int | 3 | Sample predictions to show |
| `save_predictions` | bool | true | Save predictions to file |

---

## Best Practices

### ✅ Do's

1. **Always evaluate baseline first** to establish performance floor
2. **Use full test set** (`num_eval_samples=1000+`) for final evaluation
3. **Use temperature=0.0** for deterministic, reproducible results
4. **Save all results** - they include your config for reproducibility
5. **Compare multiple checkpoints** to find the best one
6. **Monitor GPU memory** with `nvidia-smi`
7. **Document your evaluation settings** in your reports

### ❌ Don'ts

1. **Don't use small samples** for final evaluation (only for testing)
2. **Don't change temperature** between runs if comparing results
3. **Don't evaluate without baseline** unless you already have those metrics
4. **Don't ignore error examples** - they reveal model weaknesses
5. **Don't compare results** from different test sets
6. **Don't skip validation** of your config (`--cfg job`)

---

## Additional Resources

- **Hydra Documentation:** https://hydra.cc/
- **Text-to-SQL Benchmarks:** Spider, WikiSQL, BIRD datasets
- **PEFT Documentation:** https://huggingface.co/docs/peft
- **Evaluation Metrics:** Execution accuracy vs. exact match

---

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Validate your config: `python scripts/evaluate.py --cfg job`
3. Check GPU memory: `nvidia-smi`
4. Review logs: `logs/evaluation.log`
5. Verify file paths: `ls -la data/` and `ls -la outputs/`

---

**Document Version:** 1.0  
**Last Updated:** 2025-10-14  
**Compatible With:** Hydra 1.3+, Python 3.8+
