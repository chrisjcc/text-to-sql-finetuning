"""
Script to merge LoRA adapter with base model and upload to Hugging Face Hub.

Usage:
    python scripts/merge_and_upload.py
    python scripts/merge_and_upload.py --skip-merge  # Upload adapter only
    python scripts/merge_and_upload.py --skip-upload  # Merge only
"""

import sys
import os
import gc
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import HfApi, Repository
from trl import setup_chat_format

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import setup_logging, authenticate_huggingface
from config.config import Config


def merge_lora_weights(
    base_model_id: str,
    adapter_path: str,
    output_path: str,
    hf_token: str
) -> None:
    """
    Merge LoRA adapter weights with base model.
    
    Args:
        base_model_id: Base model ID from Hugging Face
        adapter_path: Path to LoRA adapter weights
        output_path: Path to save merged model
        hf_token: Hugging Face token
    """
    print("\n" + "="*80)
    print("Merging LoRA adapter with base model...")
    print("="*80)
    
    # Create quantization config to save memory during merge
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load base model
    print(f"Loading base model: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    
    # Load tokenizer from adapter (it has the special tokens)
    print(f"Loading tokenizer from: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    # Apply chat format setup to match training configuration
    print("Setting up chat format...")
    base_model, tokenizer = setup_chat_format(base_model, tokenizer)
    
    # Load PEFT adapter
    print(f"Loading PEFT adapter from: {adapter_path}")
    peft_model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.bfloat16,
    )
    
    # Merge adapter weights into base model
    print("Merging adapter weights...")
    merged_model = peft_model.merge_and_unload()
    
    # Save merged model
    print(f"Saving merged model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path, safe_serialization=True)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_path)
    
    print(f"✓ Merged model saved to {output_path}")
    
    # Cleanup
    del base_model, peft_model, merged_model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()


def create_model_card(
    repo_name: str,
    base_model: str,
    dataset: str,
    training_config: dict
) -> str:
    """
    Create a model card for Hugging Face Hub.
    
    Args:
        repo_name: Repository name
        base_model: Base model ID
        dataset: Dataset name
        training_config: Training configuration dict
        
    Returns:
        Model card content
    """
    return f"""---
language:
- en
license: apache-2.0
tags:
- text-to-sql
- llama
- fine-tuned
- sql-generation
- code-generation
datasets:
- {dataset}
base_model: {base_model}
---

# {repo_name}

This model is a fine-tuned version of [{base_model}](https://huggingface.co/{base_model}) for text-to-SQL generation.

## Model Description

- **Base Model:** {base_model}
- **Fine-tuning Method:** QLoRA (4-bit quantization with LoRA adapters)
- **Dataset:** {dataset}
- **Task:** Generate SQL queries from natural language questions

## Training Configuration

- **Epochs:** {training_config.get('epochs', 3)}
- **Batch Size:** {training_config.get('batch_size', 1)}
- **Learning Rate:** {training_config.get('learning_rate', '2e-4')}
- **LoRA Rank:** {training_config.get('lora_r', 256)}
- **LoRA Alpha:** {training_config.get('lora_alpha', 128)}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "{repo_name}",
    device_map="auto",
    torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained("{repo_name}")

# Create pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Example usage
schema = "CREATE TABLE customers (id INT, name VARCHAR(100), city VARCHAR(50));"
question = "Show all customers from New York"

messages = [
    {{"role": "system", "content": f"You are a text to SQL translator. SCHEMA:\\n{{schema}}"}},
    {{"role": "user", "content": question}}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
result = pipe(prompt, max_new_tokens=256, do_sample=False)
sql = result[0]['generated_text'][len(prompt):].strip()
print(sql)
```

## Model Card

This model was fine-tuned using production-ready code with modular architecture, comprehensive logging, and error handling.

## Citation

If you use this model, please cite:

```bibtex
@misc{{{repo_name.replace('/', '_').replace('-', '_')},
  author = {{Your Name}},
  title = {{{repo_name}}},
  year = {{2025}},
  publisher = {{Hugging Face}},
  url = {{https://huggingface.co/{repo_name}}}
}}
```

## License

This model is released under the Apache 2.0 license.
"""


def upload_to_hub(
    model_path: str,
    repo_name: str,
    username: str,
    hf_token: str,
    private: bool = False,
    config: Config = None
) -> None:
    """
    Upload model to Hugging Face Hub.
    
    Args:
        model_path: Path to model directory
        repo_name: Repository name
        username: Hugging Face username
        hf_token: Hugging Face token
        private: Whether to create private repo
        config: Configuration object for model card
    """
    print("\n" + "="*80)
    print("Uploading to Hugging Face Hub...")
    print("="*80)
    
    repo_id = f"{username}/{repo_name}"
    api = HfApi()
    
    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, token=hf_token, private=private, exist_ok=True)
        print(f"✓ Repository ready: {repo_id}")
    except Exception as e:
        print(f"Note: {e}")
    
    # Create and save model card
    if config:
        training_config = {
            'epochs': config.training.num_train_epochs,
            'batch_size': config.training.per_device_train_batch_size,
            'learning_rate': config.training.learning_rate,
            'lora_r': config.training.lora_r,
            'lora_alpha': config.training.lora_alpha,
        }
        model_card = create_model_card(
            repo_name=repo_id,
            base_model=config.hf.model_id,
            dataset=config.dataset.dataset_name,
            training_config=training_config
        )
    else:
        model_card = f"# {repo_name}\n\nFine-tuned text-to-SQL model."
    
    readme_path = os.path.join(model_path, "README.md")
    with open(readme_path, "w") as f:
        f.write(model_card)
    
    print(f"✓ Model card created: {readme_path}")
    
    # Upload model
    print(f"Uploading files to {repo_id}...")
    try:
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            token=hf_token,
            commit_message="Upload fine-tuned text-to-SQL model"
        )
        print(f"✓ Model uploaded successfully!")
        print(f"View at: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"✗ Upload failed: {e}")
        raise


def main():
    """Main merge and upload function."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter and upload to Hugging Face"
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip merging, only upload adapter"
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip uploading to Hub"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repository"
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        default="code-llama-3.1-8b-sql-adapter",
        help="Repository name on Hugging Face"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_file=Path("logs/merge_upload.log"))
    
    # Load configuration
    config = Config.load()
    
    # Authenticate
    authenticate_huggingface(config.hf.token)
    
    # Define paths
    adapter_path = config.training.output_dir
    merged_path = os.path.join(adapter_path, "merged")
    
    # Validate adapter exists
    if not os.path.exists(adapter_path):
        print(f"✗ Adapter not found at: {adapter_path}")
        print("Run training first: python scripts/train.py")
        return
    
    # Merge model
    if not args.skip_merge:
        print("\nStep 1: Merging LoRA adapter with base model")
        print("This may take several minutes and requires significant memory...")
        
        try:
            merge_lora_weights(
                base_model_id=config.hf.model_id,
                adapter_path=adapter_path,
                output_path=merged_path,
                hf_token=config.hf.token
            )
        except Exception as e:
            print(f"✗ Merge failed: {e}")
            print("\nNote: Merging requires significant CPU/GPU memory (>30GB)")
            print("If you encounter OOM errors, you can upload just the adapter.")
            return
    else:
        print("\n⊘ Skipping merge (--skip-merge flag)")
    
    # Upload to Hub
    if not args.skip_upload:
        if not config.hf.username:
            print("\n✗ HF_USERNAME not set in .env file")
            print("Add your Hugging Face username to .env:")
            print("HF_USERNAME=your_username")
            return
        
        print("\nStep 2: Uploading to Hugging Face Hub")
        
        # Choose what to upload
        if args.skip_merge or not os.path.exists(merged_path):
            print("Uploading adapter only (lightweight)...")
            upload_path = adapter_path
            repo_suffix = "-adapter"
        else:
            print("Uploading merged model (full model)...")
            upload_path = merged_path
            repo_suffix = "-merged"
        
        repo_name = args.repo_name
        if not repo_name.endswith(repo_suffix):
            repo_name += repo_suffix
        
        try:
            upload_to_hub(
                model_path=upload_path,
                repo_name=repo_name,
                username=config.hf.username,
                hf_token=config.hf.token,
                private=args.private,
                config=config
            )
        except Exception as e:
            print(f"✗ Upload failed: {e}")
            return
    else:
        print("\n⊘ Skipping upload (--skip-upload flag)")
    
    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    if not args.skip_merge:
        print(f"✓ Merged model: {merged_path}")
    if not args.skip_upload:
        repo_id = f"{config.hf.username}/{args.repo_name}"
        print(f"✓ Uploaded to: https://huggingface.co/{repo_id}")
    print("="*80)


if __name__ == "__main__":
    main()
