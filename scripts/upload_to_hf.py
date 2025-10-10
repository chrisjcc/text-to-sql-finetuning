"""
Upload trained model and dataset to Hugging Face Hub.

Usage:
    python scripts/upload_to_hf.py
    python scripts/upload_to_hf.py hf.upload.upload_merged=true
    python scripts/upload_to_hf.py hf.upload.upload_dataset=false
"""

import sys
import os
import gc
from pathlib import Path
from typing import Optional, Dict, Any
import json

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel, AutoPeftModelForCausalLM
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
from trl import setup_chat_format

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import (
    setup_logging,
    authenticate_huggingface,
    check_gpu_availability,
    validate_file_exists,
)


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"{Colors.BOLD}{title}{Colors.END}")
    print("=" * 80)


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"{Colors.GREEN}✓{Colors.END} {message}")


def print_skip(message: str) -> None:
    """Print a skip message."""
    print(f"{Colors.YELLOW}⊘{Colors.END} {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"{Colors.RED}✗{Colors.END} {message}")


def validate_prerequisites(cfg: DictConfig) -> bool:
    """
    Validate that all prerequisites are met for upload.
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        True if all checks pass, False otherwise
    """
    print_section("Validating Prerequisites")
    
    checks_passed = True
    
    # Check HF token
    if not cfg.hf.token:
        print_error("HF_TOKEN not set. Add it to your .env file.")
        checks_passed = False
    else:
        print_success("HF token found")
    
    # Check username
    if not cfg.hf.username:
        print_error("HF username not configured")
        checks_passed = False
    else:
        print_success(f"HF username: {cfg.hf.username}")
    
    # Check output directory exists
    output_dir = Path(cfg.training.output_dir)
    if not output_dir.exists():
        print_error(f"Output directory not found: {output_dir}")
        print("  Run training first: python scripts/train.py")
        checks_passed = False
    else:
        print_success(f"Output directory found: {output_dir}")
    
    # Check for adapter files
    adapter_config = output_dir / "adapter_config.json"
    if not adapter_config.exists():
        print_error("adapter_config.json not found. Is this a trained model?")
        checks_passed = False
    else:
        print_success("Adapter files found")
    
    # Check dataset if upload is enabled
    if cfg.hf.upload.upload_dataset:
        dataset_path = Path(cfg.dataset.train_dataset_path)
        if not dataset_path.exists():
            print_error(f"Dataset not found: {dataset_path}")
            checks_passed = False
        else:
            print_success(f"Dataset found: {dataset_path}")
    
    return checks_passed


def create_model_card(
    repo_id: str,
    base_model: str,
    dataset_name: str,
    training_config: Dict[str, Any],
    is_merged: bool = False,
    author_name: str = "Chris JCC",
    license: str = "apache-2.0",
) -> str:
    """
    Create a comprehensive model card for Hugging Face Hub.
    
    Args:
        repo_id: Full repository ID (username/repo-name)
        base_model: Base model ID
        dataset_name: Dataset name/ID
        training_config: Training configuration dictionary
        is_merged: Whether this is a merged model or adapter
        author_name: Author name for citation
        license: License type
        
    Returns:
        Model card content as string
    """
    model_type = "Merged Model" if is_merged else "LoRA Adapter"
    
    # Format training parameters
    epochs = training_config.get('num_train_epochs', 3)
    batch_size = training_config.get('per_device_train_batch_size', 1)
    grad_accum = training_config.get('gradient_accumulation_steps', 8)
    effective_batch = batch_size * grad_accum
    learning_rate = training_config.get('learning_rate', 2e-4)
    lora_r = training_config.get('lora_r', 8)
    lora_alpha = training_config.get('lora_alpha', 16)
    lora_dropout = training_config.get('lora_dropout', 0.05)
    max_seq_length = training_config.get('max_seq_length', 2048)
    
    usage_code = f"""```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "{repo_id}",
    device_map="auto",
    torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

# Create text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    do_sample=False,
    temperature=None,
    top_p=None,
)

# Example: Generate SQL query
schema = \"\"\"
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    created_at TIMESTAMP
);
\"\"\"

question = "Show me all users who registered in the last 7 days"

messages = [
    {{
        "role": "system",
        "content": f"You are a text to SQL translator. Given a database schema and a natural language query, generate the corresponding SQL query.\\n\\nSCHEMA:\\n{{schema}}"
    }},
    {{"role": "user", "content": question}}
]

# Generate SQL
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt)
sql_query = outputs[0]['generated_text'][len(prompt):].strip()

print("Generated SQL:", sql_query)
```"""

    if not is_merged:
        usage_code = f"""```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "{base_model}",
    device_map="auto",
    torch_dtype="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

# For inference, merge adapter for better performance (optional)
model = model.merge_and_unload()

# Create text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    do_sample=False,
)

# Example usage
schema = \"\"\"
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    created_at TIMESTAMP
);
\"\"\"

question = "Show me all users who registered in the last 7 days"

messages = [
    {{
        "role": "system",
        "content": f"You are a text to SQL translator.\\n\\nSCHEMA:\\n{{schema}}"
    }},
    {{"role": "user", "content": question}}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt)
sql_query = outputs[0]['generated_text'][len(prompt):].strip()

print("Generated SQL:", sql_query)
```"""

    card = f"""---
language:
- en
license: {license}
library_name: {'transformers' if is_merged else 'peft'}
tags:
- text-to-sql
- sql-generation
- code-generation
- llama
- fine-tuned
- lora
- text2sql
- natural-language-to-sql
datasets:
- {dataset_name}
base_model: {base_model}
pipeline_tag: text-generation
---

# {repo_id.split('/')[-1]}

This is a **{model_type}** fine-tuned from [{base_model}](https://huggingface.co/{base_model}) for **text-to-SQL** generation tasks.

## 📋 Model Description

- **Base Model**: [{base_model}](https://huggingface.co/{base_model})
- **Model Type**: {model_type}
- **Fine-tuning Method**: QLoRA (4-bit quantization with LoRA adapters)
- **Training Dataset**: {dataset_name}
- **Task**: Convert natural language questions into SQL queries
- **Language**: English
- **License**: {license}

## 🎯 Intended Use

This model is designed to translate natural language questions into SQL queries for database interaction. It works best when provided with:
1. A database schema (CREATE TABLE statements)
2. A natural language question about the data

## 🚀 Usage

{usage_code}

## ⚙️ Training Configuration

### Model Architecture
- **LoRA Rank (r)**: {lora_r}
- **LoRA Alpha**: {lora_alpha}
- **LoRA Dropout**: {lora_dropout}
- **Target Modules**: All linear layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
- **Max Sequence Length**: {max_seq_length}

### Training Hyperparameters
- **Number of Epochs**: {epochs}
- **Per-Device Batch Size**: {batch_size}
- **Gradient Accumulation Steps**: {grad_accum}
- **Effective Batch Size**: {effective_batch}
- **Learning Rate**: {learning_rate}
- **Learning Rate Scheduler**: Constant
- **Optimizer**: AdamW (torch fused)
- **Weight Decay**: 0
- **Warmup Ratio**: {training_config.get('warmup_ratio', 0.03)}
- **Max Gradient Norm**: {training_config.get('max_grad_norm', 0.3)}
- **Precision**: bfloat16

### Training Infrastructure
- **Hardware**: NVIDIA GPU with bfloat16 support
- **Framework**: Transformers + PEFT + TRL
- **Gradient Checkpointing**: Enabled
- **Flash Attention**: {'Enabled' if training_config.get('use_flash_attention', True) else 'Disabled'}

## 📊 Training Details

The model was fine-tuned using Supervised Fine-Tuning (SFT) with the following approach:

1. **Dataset Format**: Chat template with system/user/assistant roles
2. **System Prompt**: Includes database schema for context
3. **User Prompt**: Natural language question
4. **Assistant Response**: SQL query

### Example Training Sample

```json
{{
  "messages": [
    {{
      "role": "system",
      "content": "You are a text to SQL translator...\\n\\nSCHEMA:\\nCREATE TABLE..."
    }},
    {{
      "role": "user",
      "content": "Show me all customers from New York"
    }},
    {{
      "role": "assistant",
      "content": "SELECT * FROM customers WHERE city = 'New York';"
    }}
  ]
}}
```

## 🎓 Model Performance

The model has been trained to generate syntactically correct SQL queries for various database schemas. Performance may vary based on:
- Complexity of the database schema
- Ambiguity in the natural language question
- Similarity to training data

## ⚠️ Limitations

- **Schema Knowledge**: The model must be provided with the database schema at inference time
- **SQL Dialect**: Primarily trained on standard SQL; may require adjustments for specific database systems (PostgreSQL, MySQL, etc.)
- **Complex Queries**: Performance may degrade on very complex multi-join queries or advanced SQL features
- **Ambiguity**: May struggle with ambiguous natural language questions
- **Context Length**: Limited to {max_seq_length} tokens (including schema + question)

## 🔄 Version History

- **v1.0**: Initial release with {epochs} epochs of training

## 📚 Citation

If you use this model in your research or application, please cite:

```bibtex
@misc{{{repo_id.replace('/', '_').replace('-', '_')},
  author = {{{author_name}}},
  title = {{{repo_id.split('/')[-1]}: Fine-tuned Text-to-SQL Model}},
  year = {{2025}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/{repo_id}}}}}
}}
```

## 📄 License

This model is released under the **{license.upper()}** license. The base model [{base_model}](https://huggingface.co/{base_model}) has its own license terms.

## 🙏 Acknowledgments

- Base model: [{base_model}](https://huggingface.co/{base_model})
- Training framework: Hugging Face Transformers, PEFT, TRL
- Dataset: {dataset_name}

## 🤝 Contact

For questions or feedback, please open an issue on the model repository.

---

**Model Type**: {'Full fine-tuned model' if is_merged else 'LoRA adapter weights'}  
**Training Date**: 2025  
**Model Size**: {'~8B parameters' if '8B' in base_model or '8b' in base_model else '~1B parameters' if '1B' in base_model or '1b' in base_model else 'See base model'}  
"""
    
    return card


def create_dataset_card(
    repo_id: str,
    dataset_path: Path,
    dataset_name: str,
    license: str = "apache-2.0",
) -> str:
    """
    Create a dataset card for Hugging Face Hub.
    
    Args:
        repo_id: Full repository ID (username/repo-name)
        dataset_path: Path to dataset file
        dataset_name: Display name for dataset
        license: License type
        
    Returns:
        Dataset card content as string
    """
    # Load dataset to get statistics
    try:
        dataset = load_dataset("json", data_files=str(dataset_path))
        num_samples = len(dataset['train'])
        
        # Get sample for schema display
        sample = dataset['train'][0]
        
        # Try to extract schema info from first few samples
        schemas_found = set()
        for i in range(min(5, num_samples)):
            messages = dataset['train'][i]['messages']
            for msg in messages:
                if msg['role'] == 'system' and 'SCHEMA' in msg['content']:
                    # Extract just the schema part
                    content = msg['content']
                    if 'SCHEMA:' in content:
                        schema_part = content.split('SCHEMA:')[1].strip()
                        # Take first CREATE TABLE as example
                        if 'CREATE TABLE' in schema_part:
                            schemas_found.add(schema_part.split(';')[0] + ';')
                    break
        
        example_schema = list(schemas_found)[0] if schemas_found else "CREATE TABLE example (id INT, name VARCHAR(100));"
        
    except Exception as e:
        num_samples = "Unknown"
        sample = {"messages": []}
        example_schema = "CREATE TABLE example (id INT, name VARCHAR(100));"
    
    # Determine size category
    if isinstance(num_samples, int):
        if num_samples < 1000:
            size_cat = "n<1K"
        elif num_samples < 10000:
            size_cat = "1K<n<10K"
        elif num_samples < 100000:
            size_cat = "10K<n<100K"
        else:
            size_cat = "n>100K"
    else:
        size_cat = "unknown"
    
    card = f"""---
language:
- en
license: {license}
task_categories:
- text-generation
- text2text-generation
task_ids:
- text-to-sql
tags:
- sql
- code-generation
- text2sql
- database
- natural-language-to-sql
pretty_name: {dataset_name}
size_categories:
- {size_cat}
---

# {dataset_name}

A curated dataset for training text-to-SQL models. This dataset contains natural language questions paired with corresponding SQL queries, formatted for instruction fine-tuning.

## 📊 Dataset Summary

- **Total Samples**: {num_samples}
- **Format**: Chat template (system/user/assistant messages)
- **Task**: Text-to-SQL generation
- **Language**: English
- **License**: {license}

## 📁 Dataset Structure

### Data Format

Each example contains a conversation with three roles:

1. **System**: Provides the database schema and task instructions
2. **User**: Natural language question
3. **Assistant**: Corresponding SQL query

```json
{{
  "messages": [
    {{
      "role": "system",
      "content": "You are a text to SQL translator. Given a database schema and question, generate the SQL query.\\n\\nSCHEMA:\\n{example_schema}"
    }},
    {{
      "role": "user",
      "content": "Show all records from the table"
    }},
    {{
      "role": "assistant",
      "content": "SELECT * FROM example;"
    }}
  ]
}}
```

### Fields

- `messages`: List of message dictionaries
  - `role`: One of "system", "user", or "assistant"
  - `content`: Message content (schema for system, question for user, SQL for assistant)

## 💡 Example Entries

Here are a few examples from the dataset:

### Example 1: Simple SELECT
```json
{{
  "messages": [
    {{
      "role": "system",
      "content": "Schema with customers table..."
    }},
    {{
      "role": "user",
      "content": "List all customer names"
    }},
    {{
      "role": "assistant",
      "content": "SELECT name FROM customers;"
    }}
  ]
}}
```

### Example 2: JOIN Query
```json
{{
  "messages": [
    {{
      "role": "system",
      "content": "Schema with orders and customers tables..."
    }},
    {{
      "role": "user",
      "content": "Show orders with customer names"
    }},
    {{
      "role": "assistant",
      "content": "SELECT o.*, c.name FROM orders o JOIN customers c ON o.customer_id = c.id;"
    }}
  ]
}}
```

### Example 3: Aggregation
```json
{{
  "messages": [
    {{
      "role": "system",
      "content": "Schema with sales table..."
    }},
    {{
      "role": "user",
      "content": "What's the total revenue?"
    }},
    {{
      "role": "assistant",
      "content": "SELECT SUM(amount) as total_revenue FROM sales;"
    }}
  ]
}}
```

## 🎯 Intended Use

This dataset is designed for:
- Fine-tuning large language models for text-to-SQL tasks
- Training semantic parsers
- Evaluating natural language to SQL systems
- Research in natural language interfaces for databases

## 🔧 Usage

### Loading the Dataset

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{repo_id}")

# Access training data
train_data = dataset['train']

# Iterate over examples
for example in train_data:
    messages = example['messages']
    # Process messages...
```

### Using with Transformers

```python
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("your-model")

def format_example(example):
    # Apply chat template
    formatted = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return {{"text": formatted}}

# Format dataset
formatted_dataset = dataset.map(
    format_example,
    remove_columns=dataset.column_names
)
```

## 📚 Data Sources

This dataset was prepared from:
- Spider dataset
- WikiSQL dataset
- Custom curated examples

Data preparation includes:
- Schema formatting and normalization
- Question reformulation for clarity
- SQL query validation and formatting
- Conversion to chat template format

## ⚠️ Limitations

- **SQL Dialect**: Primarily standard SQL; may not cover all database-specific features
- **Schema Complexity**: Varies from simple single-table to complex multi-table schemas
- **Question Variety**: Coverage of natural language variations may be limited
- **Domain Coverage**: May not represent all possible database domains equally

## 📊 Dataset Statistics

- **Total Examples**: {num_samples}
- **Average Question Length**: Varies (typically 5-20 words)
- **SQL Query Complexity**: Ranges from simple SELECT to complex multi-table JOINs
- **Schema Types**: Single-table and multi-table schemas

## 🔄 Data Splits

Currently, this dataset includes:
- **Train**: All {num_samples} examples

For training, users should create their own validation/test splits as needed.

## 📄 License

This dataset is released under the **{license.upper()}** license.

## 🙏 Acknowledgments

Thanks to the creators of:
- Spider dataset
- WikiSQL dataset
- The open-source community for text-to-SQL research

## 📞 Contact

For questions, issues, or contributions, please open an issue on the dataset repository.

---

**Dataset Version**: 1.0  
**Last Updated**: 2025  
**Maintained by**: {repo_id.split('/')[0]}
"""
    
    return card


def upload_adapter_to_hub(
    cfg: DictConfig,
    output_dir: Path,
    repo_name: str,
) -> str:
    """
    Upload LoRA adapter to Hugging Face Hub.
    
    Args:
        cfg: Hydra configuration
        output_dir: Path to model output directory
        repo_name: Repository name (without username)
        
    Returns:
        Repository URL
    """
    print_section(f"Step 1: Uploading LoRA Adapter")
    
    repo_id = f"{cfg.hf.username}/{repo_name}"
    api = HfApi()
    
    # Create repository
    print(f"Creating repository: {repo_id}")
    try:
        api.create_repo(
            repo_id=repo_id,
            token=cfg.hf.token,
            private=False,
            exist_ok=True,
            repo_type="model"
        )
        print_success(f"Repository ready: {repo_id}")
    except Exception as e:
        print(f"Note: {e}")
    
    # Load training configuration
    training_config = {}
    try:
        config_path = output_dir / "adapter_config.json"
        if config_path.exists():
            with open(config_path) as f:
                adapter_config = json.load(f)
                training_config['lora_r'] = adapter_config.get('r', 8)
                training_config['lora_alpha'] = adapter_config.get('lora_alpha', 16)
                training_config['lora_dropout'] = adapter_config.get('lora_dropout', 0.05)
        
        # Add from Hydra config
        training_config['num_train_epochs'] = cfg.training.num_train_epochs
        training_config['per_device_train_batch_size'] = cfg.training.per_device_train_batch_size
        training_config['gradient_accumulation_steps'] = cfg.training.gradient_accumulation_steps
        training_config['learning_rate'] = cfg.training.learning_rate
        training_config['max_seq_length'] = cfg.training.max_seq_length
        training_config['warmup_ratio'] = cfg.training.get('warmup_ratio', 0.03)
        training_config['max_grad_norm'] = cfg.training.get('max_grad_norm', 0.3)
        training_config['use_flash_attention'] = cfg.training.get('use_flash_attention', True)
        
    except Exception as e:
        print(f"Warning: Could not load training config: {e}")
    
    # Create model card
    print("Generating model card...")
    model_card = create_model_card(
        repo_id=repo_id,
        base_model=cfg.hf.model_id,
        dataset_name=f"{cfg.hf.username}/{cfg.hf.upload.dataset_repo_name}",
        training_config=training_config,
        is_merged=False,
        author_name=cfg.hf.upload.author_name,
        license=cfg.hf.upload.license,
    )
    
    # Save model card
    readme_path = output_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(model_card)
    print_success(f"Model card created: {readme_path}")
    
    # Upload files
    print(f"Uploading adapter files to {repo_id}...")
    try:
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=repo_id,
            token=cfg.hf.token,
            commit_message=cfg.hf.upload.commit_message,
            ignore_patterns=["*.pt", "*.bin", "checkpoint-*", "merged", "runs"],  # Skip checkpoints and tensorboard logs
        )
        print_success("Adapter uploaded successfully!")
        
        repo_url = f"https://huggingface.co/{repo_id}"
        print_success(f"View at: {repo_url}")
        return repo_url
        
    except Exception as e:
        print_error(f"Upload failed: {e}")
        raise


def merge_and_upload_model(
    cfg: DictConfig,
    adapter_path: Path,
    repo_name: str,
) -> str:
    """
    Merge LoRA adapter with base model and upload to Hub.
    
    Args:
        cfg: Hydra configuration
        adapter_path: Path to adapter directory
        repo_name: Repository name (without username)
        
    Returns:
        Repository URL
    """
    print_section("Step 2: Merging and Uploading Full Model")
    print("⚠️  This requires significant memory (>30GB RAM/VRAM)")
    
    repo_id = f"{cfg.hf.username}/{repo_name}"
    api = HfApi()
    
    # Create output path for merged model
    merged_path = adapter_path / "merged"
    merged_path.mkdir(exist_ok=True)
    
    try:
        # Create quantization config to save memory
        print("Loading base model with 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.hf.model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            token=cfg.hf.token,
        )
        print_success(f"Base model loaded: {cfg.hf.model_id}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        print_success("Tokenizer loaded from adapter")
        
        # Apply chat format
        print("Setting up chat format...")
        base_model, tokenizer = setup_chat_format(base_model, tokenizer)
        
        # Load PEFT adapter
        print("Loading LoRA adapter...")
        peft_model = PeftModel.from_pretrained(
            base_model,
            str(adapter_path),
            torch_dtype=torch.bfloat16,
        )
        print_success("LoRA adapter loaded")
        
        # Merge adapter weights
        print("Merging adapter weights into base model...")
        merged_model = peft_model.merge_and_unload()
        print_success("Merge completed")
        
        # Save merged model
        print(f"Saving merged model to: {merged_path}")
        merged_model.save_pretrained(merged_path, safe_serialization=True)
        tokenizer.save_pretrained(merged_path)
        print_success("Merged model saved locally")
        
        # Cleanup memory
        del base_model, peft_model, merged_model
        torch.cuda.empty_cache()
        gc.collect()
        
        # Create repository
        print(f"Creating repository: {repo_id}")
        try:
            api.create_repo(
                repo_id=repo_id,
                token=cfg.hf.token,
                private=False,
                exist_ok=True,
                repo_type="model"
            )
            print_success(f"Repository ready: {repo_id}")
        except Exception as e:
            print(f"Note: {e}")
        
        # Load training configuration
        training_config = {}
        try:
            config_path = adapter_path / "adapter_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    adapter_config = json.load(f)
                    training_config['lora_r'] = adapter_config.get('r', 8)
                    training_config['lora_alpha'] = adapter_config.get('lora_alpha', 16)
                    training_config['lora_dropout'] = adapter_config.get('lora_dropout', 0.05)
            
            training_config['num_train_epochs'] = cfg.training.num_train_epochs
            training_config['per_device_train_batch_size'] = cfg.training.per_device_train_batch_size
            training_config['gradient_accumulation_steps'] = cfg.training.gradient_accumulation_steps
            training_config['learning_rate'] = cfg.training.learning_rate
            training_config['max_seq_length'] = cfg.training.max_seq_length
            training_config['warmup_ratio'] = cfg.training.get('warmup_ratio', 0.03)
            training_config['max_grad_norm'] = cfg.training.get('max_grad_norm', 0.3)
            training_config['use_flash_attention'] = cfg.training.get('use_flash_attention', True)
            
        except Exception as e:
            print(f"Warning: Could not load training config: {e}")
        
        # Create model card
        print("Generating model card for merged model...")
        model_card = create_model_card(
            repo_id=repo_id,
            base_model=cfg.hf.model_id,
            dataset_name=f"{cfg.hf.username}/{cfg.hf.upload.dataset_repo_name}",
            training_config=training_config,
            is_merged=True,
            author_name=cfg.hf.upload.author_name,
            license=cfg.hf.upload.license,
        )
        
        # Save model card
        readme_path = merged_path / "README.md"
        with open(readme_path, "w") as f:
            f.write(model_card)
        print_success(f"Model card created: {readme_path}")
        
        # Upload merged model
        print(f"Uploading merged model to {repo_id}...")
        api.upload_folder(
            folder_path=str(merged_path),
            repo_id=repo_id,
            token=cfg.hf.token,
            commit_message="Upload merged text-to-SQL model",
        )
        print_success("Merged model uploaded successfully!")
        
        repo_url = f"https://huggingface.co/{repo_id}"
        print_success(f"View at: {repo_url}")
        return repo_url
        
    except Exception as e:
        print_error(f"Merge/upload failed: {e}")
        print("\nNote: Merging requires significant memory (>30GB)")
        print("Consider uploading just the adapter instead.")
        raise


def upload_dataset_to_hub(
    cfg: DictConfig,
    dataset_path: Path,
    repo_name: str,
) -> str:
    """
    Upload dataset to Hugging Face Hub.
    
    Args:
        cfg: Hydra configuration
        dataset_path: Path to dataset file
        repo_name: Repository name (without username)
        
    Returns:
        Repository URL
    """
    print_section("Step 3: Uploading Dataset")
    
    repo_id = f"{cfg.hf.username}/{repo_name}"
    api = HfApi()
    
    # Create dataset repository
    print(f"Creating dataset repository: {repo_id}")
    try:
        api.create_repo(
            repo_id=repo_id,
            token=cfg.hf.token,
            private=False,
            exist_ok=True,
            repo_type="dataset",
        )
        print_success(f"Dataset repository ready: {repo_id}")
    except Exception as e:
        print(f"Note: {e}")
    
    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    try:
        dataset = load_dataset("json", data_files={"train": str(dataset_path)})
        num_samples = len(dataset['train'])
        print_success(f"Dataset loaded: {num_samples} samples")
    except Exception as e:
        print_error(f"Failed to load dataset: {e}")
        raise
    
    # Create dataset card
    print("Generating dataset card...")
    dataset_card = create_dataset_card(
        repo_id=repo_id,
        dataset_path=dataset_path,
        dataset_name=cfg.dataset.get('dataset_name', 'Text-to-SQL Dataset'),
        license=cfg.hf.upload.license,
    )
    
    # Save dataset card to temp location
    dataset_dir = dataset_path.parent
    readme_path = dataset_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(dataset_card)
    print_success(f"Dataset card created: {readme_path}")
    
    # Push dataset to Hub
    print(f"Uploading dataset to {repo_id}...")
    try:
        dataset.push_to_hub(
            repo_id=repo_id,
            token=cfg.hf.token,
            commit_message="Upload text-to-SQL training dataset",
        )
        print_success(f"Dataset uploaded: {num_samples} samples")
        
        # Upload README separately
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=cfg.hf.token,
            commit_message="Add dataset card",
        )
        print_success("Dataset card uploaded")
        
        repo_url = f"https://huggingface.co/datasets/{repo_id}"
        print_success(f"View at: {repo_url}")
        return repo_url
        
    except Exception as e:
        print_error(f"Dataset upload failed: {e}")
        raise


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """Main upload function."""
    
    # Setup logging
    log_file = Path(get_original_cwd()) / "logs" / "upload.log"
    setup_logging(log_file=log_file)
    
    print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}Uploading to Hugging Face Hub{Colors.END}")
    print(f"{Colors.BOLD}{'='*80}{Colors.END}\n")
    
    # Display configuration
    print("Configuration:")
    print(f"  Username: {cfg.hf.username}")
    print(f"  Base Model: {cfg.hf.model_id}")
    print(f"  Upload Adapter: {cfg.hf.upload.upload_adapter}")
    print(f"  Upload Merged: {cfg.hf.upload.upload_merged}")
    print(f"  Upload Dataset: {cfg.hf.upload.upload_dataset}")
    print()
    
    # Validate prerequisites
    if not validate_prerequisites(cfg):
        print_error("Prerequisites check failed. Please fix the issues above.")
        return
    
    # Authenticate
    authenticate_huggingface(cfg.hf.token)
    
    # Define paths
    output_dir = Path(cfg.training.output_dir).resolve()
    
    # Track results
    results = {
        'adapter': None,
        'merged': None,
        'dataset': None,
    }
    
    # Upload adapter
    if cfg.hf.upload.upload_adapter:
        try:
            adapter_repo_name = cfg.hf.model_id.split('/')[-1] + cfg.hf.upload.adapter_repo_suffix
            results['adapter'] = upload_adapter_to_hub(
                cfg=cfg,
                output_dir=output_dir,
                repo_name=adapter_repo_name,
            )
        except Exception as e:
            print_error(f"Adapter upload failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print_skip("Skipping adapter upload (disabled in config)")
    
    # Upload merged model
    if cfg.hf.upload.upload_merged:
        try:
            merged_repo_name = cfg.hf.model_id.split('/')[-1] + cfg.hf.upload.merged_repo_suffix
            results['merged'] = merge_and_upload_model(
                cfg=cfg,
                adapter_path=output_dir,
                repo_name=merged_repo_name,
            )
        except Exception as e:
            print_error(f"Merged model upload failed: {e}")
            print("Continuing with other uploads...")
            import traceback
            traceback.print_exc()
    else:
        print_skip("Skipping merged model upload (disabled in config)")
    
    # Upload dataset
    if cfg.hf.upload.upload_dataset:
        try:
            dataset_path = Path(cfg.dataset.train_dataset_path).resolve()
            results['dataset'] = upload_dataset_to_hub(
                cfg=cfg,
                dataset_path=dataset_path,
                repo_name=cfg.hf.upload.dataset_repo_name,
            )
        except Exception as e:
            print_error(f"Dataset upload failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print_skip("Skipping dataset upload (disabled in config)")
    
    # Print summary
    print_section("Upload Summary")
    
    if results['adapter']:
        print_success(f"Adapter:  {results['adapter']}")
    else:
        print_skip("Adapter:  Not uploaded")
    
    if results['merged']:
        print_success(f"Merged:   {results['merged']}")
    else:
        print_skip("Merged:   Not uploaded")
    
    if results['dataset']:
        print_success(f"Dataset:  {results['dataset']}")
    else:
        print_skip("Dataset:  Not uploaded")
    
    print(f"\n{Colors.GREEN}✅ Upload process completed!{Colors.END}\n")
    
    # Next steps
    if results['adapter'] or results['merged']:
        print("Next steps:")
        print("1. Test your model: python scripts/inference.py")
        print("2. Share on social media or community forums")
        print("3. Iterate and improve based on feedback")
        print()


if __name__ == "__main__":
    main()
