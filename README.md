# Text-to-SQL Fine-tuning

Production-ready implementation for fine-tuning language models to generate SQL queries from natural language instructions. This project uses QLoRA for efficient fine-tuning on 24GB GPUs.

## 🎯 Use Case

Fine-tune a language model to generate SQL queries based on natural language instructions, enabling:
- Reduced time to create SQL queries
- Easier database access for non-technical users
- Integration with BI tools for automated query generation

## 📋 Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Hydra Configuration**: Flexible, hierarchical configuration management with YAML files
- **Environment-based Secrets**: Secure credential management with dotenv
- **Production Ready**: Error handling, logging, and validation
- **Efficient Training**: QLoRA with 4-bit quantization and Flash Attention 2
- **Comprehensive Evaluation**: Multiple evaluation metrics and example outputs
- **Makefile Support**: Convenient commands for common tasks
- **Automated Setup**: One-command development environment setup
- **Interactive Inference**: Real-time SQL generation from natural language
- **Hub Integration**: Easy model upload to Hugging Face Hub

## 🏗️ Project Structure

```
text-to-sql-finetuning/
├── .env.example              # Example environment variables
├── .gitignore               # Git ignore file
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Modern Python packaging configuration
├── environment.yml          # Conda environment specification
├── Makefile                 # Convenient commands for common tasks
├── setup_dev_env.sh        # Automated development environment setup
├── README.md               # This file
├── CONTRIBUTING.md         # Contribution guidelines
├── DEPLOYMENT.md           # Deployment strategies and guides
├── CHANGELOG.md            # Version history and changes
├── LICENSE                 # Apache-2.0 license
├── config/                 # Hydra-based configuration
│   ├── config.yaml         # Main configuration file
│   ├── config.py           # Configuration loading example
│   ├── __init__.py
│   ├── training/
│   │   └── training.yaml   # Training hyperparameters
│   ├── dataset/
│   │   └── dataset.yaml    # Dataset configuration
│   ├── hf/
│   │   └── hf.yaml         # Hugging Face settings
│   ├── wandb/
│   │   └── wandb.yaml      # Weights & Biases configuration
│   ├── inference/
│   │   └── inference.yaml  # Inference settings
│   └── evaluation/
│       └── evaluation.yaml # Evaluation parameters
├── src/
│   ├── __init__.py
│   ├── data_preparation.py  # Dataset loading and preprocessing
│   ├── model_setup.py       # Model and tokenizer initialization
│   ├── training.py          # Training logic with SFTTrainer
│   └── utils.py            # Utility functions
├── scripts/
│   ├── prepare_data.py      # Data preparation script
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   ├── inference.py        # Interactive inference script
│   └── upload_to_hf.py     # Upload model/adapter to Hugging Face Hub
├── data/                   # Dataset storage (created automatically)
└── logs/                   # Training and evaluation logs
```

## 🚀 Quick Start

### 1. Installation

#### Option A: Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/chrisjcc/text-to-sql-finetuning.git
cd text-to-sql-finetuning

# Run automated setup script (installs Miniconda, creates environment, installs dependencies)
bash setup_dev_env.sh
```

#### Option B: Manual Setup with Make

```bash
# Clone the repository
git clone https://github.com/chrisjcc/text-to-sql-finetuning.git
cd text-to-sql-finetuning

# Install dependencies and setup project
make setup

# Optional: Install Flash Attention (requires CUDA compute capability >= 8.0)
make install-flash
```

#### Option C: Traditional pip Install

```bash
# Clone the repository
git clone https://github.com/chrisjcc/text-to-sql-finetuning.git
cd text-to-sql-finetuning

# Install dependencies
pip install -r requirements.txt

# Optional: Install Flash Attention (requires CUDA compute capability >= 8.0)
pip install ninja packaging
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

### 2. Configuration

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` and add your Hugging Face token:

```bash
HF_TOKEN=your_huggingface_token_here
```

### 3. Prepare Dataset

```bash
# Using Make (recommended)
make prepare-data

# Or directly with Python
python -m scripts.prepare_data
```

This will:
- Download the SQL dataset from Hugging Face
- Convert it to conversational format
- Split into train/test sets
- Save to `data/` directory

### 4. Train Model

```bash
# Using Make (auto-detects accelerate or python)
make train

# Or force specific method
make train-accelerate  # Use accelerate launch
make train-basic      # Use python -m

# Or directly with Python
python -m scripts.train

# Or with accelerate
accelerate launch -m scripts.train
```

Training options (Hydra overrides):
- `training.resume_from_checkpoint=true`: Resume from last checkpoint
- `training.num_train_epochs=5`: Change number of epochs
- `training.learning_rate=1e-4`: Adjust learning rate
- Example:
  ```bash
  python -m scripts.train training.resume_from_checkpoint=true training.num_train_epochs=5
  ```

Training configuration (see `config/training/training.yaml`):
- Model: Meta-Llama-3.1-8B
- Quantization: 4-bit with QLoRA
- Attention: Flash Attention 2 (if supported, falls back to SDPA)
- LoRA rank: 8
- LoRA alpha: 16
- Training epochs: 3
- Expected time: ~2 hours on g6.2xlarge

### 5. Evaluate Model

```bash
# Using Make
make evaluate

# Or directly with Python
python -m scripts.evaluate
```

This will:
- Load the fine-tuned model
- Show example predictions
- Evaluate on test samples
- Report accuracy metrics

### 6. Interactive Inference

```bash
# Using Make
make inference

# Or directly with Python
python -m scripts.inference --interactive
```

### 7. Upload to Hugging Face Hub (Optional)

```bash
# Using Make
make upload-to-hf

# Or directly with Python
python -m scripts.upload_to_hf

# Upload options (use Hydra overrides)
python -m scripts.upload_to_hf skip_merge=true    # Upload adapter only (lightweight)
python -m scripts.upload_to_hf skip_upload=true   # Merge only, don't upload
python -m scripts.upload_to_hf private=true       # Create private repository
```

Note: Merging requires significant memory (>30GB). If you encounter OOM errors, upload just the adapter.

## ⚙️ Configuration

This project uses **Hydra** for configuration management. Configuration files are organized in the `config/` directory:

### Configuration Structure

```
config/
├── config.yaml              # Main config (combines all below)
├── training/training.yaml   # Training hyperparameters
├── dataset/dataset.yaml     # Dataset settings
├── hf/hf.yaml              # Hugging Face settings
├── wandb/wandb.yaml        # Weights & Biases settings
├── inference/inference.yaml # Inference parameters
└── evaluation/evaluation.yaml # Evaluation settings
```

### Environment Variables (.env file)

Sensitive credentials are stored in `.env`:

```bash
# Hugging Face (Required)
HF_TOKEN=your_token_here
HF_USERNAME=your_hf_username  # Required for uploading to Hub

# Weights & Biases (Optional)
WANDB_API_KEY=your_wandb_key_here  # Optional: for experiment tracking
```

### Modifying Configuration

You can override any configuration parameter using Hydra's command-line syntax:

```bash
# Override training parameters
python -m scripts.train training.num_train_epochs=5 training.learning_rate=1e-4

# Override dataset parameters
python -m scripts.train dataset.train_samples=20000 dataset.test_samples=5000

# Override multiple parameters
python -m scripts.train training.per_device_train_batch_size=2 training.lora_r=16
```

### Key Configuration Parameters

**Training** (`config/training/training.yaml`):
- `output_dir`: Directory for model checkpoints
- `num_train_epochs`: Number of training epochs (default: 3)
- `per_device_train_batch_size`: Batch size per device (default: 1)
- `gradient_accumulation_steps`: Gradient accumulation (default: 8)
- `learning_rate`: Learning rate (default: 2e-4)
- `max_seq_length`: Maximum sequence length (default: 2048)
- `lora_r`: LoRA rank (default: 8)
- `lora_alpha`: LoRA alpha (default: 16)
- `lora_dropout`: LoRA dropout (default: 0.05)

**Dataset** (`config/dataset/dataset.yaml`):
- `name`: HuggingFace dataset name (default: b-mc2/sql-create-context)
- `train_samples`: Number of training samples (default: 10000)
- `test_samples`: Number of test samples (default: 2500)

## 📊 Model Architecture

### Base Model
- **Meta-Llama-3.1-8B**: 8 billion parameter model
- **Quantization**: 4-bit NF4 with double quantization
- **Attention**: Flash Attention 2 for efficient training

### LoRA Configuration
- **Rank (r)**: 8 (configurable in `config/training/training.yaml`)
- **Alpha**: 16
- **Dropout**: 0.05
- **Target modules**: all-linear layers

### Training Strategy
- **Optimizer**: AdamW (fused)
- **Learning rate**: 2e-4 (constant)
- **Gradient accumulation**: 8 steps
- **Mixed precision**: BF16
- **Gradient checkpointing**: Enabled

## 📈 Expected Results

Based on the original implementation:
- **Training time**: ~2 hours on g6.2xlarge (24GB GPU)
- **Evaluation accuracy**: ~80% (exact match on 1000 samples)
- **Model size**: ~4GB (quantized with LoRA adapters)

## 🔧 Advanced Usage

### Using Makefile Commands

The project includes a `Makefile` for convenience:

```bash
make help              # Show all available commands
make setup             # Initial project setup
make prepare-data      # Prepare datasets
make train             # Train model (auto-detects accelerate)
make train-accelerate  # Force use accelerate
make train-basic       # Force use python
make evaluate          # Evaluate trained model
make inference         # Run interactive inference
make upload-to-hf      # Upload to Hugging Face Hub
make clean             # Clean generated files
```

### Custom Dataset

Modify the dataset configuration or pass overrides:

```bash
# Override dataset name
python -m scripts.train dataset.name=your/dataset/name

# Or edit config/dataset/dataset.yaml directly
```

### Training with Different Models

Update `config/hf/hf.yaml` or use environment variable:

```bash
# In .env file
HF_MODEL_ID=mistralai/Mistral-7B-v0.1

# Or override via command line
python -m scripts.train hf.model_id=mistralai/Mistral-7B-v0.1
```

### Adjusting LoRA Parameters

Modify `config/training/training.yaml` or use Hydra overrides:

```bash
# Via command line
python -m scripts.train training.lora_r=16 training.lora_alpha=32 training.lora_dropout=0.1

# Or edit config/training/training.yaml directly
```

## 📝 Logging

Logs are automatically saved to the `logs/` directory:
- `logs/data_preparation.log`: Dataset preparation logs
- `logs/training.log`: Training progress and metrics
- `logs/evaluation.log`: Evaluation results

## 🧪 Testing

The evaluation script provides multiple metrics:
- **Exact match accuracy**: Compares generated SQL with ground truth
- **Example predictions**: Shows sample inputs and outputs
- **Success rate**: Percentage of correct predictions

### Alternative Evaluation Methods

For more robust evaluation, consider:
1. **Query execution**: Run queries against a database and compare results
2. **Semantic similarity**: Use embeddings to compare SQL semantics
3. **Human evaluation**: Manual review of generated queries

## 🚀 Deployment

For detailed deployment strategies, see [DEPLOYMENT.md](DEPLOYMENT.md).

### Quick Local Deployment

```python
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="./meta-llama-3-1-8B-text-to-sql-adapter"
)

# Generate SQL
messages = [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Show me all customers from New York"}
]
result = pipe(messages)
```

### Deployment Options

- **Local Deployment**: Run inference on your own hardware
- **Hugging Face Inference Endpoints**: Managed API endpoints
- **Docker Deployment**: Containerized deployment
- **API Server**: FastAPI-based REST API

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete guides on each option.

## 🔒 Security

- **Never commit** `.env` file or tokens to version control
- Store Hugging Face tokens securely
- Use appropriate access controls for production deployments
- Validate and sanitize generated SQL before execution

## 🐛 Troubleshooting

### Flash Attention Issues

If Flash Attention installation fails:
```bash
# Check compute capability
python -c "import torch; print(torch.cuda.get_device_capability())"

# If < 8.0, the code will automatically fall back to SDPA
# No manual configuration needed
```

### Out of Memory

Reduce batch size or sequence length in `config/training/training.yaml` or via overrides:
```bash
python -m scripts.train training.per_device_train_batch_size=1 \
  training.gradient_accumulation_steps=16 \
  training.max_seq_length=1024
```

### Dataset Download Issues

Ensure you have Hugging Face access:
```bash
huggingface-cli login
# Or set HF_TOKEN in .env file
```

### Hydra Configuration Errors

If you encounter Hydra-related errors:
```bash
# Clear Hydra cache
rm -rf outputs/ .hydra/

# Check config syntax
python -c "from omegaconf import OmegaConf; OmegaConf.load('config/config.yaml')"
```

## 📚 References

- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Hydra Documentation](https://hydra.cc/docs/intro/)

## ⚠️ Important Notes

### Configuration Management with Hydra

This project uses **Hydra** for flexible configuration management:

- All configuration is split into modular YAML files in the `config/` directory
- Override any parameter via command line: `python -m scripts.train training.learning_rate=1e-4`
- Sensitive credentials are stored in `.env` file, not in config files
- Hydra outputs are saved to `outputs/` directory (can be excluded from version control)

### API Changes in Latest TRL

This codebase uses the **latest TRL API** which requires:

1. **Pre-formatting datasets**: Apply chat template to dataset before training
   ```python
   # Format dataset with chat template
   train_dataset = train_dataset.map(format_for_training)

   # Then pass to SFTTrainer (simplified API)
   trainer = SFTTrainer(
       model=model,
       args=args,
       train_dataset=train_dataset,  # Already formatted
       peft_config=peft_config,
   )
   ```

2. **Flash Attention fallback**: Gracefully falls back to SDPA if Flash Attention unavailable
   ```python
   try:
       import flash_attn
       attn_implementation = "flash_attention_2"
   except ImportError:
       attn_implementation = "sdpa"
   ```

3. **Resume from checkpoint**: Supports automatic checkpoint resuming via configuration
   ```python
   # Set in config/training/training.yaml or via override
   training.resume_from_checkpoint=true
   ```

These changes ensure compatibility with the latest versions of `transformers`, `trl`, and `peft` libraries.

## 📄 License

Apache-2.0 license - see LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on:
- Development setup
- Coding standards
- Testing requirements
- Pull request process

## 📧 Contact

For questions or issues, please open a GitHub issue or contact [chrisjcc.physics@gmail.com].

**Note**: This is a research/educational project. Always validate SQL outputs before executing on production databases.
