# Text-to-SQL Fine-tuning

Production-ready implementation for fine-tuning language models to generate SQL queries from natural language instructions. This project uses QLoRA for efficient fine-tuning on 24GB GPUs.

## 🎯 Use Case

Fine-tune a language model to generate SQL queries based on natural language instructions, enabling:
- Reduced time to create SQL queries
- Easier database access for non-technical users
- Integration with BI tools for automated query generation

## 📋 Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Environment-based Configuration**: Secure credential management with dotenv
- **Production Ready**: Error handling, logging, and validation
- **Efficient Training**: QLoRA with 4-bit quantization and Flash Attention
- **Comprehensive Evaluation**: Multiple evaluation metrics and example outputs

## 🏗️ Project Structure

```
text-to-sql-finetuning/
├── .env.example              # Example environment variables
├── .gitignore               # Git ignore file
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── config/
│   └── config.py           # Configuration management
├── src/
│   ├── __init__.py
│   ├── data_preparation.py  # Dataset loading and preprocessing
│   ├── model_setup.py       # Model and tokenizer initialization
│   ├── training.py          # Training logic with SFTTrainer
│   └── utils.py            # Utility functions
├── scripts/
│   ├── prepare_data.py      # Data preparation script
│   ├── train.py            # Training script
│   └── evaluate.py         # Evaluation script
└── data/                   # Dataset storage (created automatically)
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd text-to-sql-finetuning

# Install dependencies
pip install -r requirements.txt

# Install Flash Attention (requires CUDA compute capability >= 8.0)
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
python scripts/prepare_data.py
```

This will:
- Download the SQL dataset from Hugging Face
- Convert it to conversational format
- Split into train/test sets
- Save to `data/` directory

### 4. Train Model

```bash
python scripts/train.py
```

Training configuration:
- Model: Meta-Llama-3.1-8B
- Quantization: 4-bit with QLoRA
- Attention: Flash Attention 2 (if supported)
- LoRA rank: 256
- Training epochs: 3
- Expected time: ~2 hours on g6.2xlarge

### 5. Evaluate Model

```bash
python scripts/evaluate.py
```

This will:
- Load the fine-tuned model
- Show example predictions
- Evaluate on 1000 test samples
- Report accuracy metrics

## ⚙️ Configuration

All configuration is managed through environment variables in the `.env` file:

### Hugging Face Settings
```bash
HF_TOKEN=your_token_here
HF_MODEL_ID=meta-llama/Meta-Llama-3.1-8B
```

### Training Parameters
```bash
OUTPUT_DIR=code-llama-3-1-8b-text-to-sql
NUM_TRAIN_EPOCHS=3
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE=2e-4
MAX_SEQ_LENGTH=2048
```

### Dataset Configuration
```bash
DATASET_NAME=b-mc2/sql-create-context
TRAIN_SAMPLES=10000
TEST_SAMPLES=2500
```

## 📊 Model Architecture

### Base Model
- **Meta-Llama-3.1-8B**: 8 billion parameter model
- **Quantization**: 4-bit NF4 with double quantization
- **Attention**: Flash Attention 2 for efficient training

### LoRA Configuration
- **Rank (r)**: 256
- **Alpha**: 128
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

### Custom Dataset

Modify `src/data_preparation.py` to use your own dataset:

```python
processor = DatasetProcessor("your/dataset/name")
```

### Training with Different Models

Update `.env` to use alternative models:

```bash
HF_MODEL_ID=mistralai/Mistral-7B-v0.1
```

### Adjusting LoRA Parameters

Modify `config/config.py` in the `TrainingConfig` class:

```python
lora_alpha: int = 128
lora_dropout: float = 0.05
lora_r: int = 256
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

### Using Hugging Face Inference Endpoints

1. Push your model to Hugging Face Hub (set `push_to_hub=True` in training)
2. Create an Inference Endpoint in Hugging Face
3. Use the API for production inference

### Local Deployment

```python
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="./code-llama-3-1-8b-text-to-sql"
)

# Generate SQL
messages = [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Show me all customers from New York"}
]
result = pipe(messages)
```

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

# If < 8.0, disable Flash Attention in train.py
use_flash_attention = False
```

### Out of Memory

Reduce batch size or sequence length in `.env`:
```bash
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=16
MAX_SEQ_LENGTH=1024
```

### Dataset Download Issues

Ensure you have Hugging Face access:
```bash
huggingface-cli login
```

## 📚 References

- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [PEFT Documentation](https://huggingface.co/docs/peft)

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Contact

For questions or issues, please open a GitHub issue or contact [chrisjcc.physics@gmail.com].

**Note**: This is a research/educational project. Always validate SQL outputs before executing on production databases.
