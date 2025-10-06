"""
Configuration module for Text-to-SQL fine-tuning project.
Loads environment variables and provides configuration classes.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class HuggingFaceConfig:
    """Configuration for Hugging Face authentication and model."""
    
    token: str
    model_id: str
    
    @classmethod
    def from_env(cls) -> "HuggingFaceConfig":
        """Create configuration from environment variables."""
        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN not found in environment variables")
        
        model_id = os.getenv("HF_MODEL_ID", "meta-llama/Meta-Llama-3.1-8B")
        return cls(token=token, model_id=model_id)


@dataclass
class DatasetConfig:
    """Configuration for dataset preparation."""
    
    dataset_name: str
    train_samples: int
    test_samples: int
    train_path: Path
    test_path: Path
    
    @classmethod
    def from_env(cls) -> "DatasetConfig":
        """Create configuration from environment variables."""
        dataset_name = os.getenv("DATASET_NAME", "b-mc2/sql-create-context")
        train_samples = int(os.getenv("TRAIN_SAMPLES", "10000"))
        test_samples = int(os.getenv("TEST_SAMPLES", "2500"))
        train_path = Path(os.getenv("TRAIN_DATASET_PATH", "data/train_dataset.json"))
        test_path = Path(os.getenv("TEST_DATASET_PATH", "data/test_dataset.json"))
        
        # Create data directory if it doesn't exist
        train_path.parent.mkdir(parents=True, exist_ok=True)
        
        return cls(
            dataset_name=dataset_name,
            train_samples=train_samples,
            test_samples=test_samples,
            train_path=train_path,
            test_path=test_path,
        )


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    output_dir: str
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    max_seq_length: int
    
    # LoRA parameters
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_r: int = 256
    
    # Additional training parameters
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    
    @classmethod
    def from_env(cls) -> "TrainingConfig":
        """Create configuration from environment variables."""
        return cls(
            output_dir=os.getenv("OUTPUT_DIR", "code-llama-3-1-8b-text-to-sql"),
            num_train_epochs=int(os.getenv("NUM_TRAIN_EPOCHS", "3")),
            per_device_train_batch_size=int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", "1")),
            gradient_accumulation_steps=int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "8")),
            learning_rate=float(os.getenv("LEARNING_RATE", "2e-4")),
            max_seq_length=int(os.getenv("MAX_SEQ_LENGTH", "2048")),
        )


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    
    model_path: str
    test_dataset_path: Path
    num_eval_samples: int
    
    @classmethod
    def from_env(cls) -> "EvaluationConfig":
        """Create configuration from environment variables."""
        output_dir = os.getenv("OUTPUT_DIR", "code-llama-3-1-8b-text-to-sql")
        test_path = Path(os.getenv("TEST_DATASET_PATH", "data/test_dataset.json"))
        num_eval_samples = int(os.getenv("NUM_EVAL_SAMPLES", "1000"))
        
        return cls(
            model_path=f"./{output_dir}",
            test_dataset_path=test_path,
            num_eval_samples=num_eval_samples,
        )


@dataclass
class Config:
    """Main configuration class combining all sub-configurations."""
    
    hf: HuggingFaceConfig
    dataset: DatasetConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    
    @classmethod
    def load(cls) -> "Config":
        """Load all configurations from environment variables."""
        return cls(
            hf=HuggingFaceConfig.from_env(),
            dataset=DatasetConfig.from_env(),
            training=TrainingConfig.from_env(),
            evaluation=EvaluationConfig.from_env(),
        )
