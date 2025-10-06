"""
Text-to-SQL Fine-tuning Package

This package provides tools for fine-tuning language models for SQL generation.
"""

__version__ = "1.0.0"

from .data_preparation import DatasetProcessor, prepare_and_save_datasets
from .model_setup import ModelSetup, initialize_model_for_training
from .training import ModelTrainer, train_model
from .utils import (
    setup_logging,
    authenticate_huggingface,
    check_gpu_availability,
    check_flash_attention_support,
    print_trainable_parameters,
)

__all__ = [
    "DatasetProcessor",
    "prepare_and_save_datasets",
    "ModelSetup",
    "initialize_model_for_training",
    "ModelTrainer",
    "train_model",
    "setup_logging",
    "authenticate_huggingface",
    "check_gpu_availability",
    "check_flash_attention_support",
    "print_trainable_parameters",
]
