"""
Utility functions for the text-to-SQL fine-tuning project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import login


def setup_logging(
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_file: Optional path to log file
        level: Logging level
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def authenticate_huggingface(token: str) -> None:
    """
    Authenticate with Hugging Face Hub.
    
    Args:
        token: Hugging Face API token
    """
    logger = logging.getLogger(__name__)
    logger.info("Authenticating with Hugging Face Hub")
    
    try:
        login(token=token, add_to_git_credential=True)
        logger.info("Successfully authenticated with Hugging Face")
    except Exception as e:
        logger.error(f"Failed to authenticate with Hugging Face: {e}")
        raise


def check_gpu_availability() -> None:
    """Check if GPU is available and log device information."""
    logger = logging.getLogger(__name__)
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"GPU available: {device_count} device(s)")
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            device_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"  Device {i}: {device_name} ({device_memory:.2f} GB)")
    else:
        logger.warning("No GPU available. Training will be slow on CPU.")


def check_flash_attention_support() -> bool:
    """
    Check if the current GPU supports Flash Attention.
    
    Returns:
        True if Flash Attention is supported, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    if not torch.cuda.is_available():
        logger.warning("No GPU available. Flash Attention requires CUDA.")
        return False
    
    try:
        compute_capability = torch.cuda.get_device_capability()[0]
        if compute_capability >= 8:
            logger.info(f"GPU compute capability: {compute_capability} - Flash Attention supported")
            return True
        else:
            logger.warning(
                f"GPU compute capability: {compute_capability} - Flash Attention requires >= 8.0"
            )
            return False
    except Exception as e:
        logger.error(f"Failed to check compute capability: {e}")
        return False


def print_trainable_parameters(model) -> None:
    """
    Print the number of trainable parameters in the model.
    
    Args:
        model: The model to inspect
    """
    logger = logging.getLogger(__name__)
    
    trainable_params = 0
    all_params = 0
    
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    trainable_percent = 100 * trainable_params / all_params
    
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"All parameters: {all_params:,}")
    logger.info(f"Trainable%: {trainable_percent:.2f}%")


def validate_file_exists(file_path: Path, file_description: str = "File") -> None:
    """
    Validate that a file exists.
    
    Args:
        file_path: Path to the file
        file_description: Description of the file for error messages
        
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"{file_description} not found: {file_path}")


def create_directory(directory: Path, description: str = "Directory") -> None:
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory: Path to the directory
        description: Description of the directory for logging
    """
    logger = logging.getLogger(__name__)
    
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"{description} created: {directory}")
    else:
        logger.info(f"{description} already exists: {directory}")


def get_model_size_mb(model) -> float:
    """
    Calculate the size of a model in megabytes.
    
    Args:
        model: The model to measure
        
    Returns:
        Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb
