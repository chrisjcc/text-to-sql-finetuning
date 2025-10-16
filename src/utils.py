"""
Utility functions for the text-to-SQL fine-tuning project.
"""

import logging
import re
import sys
from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv
from typing import Tuple
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
from peft import PeftModel, PeftConfig

import torch

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
    # Create formatters
    formatter = logging.Formatter(
        '[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Setup handlers
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add our handlers
    for handler in handlers:
        root_logger.addHandler(handler)


def authenticate_huggingface(token: Optional[str] = None) -> None:
    """
    Optionally authenticate with Hugging Face Hub. Login is only required
    for write operations (pushing models/datasets).

    Args:
        token: Hugging Face API token
    """
    logger = logging.getLogger(__name__)

    # Load token from .env if not provided
    if token is None:
        load_dotenv()
        token = os.environ.get("HF_TOKEN")

    if token:
        try:
            login(token=token, add_to_git_credential=True)
            logger.info("Successfully authenticated with Hugging Face")
        except Exception as e:
            logger.warning(f"Hugging Face login failed, continuing anyway: {e}")
    else:
        logger.info("No Hugging Face token provided; skipping login.")


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
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
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
    # Skip validation for remote Hugging Face repos
    if isinstance(file_path, str) and (
        file_path.startswith("http")
        or "/" in file_path  # covers "org/model" format
        and not Path(file_path).exists()
    ):
        print(f"🔹 Skipping local existence check for remote {file_description}: {file_path}")
        return

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
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024**2

def extract_sql(generated_text: str) -> str:
    """
    Extract a clean SQL query from model output.

    Handles:
      - Chat template output (system/user/assistant markers)
      - End-of-sequence tokens (<|im_end|>, <|_|>, etc.)
      - Markdown code blocks (```sql, ```SQL, ```)
      - Multi-line SQL queries
      - Trailing explanations and hallucinated text
      - SQL with or without semicolons
      - Hallucination markers (Marilyn, assistant responses, etc.)
      - Non-ASCII characters (Tibetan unicode artifacts, etc.)

    Args:
        generated_text: Raw model output

    Returns:
        Clean SQL string, preferably ending with a semicolon
    """
    if not generated_text:
        return ""

    text = generated_text.strip()

    # CRITICAL FIX: Extract assistant response from chat template
    # The model may generate: "system\n...\nuser\n...\nassistant\nSELECT..."
    if 'assistant' in text.lower():
        # Split by 'assistant' marker (case-insensitive)
        parts = re.split(r'\bassistant\b', text, flags=re.IGNORECASE, maxsplit=1)
        if len(parts) > 1:
            text = parts[1].strip()

    # Remove common end-of-sequence tokens that may appear after SQL
    # Examples: <|im_end|>, <|_|_|_|...>, etc.
    text = re.split(r'<\|[^|]*\|>', text)[0].strip()

    # Remove system/user prompts if they somehow remain
    text = re.split(r'\b(system|user)\b', text, flags=re.IGNORECASE)[0].strip()

    # Remove markdown code blocks, ignore case
    text = re.sub(r'```(?:sql)?', '', text, flags=re.IGNORECASE)
    text = text.strip()

    # NEW: Stop at common hallucination markers
    hallucination_markers = [
        r'\bMarilyn\b',
        r'\bI am\b',
        r'\bYou are\b',
        r'\bThe (answer|query|SQL|result)\b',
        r'\bWhat (is|was|are)\b',
        r'\bWhich\b',
        r'\bNote:',
        r'\bHere',
    ]
    for marker in hallucination_markers:
        match = re.search(marker, text, flags=re.IGNORECASE)
        if match:
            text = text[:match.start()].strip()
            break

    # NEW: Stop at non-ASCII characters (like Tibetan unicode)
    ascii_match = re.search(r'[^\x00-\x7F]', text)
    if ascii_match:
        text = text[:ascii_match.start()].strip()

    # Prefer up to first semicolon if present
    if ';' in text:
        sql = text.split(';')[0].strip() + ';'
        return sql

    # FIXED: Collect multi-line SQL until we hit hallucination markers or double newlines
    # SQL queries can span multiple lines, so we need to be careful not to truncate them

    # Stop at double newline (indicates end of SQL and start of explanation)
    if '\n\n' in text:
        text = text.split('\n\n')[0].strip()

    # Stop at triple dash or '###' separators
    for separator in ['\n---\n', '\n###']:
        if separator in text:
            text = text.split(separator)[0].strip()

    # Return the complete multi-line SQL query
    return text.strip()
