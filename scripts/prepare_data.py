"""
Data preparation module for Text-to-SQL fine-tuning using Hydra hierarchical configs.
Handles dataset loading, processing, conversion, and saving.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Any
from datasets import load_dataset, Dataset, DatasetDict
from dotenv import load_dotenv
import os
import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import setup_logging

logger = logging.getLogger(__name__)

# Load secrets from .env
load_dotenv()


class DatasetProcessor:
    """Handles dataset loading and preprocessing for text-to-SQL tasks."""

    SYSTEM_MESSAGE_TEMPLATE = """You are a text-to-SQL query translator. Users will ask questions in English and you will generate a SQL query based on the provided SCHEMA.
SCHEMA:
{schema}"""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def load_dataset(self, split: str = "train") -> Dataset:
        """Load dataset from Hugging Face Hub using token (avoids use_auth_token deprecation)."""
        hf_token = os.getenv("HF_TOKEN")
        logger.info(f"Loading dataset '{self.dataset_name}', split={split}")

        try:
            dataset = load_dataset(self.dataset_name, split=split, token=hf_token)
            logger.info(f"Successfully loaded {len(dataset)} samples")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def create_conversation(self, sample: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
        """Convert a dataset sample to conversational format."""
        return {
            "messages": [
                {"role": "system", "content": self.SYSTEM_MESSAGE_TEMPLATE.format(schema=sample["context"])},
                {"role": "user", "content": sample["question"]},
                {"role": "assistant", "content": sample["answer"]},
            ]
        }

    def prepare_dataset(self, total_samples: int, test_size: int, seed: int = 42) -> DatasetDict:
        """Prepare and split dataset for training and testing."""
        logger.info(f"Preparing dataset with {total_samples} samples")

        dataset = self.load_dataset().shuffle(seed=seed).select(range(total_samples))

        logger.info("Converting dataset to conversational format")
        dataset = dataset.map(self.create_conversation, remove_columns=dataset.features, batched=False, desc="Converting to conversation format")

        test_ratio = test_size / total_samples
        dataset = dataset.train_test_split(test_size=test_ratio, seed=seed)

        logger.info(f"Train samples: {len(dataset['train'])}, Test samples: {len(dataset['test'])}")
        return dataset

    def save_datasets(self, dataset: DatasetDict, train_path: Path, test_path: Path):
        """Save datasets to disk in JSON format."""
        train_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving training dataset to {train_path}")
        dataset["train"].to_json(str(train_path), orient="records")

        logger.info(f"Saving test dataset to {test_path}")
        dataset["test"].to_json(str(test_path), orient="records")

        logger.info("Datasets saved successfully")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """
    Prepare and save datasets using hierarchical Hydra config.

    Args:
        cfg: DictConfig loaded from Hydra YAMLs
    """
    # Setup logging - use get_original_cwd() to write to project root
    log_file = Path(get_original_cwd()) / "logs" / "prepare_data.log"
    setup_logging(log_file)

    processor = DatasetProcessor(cfg.dataset.name)
    total_samples = cfg.dataset.train_samples + cfg.dataset.test_samples
    dataset = processor.prepare_dataset(total_samples=total_samples, test_size=cfg.dataset.test_samples)

    train_path = Path(cfg.dataset.train_dataset_path).resolve()
    test_path = Path(cfg.dataset.test_dataset_path).resolve()
    processor.save_datasets(dataset, train_path, test_path)

    logger.info(f"Datasets saved: {train_path}, {test_path}")
    logger.info("Example conversation:\n%s", dataset["train"][0]["messages"])


if __name__ == "__main__":
    main()
