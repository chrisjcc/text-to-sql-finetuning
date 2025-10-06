"""
Data preparation module for Text-to-SQL fine-tuning.
Handles dataset loading, processing, and conversion.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any

from datasets import load_dataset, Dataset, DatasetDict

logger = logging.getLogger(__name__)


class DatasetProcessor:
    """Handles dataset loading and preprocessing for text-to-SQL tasks."""
    
    SYSTEM_MESSAGE_TEMPLATE = """You are an text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA.
SCHEMA:
{schema}"""
    
    def __init__(self, dataset_name: str):
        """
        Initialize the dataset processor.
        
        Args:
            dataset_name: Name of the dataset to load from Hugging Face Hub
        """
        self.dataset_name = dataset_name
        
    def load_dataset(self, split: str = "train") -> Dataset:
        """
        Load dataset from Hugging Face Hub.
        
        Args:
            split: Dataset split to load (default: "train")
            
        Returns:
            Loaded dataset
        """
        logger.info(f"Loading dataset: {self.dataset_name}, split: {split}")
        try:
            dataset = load_dataset(self.dataset_name, split=split)
            logger.info(f"Successfully loaded {len(dataset)} samples")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def create_conversation(self, sample: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
        """
        Convert a dataset sample to conversational format.
        
        Args:
            sample: Raw dataset sample with 'context', 'question', and 'answer' keys
            
        Returns:
            Dictionary with 'messages' key containing conversation history
        """
        return {
            "messages": [
                {
                    "role": "system",
                    "content": self.SYSTEM_MESSAGE_TEMPLATE.format(schema=sample["context"])
                },
                {
                    "role": "user",
                    "content": sample["question"]
                },
                {
                    "role": "assistant",
                    "content": sample["answer"]
                }
            ]
        }
    
    def prepare_dataset(
        self,
        total_samples: int,
        test_size: int,
        seed: int = 42
    ) -> DatasetDict:
        """
        Prepare and split dataset for training and testing.
        
        Args:
            total_samples: Total number of samples to use
            test_size: Number of samples for test set
            seed: Random seed for shuffling
            
        Returns:
            DatasetDict with 'train' and 'test' splits
        """
        logger.info(f"Preparing dataset with {total_samples} samples")
        
        # Load and sample dataset
        dataset = self.load_dataset()
        dataset = dataset.shuffle(seed=seed).select(range(total_samples))
        
        # Convert to conversational format
        logger.info("Converting dataset to conversational format")
        dataset = dataset.map(
            self.create_conversation,
            remove_columns=dataset.features,
            batched=False,
            desc="Converting to conversation format"
        )
        
        # Split into train and test
        test_ratio = test_size / total_samples
        logger.info(f"Splitting dataset: {1-test_ratio:.1%} train, {test_ratio:.1%} test")
        dataset = dataset.train_test_split(test_size=test_ratio, seed=seed)
        
        logger.info(f"Train samples: {len(dataset['train'])}")
        logger.info(f"Test samples: {len(dataset['test'])}")
        
        return dataset
    
    def save_datasets(
        self,
        dataset: DatasetDict,
        train_path: Path,
        test_path: Path
    ) -> None:
        """
        Save train and test datasets to disk.
        
        Args:
            dataset: DatasetDict with 'train' and 'test' splits
            train_path: Path to save training dataset
            test_path: Path to save test dataset
        """
        logger.info(f"Saving training dataset to {train_path}")
        dataset["train"].to_json(str(train_path), orient="records")
        
        logger.info(f"Saving test dataset to {test_path}")
        dataset["test"].to_json(str(test_path), orient="records")
        
        logger.info("Datasets saved successfully")
    
    def load_prepared_dataset(self, data_path: Path) -> Dataset:
        """
        Load a prepared dataset from disk.
        
        Args:
            data_path: Path to the JSON dataset file
            
        Returns:
            Loaded dataset
        """
        logger.info(f"Loading prepared dataset from {data_path}")
        try:
            dataset = load_dataset("json", data_files=str(data_path), split="train")
            logger.info(f"Successfully loaded {len(dataset)} samples")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load prepared dataset: {e}")
            raise


def prepare_and_save_datasets(
    dataset_name: str,
    total_samples: int,
    test_samples: int,
    train_path: Path,
    test_path: Path,
    seed: int = 42
) -> DatasetDict:
    """
    Convenience function to prepare and save datasets.
    
    Args:
        dataset_name: Name of the dataset to load
        total_samples: Total number of samples to use
        test_samples: Number of samples for test set
        train_path: Path to save training dataset
        test_path: Path to save test dataset
        seed: Random seed for reproducibility
        
    Returns:
        Prepared DatasetDict
    """
    processor = DatasetProcessor(dataset_name)
    dataset = processor.prepare_dataset(total_samples, test_samples, seed)
    processor.save_datasets(dataset, train_path, test_path)
    
    # Print example
    logger.info("\nExample conversation:")
    logger.info(dataset["train"][0]["messages"])
    
    return dataset
