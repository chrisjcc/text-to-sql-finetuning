"""
Training module for fine-tuning models using SFTTrainer.
"""

import logging
from typing import Optional

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from datasets import Dataset

from transformers import set_seed


logger = logging.getLogger(__name__)


def format_dataset_for_training(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int = 2048,
) -> Dataset:
    """
    Format dataset by applying chat template to create text field.
    This is required for the latest trl SFTTrainer API.

    Args:
        dataset: Dataset with 'messages' field
        tokenizer: Tokenizer with chat template

    Returns:
        Formatted dataset with 'text' field
    """
    logger.info("Formatting dataset with chat template...")

    def format_for_training(example):
        """Apply chat template and return as text field"""
        formatted_text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,   # don't tokenize here; leave for trainer
            max_length=max_seq_length,  # truncate to desired length
            add_generation_prompt=True
        )
        return {"text": formatted_text}

    formatted_dataset = dataset.map(
        format_for_training,
        remove_columns=dataset.column_names,
        desc="Applying chat template"
    )

    sample = formatted_dataset[0]['text']
    token_len = len(tokenizer(sample).input_ids)
    logger.info(f"Sample formatted text ({token_len} tokens):\n{sample[:500]}...")
    return formatted_dataset


class ModelTrainer:
    """Handles model training with SFTTrainer and LoRA."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        peft_config: LoraConfig,
        output_dir: str,
        max_seq_length: int = 2048,
    ):
        """
        Initialize the model trainer.

        Args:
            model: Pre-trained model to fine-tune
            tokenizer: Tokenizer for the model
            train_dataset: Training dataset (should have 'text' field)
            peft_config: PEFT/LoRA configuration
            output_dir: Directory to save model checkpoints
            max_seq_length: Maximum sequence length for training
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.peft_config = peft_config
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length
        self.trainer: Optional[SFTTrainer] = None

    def create_training_arguments(
        self,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        learning_rate: float = 2e-4,
        max_grad_norm: float = 0.3,
        warmup_ratio: float = 0.03,
        logging_steps: int = 10,
        push_to_hub: bool = False,
        report_to: Optional[str] = "tensorboard",
    ) -> SFTConfig:
        """
        Create training arguments for the trainer.

        Args:
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size per device
            gradient_accumulation_steps: Number of gradient accumulation steps
            learning_rate: Learning rate
            max_grad_norm: Maximum gradient norm
            warmup_ratio: Warmup ratio for learning rate scheduler
            logging_steps: Number of steps between logging
            push_to_hub: Whether to push model to Hugging Face Hub

        Returns:
            SFTConfig object
        """
        logger.info("Creating SFT configuration")
        tf32 = torch.cuda.get_device_capability()[0] >= 8  # Only available on NVIDIA Ampere or newer GPUs (e.g. A100, RTX 30xx, H100)

        return SFTConfig(
            # Training arguments
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            logging_steps=logging_steps,
            save_strategy="epoch",
            learning_rate=learning_rate,
            save_safetensors=True,
            packing=True,
            bf16=True,
            tf32=tf32,
            max_grad_norm=max_grad_norm,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type="constant",
            push_to_hub=push_to_hub,
            report_to=report_to,
            # SFT-specific arguments
            dataset_text_field="text",
        )

    def create_trainer(self, training_args: SFTConfig) -> SFTTrainer:
        """
        Create SFTTrainer instance.
        Updated to use the new simplified API that works with pre-formatted datasets.

        Args:
            training_args: Training arguments

        Returns:
            Configured SFTTrainer
        """
        logger.info("Creating SFTTrainer")

        try:
            # New simplified API - dataset should already have 'text' field
            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                peft_config=self.peft_config,
            )
            logger.info("SFTTrainer created successfully")
            return trainer

        except Exception as e:
            logger.error(f"Failed to create trainer: {e}")
            raise

    def train(self, training_args: SFTConfig, resume_from_checkpoint: Optional[str] = None) -> None:
        """
        Train the model.

        Args:
            training_args: Training arguments
            resume_from_checkpoint: Whether to resume from latest checkpoint if available
        """
        set_seed(42)
        logger.info("Starting training")

        # Create trainer
        self.trainer = self.create_trainer(training_args)

        # Only pass a valid path or None
        checkpoint_path = None
        if resume_from_checkpoint is True:
            checkpoint_path = self.output_dir
        elif isinstance(resume_from_checkpoint, str):
            checkpoint_path = resume_from_checkpoint
        # else: start fresh, leave checkpoint_path as None

        # Start training
        try:
            # This ensures training resumes automatically if a checkpoint exists.
            self.trainer.train(resume_from_checkpoint=checkpoint_path)
            logger.info("Training completed successfully")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def save_model(self) -> None:
        """Save the trained model and tokenizer."""
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call train() first.")

        logger.info(f"Saving model to {self.output_dir}")
        try:
            self.trainer.save_model()
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def cleanup(self) -> None:
        """Clean up GPU memory after training."""
        logger.info("Cleaning up GPU memory")
        del self.model
        del self.trainer
        torch.cuda.empty_cache()
        logger.info("Cleanup completed")


def train_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    peft_config: LoraConfig,
    training_args: SFTConfig,
    resume_from_checkpoint: bool = True,
) -> ModelTrainer:
    """
    Convenience function to train a model.

    Args:
        model: Pre-trained model
        tokenizer: Tokenizer
        train_dataset: Training dataset (should have 'text' field)
        peft_config: LoRA configuration
        training_args: Training arguments
        resume_from_checkpoint: Whether to resume from checkpoint

    Returns:
        ModelTrainer instance
    """
    trainer_wrapper = ModelTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        peft_config=peft_config,
        output_dir=training_args.output_dir,
    )

    trainer_wrapper.train(training_args, resume_from_checkpoint=resume_from_checkpoint)
    trainer_wrapper.save_model()

    return trainer_wrapper
