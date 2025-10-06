"""
Training module for fine-tuning models using SFTTrainer.
"""

import logging
from typing import Optional

import torch
from transformers import (
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from trl import SFTTrainer
from peft import LoraConfig
from datasets import Dataset

logger = logging.getLogger(__name__)


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
            train_dataset: Training dataset
            peft_config: PEFT/LoRA configuration
            output_dir: Directory to save model checkpoints
            max_seq_length: Maximum sequence length
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
        push_to_hub: bool = True,
    ) -> TrainingArguments:
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
            TrainingArguments object
        """
        logger.info("Creating training arguments")
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            logging_steps=logging_steps,
            save_strategy="epoch",
            learning_rate=learning_rate,
            bf16=True,
            tf32=True,
            max_grad_norm=max_grad_norm,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type="constant",
            push_to_hub=push_to_hub,
            report_to="tensorboard",
        )
    
    def create_trainer(self, training_args: TrainingArguments) -> SFTTrainer:
        """
        Create SFTTrainer instance.
        
        Args:
            training_args: Training arguments
            
        Returns:
            Configured SFTTrainer
        """
        logger.info("Creating SFTTrainer")
        
        try:
            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                peft_config=self.peft_config,
                max_seq_length=self.max_seq_length,
                tokenizer=self.tokenizer,
                packing=True,
                dataset_kwargs={
                    "add_special_tokens": False,  # We template with special tokens
                    "append_concat_token": False,  # No need to add additional separator token
                }
            )
            logger.info("SFTTrainer created successfully")
            return trainer
            
        except Exception as e:
            logger.error(f"Failed to create trainer: {e}")
            raise
    
    def train(self, training_args: TrainingArguments) -> None:
        """
        Train the model.
        
        Args:
            training_args: Training arguments
        """
        logger.info("Starting training")
        
        # Create trainer
        self.trainer = self.create_trainer(training_args)
        
        # Start training
        try:
            self.trainer.train()
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
    training_args: TrainingArguments,
    max_seq_length: int = 2048,
) -> ModelTrainer:
    """
    Convenience function to train a model.
    
    Args:
        model: Pre-trained model
        tokenizer: Tokenizer
        train_dataset: Training dataset
        peft_config: LoRA configuration
        training_args: Training arguments
        max_seq_length: Maximum sequence length
        
    Returns:
        ModelTrainer instance
    """
    trainer_wrapper = ModelTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        peft_config=peft_config,
        output_dir=training_args.output_dir,
        max_seq_length=max_seq_length,
    )
    
    trainer_wrapper.train(training_args)
    trainer_wrapper.save_model()
    
    return trainer_wrapper
