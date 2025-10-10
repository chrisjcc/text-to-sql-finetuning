"""
Training script for Text-to-SQL fine-tuning using Hydra hierarchical configs.

Usage:
    python scripts/train.py
    python scripts/train.py training.num_train_epochs=5  # override parameters from CLI
"""

import sys
from pathlib import Path
import os
import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv

from pathlib import Path
from hydra.utils import get_original_cwd

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_preparation import DatasetProcessor
from src.model_setup import initialize_model_for_training, ModelSetup
from src.training import ModelTrainer, format_dataset_for_training
from src.utils import (
    setup_logging,
    authenticate_huggingface,
    check_gpu_availability,
    print_trainable_parameters,
    validate_file_exists,
)

# Load environment variables from .env before Hydra sees them
load_dotenv()


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """Main training function using hierarchical Hydra configs."""
    # Setup logging
    log_file = Path(get_original_cwd()) / "logs" / "training.log"
    setup_logging(log_file=log_file)

    # Authenticate with Hugging Face using token
    if cfg.hf.token:
        authenticate_huggingface(cfg.hf.token)

    # Check GPU availability
    check_gpu_availability()

    # Resolve and validate training dataset path
    train_path = Path(cfg.dataset.train_dataset_path).resolve()
    validate_file_exists(train_path, "Training dataset")

    # Load dataset using DatasetProcessor
    processor = DatasetProcessor(cfg.dataset.name)
    train_dataset = processor.load_prepared_dataset(train_path)

    # Initialize model and tokenizer
    use_flash_attention = cfg.training.get("use_flash_attention", True)
    model, tokenizer, _ = initialize_model_for_training(
        model_id=cfg.hf.model_id,
        use_flash_attention=use_flash_attention,
        max_seq_length=cfg.training.max_seq_length,
    )

    # Configure LoRA if applicable
    lora_config = ModelSetup.create_lora_config(
        lora_alpha=cfg.training.get("lora_alpha", 16),
        lora_dropout=cfg.training.get("lora_dropout", 0.05),
        lora_r=cfg.training.get("lora_r", 8),
    )

    # Print number of trainable parameters
    print_trainable_parameters(model)

    # Format dataset for training
    train_dataset = format_dataset_for_training(train_dataset, tokenizer, max_seq_length=2048)

    # Setup trainer
    trainer = ModelTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        peft_config=lora_config,
        output_dir=Path(cfg.training.output_dir).resolve(),
        max_seq_length=cfg.training.max_seq_length,
    )

    report_to = "wandb" if cfg.wandb.enable and cfg.wandb.api_key else None

    # Prepare training arguments
    training_args = trainer.create_training_arguments(
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        max_grad_norm=cfg.training.get("max_grad_norm", 1.0),
        warmup_ratio=cfg.training.get("warmup_ratio", 0.03),
        logging_steps=cfg.training.get("logging_steps", 10),
        push_to_hub=False,
        report_to=report_to,
    )

    # Start training
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)
    print(f"Output directory: {cfg.training.output_dir}")
    print(f"Number of epochs: {cfg.training.num_train_epochs}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Resume from checkpoint: {cfg.training.resume_from_checkpoint}")
    print("="*80 + "\n")

    trainer.train(
        training_args,
        resume_from_checkpoint=cfg.training.resume_from_checkpoint
    )

    # Save model and clean up
    print("\n" + "="*80)
    print("Saving model...")
    print("="*80)
    trainer.save_model()

    # Cleanup
    print("\n" + "="*80)
    print("Cleaning up...")
    print("="*80)
    trainer.cleanup()

    print("\n" + "="*80)
    print("Training completed successfully!")
    print("="*80)
    print(f"Model saved to: {cfg.training.output_dir}")
    print("\nNext steps:")
    print("1. Evaluate the model: python scripts/evaluate.py")
    print("2. Test inference: python scripts/inference.py --interactive")
    print("3. Merge and upload: python scripts/upload_to_hf.py")
    print("="*80)

if __name__ == "__main__":
    main()
