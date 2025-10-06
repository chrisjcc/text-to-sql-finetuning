"""
Script to fine-tune a language model for text-to-SQL generation.

Usage:
    python scripts/train.py
    python scripts/train.py --no-resume  # Start fresh without resuming
"""

import sys
import argparse
from pathlib import Path

# Add src to path
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
from config.config import Config


def setup_wandb(config):
    """Setup Weights & Biases tracking if enabled."""
    if config.wandb.enabled:
        try:
            import wandb
            wandb.login(key=config.wandb.api_key)
            wandb.init(project=config.wandb.project)
            print(f"✓ Weights & Biases tracking enabled (project: {config.wandb.project})")
        except ImportError:
            print("⚠ wandb not installed. Install with: pip install wandb")
        except Exception as e:
            print(f"⚠ Failed to setup WandB: {e}")
    else:
        print("WandB tracking disabled (no API key found)")


def main():
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train text-to-SQL model")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh without resuming from checkpoint"
    )
    parser.add_argument(
        "--no-flash-attention",
        action="store_true",
        help="Disable Flash Attention 2 (use SDPA instead)"
    )
    args = parser.parse_args()

    # Setup logging
    setup_logging(log_file=Path("logs/training.log"))

    # Load configuration
    config = Config.load()

    # Authenticate with Hugging Face
    authenticate_huggingface(config.hf.token)

    # Setup WandB if configured
    setup_wandb(config)

    # Check GPU availability
    check_gpu_availability()

    # Validate training dataset exists
    validate_file_exists(config.dataset.train_path, "Training dataset")

    # Load training dataset
    print("\n" + "="*80)
    print("Loading training dataset...")
    print("="*80)
    processor = DatasetProcessor(config.dataset.name)
    train_dataset = processor.load_prepared_dataset(config.dataset.train_path)

    # Initialize model for training
    print("\n" + "="*80)
    print("Initializing model and tokenizer...")
    print("="*80)

    use_flash_attention = not args.no_flash_attention

    model, tokenizer, lora_config = initialize_model_for_training(
        model_id=config.hf.model_id,
        use_flash_attention=use_flash_attention,
        max_seq_length=config.training.max_seq_length,
    )

    # Update LoRA config with training config values
    lora_config = ModelSetup.create_lora_config(
        lora_alpha=config.training.lora_alpha,
        lora_dropout=config.training.lora_dropout,
        lora_r=config.training.lora_r,
    )

    # Print trainable parameters
    print_trainable_parameters(model)

    # CRITICAL: Format dataset with chat template BEFORE training
    print("\n" + "="*80)
    print("Formatting dataset with chat template...")
    print("="*80)
    train_dataset = format_dataset_for_training(train_dataset, tokenizer)

    # Create trainer
    print("\n" + "="*80)
    print("Setting up trainer...")
    print("="*80)

    trainer = ModelTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        peft_config=lora_config,
        output_dir=config.training.output_dir,
    )

    # Create training arguments
    training_args = trainer.create_training_arguments(
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        max_grad_norm=config.training.max_grad_norm,
        warmup_ratio=config.training.warmup_ratio,
        logging_steps=config.training.logging_steps,
        push_to_hub=False,  # Set to True if you want to push to Hub
    )

    # Start training
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)
    print(f"Output directory: {config.training.output_dir}")
    print(f"Number of epochs: {config.training.num_train_epochs}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Resume from checkpoint: {not args.no_resume}")
    print("="*80 + "\n")

    resume = not args.no_resume
    trainer.train(training_args, resume_from_checkpoint=resume)

    # Save model
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
    print(f"Model saved to: {config.training.output_dir}")
    print("\nNext steps:")
    print("1. Evaluate the model: python scripts/evaluate.py")
    print("2. Test inference: python scripts/inference.py --interactive")
    print("3. Merge and upload: python scripts/merge_and_upload.py")
    print("="*80)


if __name__ == "__main__":
    main()
