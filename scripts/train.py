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
    get_hp_space_function,
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
    setup_chat_format = cfg.hf.get("setup_chat_format", True)
    force_chat_setup = cfg.hf.get("force_chat_setup", False)

    model, tokenizer, _ = initialize_model_for_training(
        model_id=cfg.hf.model_id,
        use_flash_attention=use_flash_attention,
        max_seq_length=cfg.training.max_seq_length,
        setup_chat_format=setup_chat_format,
        force_chat_setup=force_chat_setup,
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

    output_dir = Path(get_original_cwd()) / cfg.training.output_dir

    # Setup trainer
    trainer = ModelTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        peft_config=lora_config,
        output_dir=output_dir,
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
        push_to_hub=cfg.hf.upload.push_to_hub,
        report_to=report_to,
    )

    # Check if hyperparameter optimization is enabled
    hpo_enabled = cfg.training.get("hpo", {}).get("enabled", False)

    if hpo_enabled:
        # Hyperparameter optimization mode
        print("\n" + "="*80)
        print("Starting Hyperparameter Optimization...")
        print("="*80)
        hpo_config = cfg.training.hpo
        print(f"Backend: {hpo_config.backend}")
        print(f"Method: {hpo_config.method}")
        print(f"Number of trials: {hpo_config.n_trials}")
        print(f"Optimizing metric: {hpo_config.metric.name} ({hpo_config.metric.goal})")
        print(f"Output directory: {cfg.training.output_dir}")
        print(f"Training samples: {len(train_dataset)}")
        print("="*80 + "\n")

        # Get the appropriate hp_space function
        hp_space_fn = get_hp_space_function(hpo_config.backend)

        # Create a wrapper that passes the hpo_config to the hp_space function
        def hp_space_wrapper(trial):
            return hp_space_fn(trial, dict(hpo_config))

        # Create the trainer with training arguments
        sft_trainer = trainer.create_trainer(training_args)

        # Run hyperparameter search
        try:
            best_trial = sft_trainer.hyperparameter_search(
                hp_space=hp_space_wrapper,
                backend=hpo_config.backend,
                n_trials=hpo_config.n_trials,
                direction="minimize" if hpo_config.metric.goal == "minimize" else "maximize",
            )

            print("\n" + "="*80)
            print("Hyperparameter optimization completed!")
            print("="*80)
            print(f"Best trial: {best_trial}")
            print("="*80 + "\n")

        except Exception as e:
            print(f"\nHyperparameter optimization failed: {e}")
            raise

    else:
        # Normal training mode
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

    # Save model and clean up (only in normal training mode)
    if not hpo_enabled:
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
    else:
        print("\n" + "="*80)
        print("Hyperparameter optimization completed!")
        print("="*80)
        print("Best hyperparameters have been identified.")
        print("To train with the best hyperparameters:")
        print("1. Check your Weights & Biases dashboard for the best trial")
        print("2. Update config/training/training.yaml with the best hyperparameters")
        print("3. Run training with hpo.enabled=false")
        print("="*80)

if __name__ == "__main__":
    main()
