"""
Configuration module for Text-to-SQL fine-tuning project.
Loads environment variables and provides configuration classes.
"""
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    class hf:
        token = os.getenv("HF_TOKEN")
        model_id = os.getenv("HF_MODEL_ID")
        username = os.getenv("HF_USERNAME")

    class wandb:
        api_key = os.getenv("WANDB_API_KEY")
        project = os.getenv("WANDB_PROJECT")
        enabled = bool(api_key)  # WandB is enabled if API key is set

    class training:
        output_dir = Path(os.getenv("OUTPUT_DIR", "output"))
        num_train_epochs = int(os.getenv("NUM_TRAIN_EPOCHS", 3))
        per_device_train_batch_size = int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", 1))
        gradient_accumulation_steps = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", 8))
        learning_rate = float(os.getenv("LEARNING_RATE", 2e-4))
        max_seq_length = int(os.getenv("MAX_SEQ_LENGTH", 2048))

    class dataset:
        name = os.getenv("DATASET_NAME")
        train_samples = int(os.getenv("TRAIN_SAMPLES", 10000))
        test_samples = int(os.getenv("TEST_SAMPLES", 2500))
        train_path = Path(os.getenv("TRAIN_DATASET_PATH", "data/train_dataset.json"))
        test_path = Path(os.getenv("TEST_DATASET_PATH", "data/test_dataset.json"))

    class evaluation:
        num_eval_samples = int(os.getenv("NUM_EVAL_SAMPLES", 1000))

    @staticmethod
    def load():
        return Config
