import os
from dotenv import load_dotenv
import hydra
from omegaconf import DictConfig
from pathlib import Path

load_dotenv()  # load secrets

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # cfg.training.learning_rate, cfg.dataset.train_path, etc.
    print("HF Token:", os.getenv("HF_TOKEN")[:4] + "…")
    print("Learning rate:", cfg.training.learning_rate)
    print("Dataset path:", Path(cfg.dataset.train_dataset_path))

if __name__ == "__main__":
    main()
