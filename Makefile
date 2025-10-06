.PHONY: help install install-flash install-wandb setup prepare-data train evaluate merge-upload inference clean

help:
	@echo "Available commands:"
	@echo "  make install         - Install Python dependencies"
	@echo "  make install-flash   - Install Flash Attention (requires CUDA)"
	@echo "  make install-wandb   - Install Weights & Biases for tracking"
	@echo "  make setup          - Setup project (install + create directories)"
	@echo "  make prepare-data   - Prepare and save datasets"
	@echo "  make train          - Train the model"
	@echo "  make evaluate       - Evaluate the trained model"
	@echo "  make merge-upload   - Merge LoRA and upload to HuggingFace"
	@echo "  make inference      - Run interactive inference"
	@echo "  make clean          - Clean up generated files"

install:
	pip install -r requirements.txt

install-flash:
	pip install ninja packaging
	MAX_JOBS=4 pip install flash-attn --no-build-isolation

install-wandb:
	pip install wandb

setup: install
	mkdir -p data logs config
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file. Please edit it with your credentials."; \
	fi

# uses module-style execution
prepare-data:
	@mkdir -p data
	@echo "Running data preparation..."
	@python -m scripts.prepare_data || { echo "⚠ Data preparation failed. Check HF_TOKEN and dataset access."; exit 1; }
	@echo "✓ Data preparation completed successfully"

train:
	@mkdir -p logs
	@python -m scripts.train || { echo "⚠ Training failed."; exit 1; }

evaluate:
	python -m scripts.evaluate

merge-upload:
	python -m scripts.merge_and_upload

inference:
	python -m scripts.inference --interactive

clean:
	rm -rf __pycache__ src/__pycache__ config/__pycache__ scripts/__pycache__
	rm -rf logs/*.log
	rm -rf data/*.json
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
