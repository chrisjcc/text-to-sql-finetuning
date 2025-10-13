.PHONY: help install install-flash install-wandb setup prepare-data train train-accelerate train-basic evaluate  upload-to-hf inference clean

help:
	@echo "Available commands:"
	@echo "  make install           - Install Python dependencies"
	@echo "  make install-flash     - Install Flash Attention (requires CUDA)"
	@echo "  make setup             - Setup project (install + create directories)"
	@echo "  make prepare-data      - Prepare and save datasets"
	@echo "  make train             - Train (auto-detect accelerate/python)"
	@echo "  make train-accelerate  - Train with accelerate (force)"
	@echo "  make train-basic       - Train with python (force)"
	@echo "  make evaluate          - Evaluate the trained model"
	@echo "  make upload-to-hf      - Merge LoRA and upload to HuggingFace"
	@echo "  make inference         - Run interactive inference"
	@echo "  make clean             - Clean up generated files"

install:
	pip install -r requirements.txt

install-flash:
	pip install ninja packaging
	MAX_JOBS=4 pip install flash-attn --no-build-isolation

setup:
	@echo "🔧 Creating project directories..."
	mkdir -p data logs config
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file. Please edit it with your credentials."; \
	fi

prepare-data:
	@mkdir -p data
	@echo "Running data preparation..."
	@python -m scripts.prepare_data || { echo "⚠ Data preparation failed. Check HF_TOKEN and dataset access."; exit 1; }
	@echo "✓ Data preparation completed successfully"

# Auto-detect and use best training method
train:
	@mkdir -p logs
	@if command -v accelerate >/dev/null 2>&1; then \
		$(MAKE) train-accelerate; \
	else \
		$(MAKE) train-basic; \
	fi

# Force use of accelerate
train-accelerate:
	@echo "🚀 Training with accelerate launch..."
	@accelerate launch -m scripts.train || { echo "⚠ Training with accelerate failed."; exit 1; }

# Force use of basic python
train-basic:
	@echo "🐍 Training with python -m..."
	@python -m scripts.train || { echo "⚠ Training failed."; exit 1; }

# Designed for single-GPU
evaluate:
	@echo "📊 Running evaluation..."
	@python -m scripts.evaluate || { echo "⚠ Evaluation failed."; exit 1; }

 upload-to-hf:
	python -m scripts.upload_to_hf

inference:
	python -m scripts.inference --interactive

clean:
	rm -rf __pycache__ src/__pycache__ config/__pycache__ scripts/__pycache__
	rm -rf logs/*.log
	rm -rf data/*.json
	rm -rf wandb/*.log
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
