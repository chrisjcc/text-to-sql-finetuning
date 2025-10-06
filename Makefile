.PHONY: help install install-flash setup prepare-data train evaluate clean

help:
	@echo "Available commands:"
	@echo "  make install         - Install Python dependencies"
	@echo "  make install-flash   - Install Flash Attention (requires CUDA)"
	@echo "  make setup          - Setup project (install + create directories)"
	@echo "  make prepare-data   - Prepare and save datasets"
	@echo "  make train          - Train the model"
	@echo "  make evaluate       - Evaluate the trained model"
	@echo "  make clean          - Clean up generated files"

install:
	pip install -r requirements.txt

install-flash:
	pip install ninja packaging
	MAX_JOBS=4 pip install flash-attn --no-build-isolation

setup: install
	mkdir -p data logs config
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file. Please edit it with your credentials."; \
	fi

prepare-data:
	python scripts/prepare_data.py

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

clean:
	rm -rf __pycache__ src/__pycache__ config/__pycache__ scripts/__pycache__
	rm -rf logs/*.log
	rm -rf data/*.json
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
