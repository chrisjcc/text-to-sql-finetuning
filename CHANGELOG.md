# Changelog

All notable changes and API updates for this project.

## [1.0.0] - 2025-01-XX

### ✨ Features

- **Production-ready modular codebase** with clean separation of concerns
- **Environment-based configuration** using python-dotenv
- **Comprehensive logging** with file and console outputs
- **Error handling and validation** throughout
- **WandB integration** for experiment tracking (optional)
- **Resume from checkpoint** support
- **Merge and upload scripts** for Hugging Face Hub deployment
- **Interactive inference mode** for testing

### 🔧 API Updates (Based on Latest TRL)

#### 1. Dataset Pre-formatting (CRITICAL)

**Old approach (deprecated):**
```python
# Pass dataset with 'messages' field directly
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,  # Has 'messages' field
    max_seq_length=2048,
    tokenizer=tokenizer,
    packing=True,
    dataset_kwargs={...}
)
```

**New approach (current):**
```python
# Pre-format dataset with chat template BEFORE training
def format_for_training(example):
    return {"text": tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )}

dataset = dataset.map(format_for_training, remove_columns=dataset.column_names)

# Then use simplified SFTTrainer API
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,  # Has 'text' field
    peft_config=peft_config,
)
```

**Why:** Latest TRL requires pre-formatted datasets. The `max_seq_length`, `tokenizer`, `packing`, and `dataset_kwargs` parameters are no longer needed when using pre-formatted datasets.

#### 2. Flash Attention Graceful Fallback

**Old approach:**
```python
# Hard-coded, fails if Flash Attention not available
attn_implementation = "flash_attention_2"
```

**New approach:**
```python
# Graceful fallback to SDPA
try:
    import flash_attn
    attn_implementation = "flash_attention_2"
    print("✓ Flash Attention 2 available")
except ImportError:
    attn_implementation = "sdpa"
    print("⚠ Flash Attention 2 not available, using SDPA")
```

**Why:** Not all systems have Flash Attention installed or compatible GPUs (requires compute capability >= 8.0).

#### 3. Tokenizer Max Length

**Added:**
```python
tokenizer.model_max_length = 2048  # Explicitly set
```

**Why:** Ensures consistent sequence length handling across training and inference.

#### 4. Resume from Checkpoint

**Added:**
```python
trainer.train(resume_from_checkpoint=True)
```

**Why:** Allows training to resume automatically from the latest checkpoint if interrupted.

#### 5. Model Merging and Upload

**Added:** Complete workflow for merging LoRA adapters and uploading to Hub
- Merge LoRA weights with base model
- Generate model cards automatically
- Upload to Hugging Face Hub with proper metadata

### 📦 Dependencies

#### Core Requirements
- `torch==2.4.0`
- `transformers==4.44.2`
- `trl==0.9.6` (latest API)
- `peft==0.12.0`
- `datasets==2.21.0`
- `accelerate==0.33.0`
- `bitsandbytes==0.43.3`
- `python-dotenv`

#### Optional
- `wandb` - Experiment tracking
- `flash-attn` - Flash Attention 2 (requires compatible GPU)

### 🔄 Migration Guide

If you have code using the old TRL API:

1. **Update dataset preparation:**
   - Add `format_dataset_for_training()` call before creating trainer
   - Remove `max_seq_length`, `tokenizer`, `packing`, `dataset_kwargs` from SFTTrainer

2. **Update model loading:**
   - Add Flash Attention fallback logic
   - Set `tokenizer.model_max_length` explicitly

3. **Update training:**
   - Add `resume_from_checkpoint=True` parameter
   - Consider adding WandB integration

4. **Add merge and upload:**
   - Use new `merge_and_upload.py` script
   - Set `HF_USERNAME` in `.env`

### 🐛 Bug Fixes

- Fixed Flash Attention crashes on incompatible hardware
- Fixed tokenizer padding warnings
- Fixed checkpoint resuming issues
- Fixed memory cleanup after training

### 📝 Documentation

- Comprehensive README with all features
- Detailed DEPLOYMENT guide
- CONTRIBUTING guidelines
- API migration guide
- Inline code documentation

### 🎯 Breaking Changes

- **SFTTrainer API:** Must pre-format datasets with chat template
- **Model loading:** Flash Attention no longer assumed available
- **Configuration:** All settings now via environment variables

### 🔜 Planned Features

- [ ] Support for more model architectures (Mistral, Phi, etc.)
- [ ] Additional evaluation metrics (BLEU, execution-based)
- [ ] Query validation and correction
- [ ] Multi-database dialect support
- [ ] Web UI for testing
- [ ] Docker Compose setup
- [ ] CI/CD pipelines

## Version History

- **v1.0.0** - Initial production release with latest TRL API
