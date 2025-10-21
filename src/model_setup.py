"""
Model setup module for loading and configuring models for fine-tuning.
"""

import os
import json
import logging
from typing import Optional, Tuple

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from trl import setup_chat_format
from peft import LoraConfig, PeftConfig, PeftModel, TaskType

logger = logging.getLogger(__name__)


class ModelSetup:
    """Handles model and tokenizer loading and configuration."""

    @staticmethod
    def create_bnb_config() -> BitsAndBytesConfig:
        """
        Create BitsAndBytes configuration for 4-bit quantization.

        Returns:
            BitsAndBytesConfig for 4-bit quantization with double quantization
        """
        logger.info("Creating BitsAndBytes configuration for 4-bit quantization")
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    @staticmethod
    def load_model_and_tokenizer(
        model_id: str,
        use_flash_attention: bool = True,
        max_seq_length: int = 2048
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load model and tokenizer with quantization and Flash Attention.

        Args:
            model_id: Hugging Face model ID
            use_flash_attention: Whether to attempt Flash Attention 2
            max_seq_length: Maximum sequence length for the model

        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model: {model_id}")

        # Create quantization config
        bnb_config = ModelSetup.create_bnb_config()

        # Determine attention implementation with graceful fallback
        if use_flash_attention:
            try:
                import flash_attn
                attn_implementation = "flash_attention_2"
                logger.info("✓ Flash Attention 2 available")
            except ImportError:
                attn_implementation = "sdpa"
                logger.warning("⚠ Flash Attention 2 not available, using SDPA")
        else:
            attn_implementation = "sdpa"
            logger.info("Using SDPA attention (Flash Attention disabled)")

        try:
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config
            )
            logger.info(f"Model loaded successfully with {attn_implementation} attention")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.padding_side = 'right'  # Prevent warnings
            tokenizer.model_max_length = max_seq_length  # Set max sequence length
            logger.info(f"Tokenizer loaded successfully (max_length={max_seq_length})")

            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load model or tokenizer: {e}")
            raise

    @staticmethod
    def setup_for_chat(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        force_setup: bool = False
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Setup model and tokenizer for chat format.

        Args:
            model: Pre-trained model
            tokenizer: Pre-trained tokenizer
            force_setup: If True, force setup even if chat template exists (will set to None first)

        Returns:
            Tuple of (configured model, configured tokenizer)
        """
        logger.info("Setting up chat format for model and tokenizer")

        # Check if tokenizer already has a chat template
        has_chat_template = hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None

        if has_chat_template:
            if force_setup:
                logger.warning("Chat template already present, but force_setup=True. Setting to None and re-applying.")
                tokenizer.chat_template = None
            else:
                logger.info("Chat template already present in tokenizer, skipping setup_chat_format()")
                logger.info("This is expected for Instruct models like Llama-3.1-8B-Instruct")
                return model, tokenizer

        try:
            logger.info("Applying chat format setup...")
            model, tokenizer = setup_chat_format(model, tokenizer)
            logger.info("Chat format setup completed")
            return model, tokenizer
        except Exception as e:
            logger.error(f"Failed to setup chat format: {e}")
            raise

    @staticmethod
    def create_lora_config(
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        lora_r: int = 256,
        target_modules: str = "all-linear"
    ) -> LoraConfig:
        """
        Create LoRA configuration for parameter-efficient fine-tuning.

        Args:
            lora_alpha: LoRA alpha parameter
            lora_dropout: Dropout probability for LoRA layers
            lora_r: LoRA rank
            target_modules: Target modules for LoRA adaptation

        Returns:
            LoraConfig object
        """
        logger.info(f"Creating LoRA config: alpha={lora_alpha}, r={lora_r}, dropout={lora_dropout}")
        return LoraConfig(
            lora_alpha=lora_alpha,  # scaling factor
            lora_dropout=lora_dropout,  # dropout of LoRA layers
            r=lora_r,  # dimension of the smaller matrices
            bias="none",
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,  # type of task to train on
            inference_mode=False,  # set to False for training
        )

    @staticmethod
    def load_trained_model(
        model_path: str,
        adapter_path: Optional[str] = None,
        device_map: str = "auto",
        setup_chat_format_flag: bool = True,
        force_chat_setup: bool = False,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a trained PEFT model (adapter + tokenizer) for inference.

        Args:
            model_path: Base model path or HuggingFace ID
            adapter_path: Optional path to adapter (local or HuggingFace Hub ID). If None, loads base model only.
            device_map: Device mapping strategy
            setup_chat_format_flag: Whether to apply chat format setup (default: True)
            force_chat_setup: If True, force chat setup even if template exists (default: False)

        Returns:
            Tuple of (model, tokenizer)
        """
        import json
        from pathlib import Path
        from trl import setup_chat_format  # Import here
        from huggingface_hub import hf_hub_download, list_repo_files

        # Case 1: No adapter - just load base model
        if adapter_path is None:
            logger.info(f"Loading base model (no adapter): {model_path}")

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                trust_remote_code=True,
            )
            logger.info(f"✓ Base model loaded (vocab size: {model.config.vocab_size})")

            tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")

            # Handle chat format setup based on configuration
            if not setup_chat_format_flag:
                logger.info("⊗ Skipping chat format setup (setup_chat_format=false in config)")
                logger.info(f"  Tokenizer vocab: {len(tokenizer)}, Model vocab: {model.config.vocab_size}")
                has_chat_template = tokenizer.chat_template is not None
                has_chat_tokens = all(token in tokenizer.get_vocab() for token in ["<|im_start|>", "<|im_end|>"])
                logger.info(f"  Has chat template: {has_chat_template}, Has chat tokens: {has_chat_tokens}")
            else:
                # Check if we need to apply chat format
                has_chat_template = tokenizer.chat_template is not None
                vocab_already_extended = len(tokenizer) > model.config.vocab_size

                if has_chat_template and vocab_already_extended:
                    logger.info("⊗ Skipping chat format setup (already configured)")
                    logger.info(f"  Tokenizer vocab: {len(tokenizer)}, Model vocab: {model.config.vocab_size}")
                else:
                    logger.info("⚠️  Applying chat format setup...")
                    logger.info(f"  has_chat_template: {has_chat_template}, vocab_already_extended: {vocab_already_extended}")
                    model, tokenizer = setup_chat_format(model, tokenizer)
                    logger.info("✓ Chat format applied")

            # Set padding token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = tokenizer.eos_token_id
                logger.info("✓ Padding token configured")

            logger.info("✅ Base model loaded successfully")
            return model, tokenizer

        # Case 2: Load adapter model
        logger.info(f"Loading trained model with adapter from {adapter_path}")

        # Determine if this is a local path or HuggingFace Hub ID
        is_hub_model = False
        local_path = Path(adapter_path)
        
        if local_path.exists():
            # It's a local path
            logger.info(f"Loading from local path: {adapter_path}")
            adapter_config_path = local_path / "adapter_config.json"
        elif "/" in adapter_path and not adapter_path.startswith("./") and not adapter_path.startswith("../"):
            # Likely a HuggingFace Hub ID (e.g., "username/model-name")
            logger.info(f"Detected HuggingFace Hub ID: {adapter_path}")
            try:
                # Verify the repo exists by checking for adapter_config.json
                list_repo_files(adapter_path)
                is_hub_model = True
                logger.info(f"✓ Model found on HuggingFace Hub")
                # For Hub models, we'll download the config file to read it
                adapter_config_path = hf_hub_download(
                    repo_id=adapter_path,
                    filename="adapter_config.json"
                )
            except Exception as e:
                raise FileNotFoundError(
                    f"Could not access HuggingFace Hub model: {adapter_path}. "
                    f"Make sure the model exists and you have proper authentication. "
                    f"Error: {e}"
                )
        else:
            raise FileNotFoundError(
                f"Model path does not exist and is not a valid HuggingFace Hub ID: {adapter_path}"
            )

        try:
            # Step 1: Get base model name from adapter config
            if not Path(adapter_config_path).exists():
                raise FileNotFoundError(f"adapter_config.json not found")

            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)

            base_model_name = adapter_config.get('base_model_name_or_path')
            if not base_model_name:
                raise ValueError("base_model_name_or_path not found in adapter_config.json")

            logger.info(f"Loading base model: {base_model_name}")

            # Step 2: Load base model
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                trust_remote_code=True,
            )
            logger.info(f"✓ Base model loaded (vocab size: {model.config.vocab_size})")

            # Step 3: Load tokenizer from checkpoint (should include special tokens if saved during training)
            logger.info("Loading tokenizer from checkpoint...")
            # Use the original adapter_path (string) for from_pretrained - works for both local and Hub
            tokenizer = AutoTokenizer.from_pretrained(adapter_path)
            tokenizer_vocab_size = len(tokenizer)
            logger.info(f"✓ Tokenizer loaded (vocab size: {tokenizer_vocab_size})")

            # Step 4: Check if special tokens are already present
            # If tokenizer was uploaded with special tokens from training, skip setup_chat_format
            # FIXED: Better detection of whether chat format has already been applied
            # Check for the presence of special tokens that setup_chat_format adds
            has_chat_template = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None

            # More robust check: Look for specific special tokens added by setup_chat_format
            # These tokens are typically added: <|im_start|> and <|im_end|>
            special_token_ids = set(tokenizer.all_special_ids) if hasattr(tokenizer, 'all_special_ids') else set()
            base_special_tokens = {'<|begin_of_text|>', '<|end_of_text|>', '<|eot_id|>', '<|start_header_id|>', '<|end_header_id|>'}
            chat_special_tokens = {'<|im_start|>', '<|im_end|>'}

            # Check if chat tokens are in the vocabulary
            has_chat_tokens = any(token in tokenizer.get_vocab() for token in chat_special_tokens)
            vocab_already_extended = tokenizer_vocab_size > model.config.vocab_size

            # Check if we should apply chat format setup based on configuration
            if not setup_chat_format_flag:
                # User explicitly disabled chat format setup via configuration
                logger.info("⊗ Skipping chat format setup (setup_chat_format=false in config)")
                logger.info(f"  Tokenizer vocab: {tokenizer_vocab_size}, Model vocab: {model.config.vocab_size}")
                logger.info(f"  Has chat template: {has_chat_template}, Has chat tokens: {has_chat_tokens}")

                # Resize model embeddings to match tokenizer if needed
                if tokenizer_vocab_size != model.config.vocab_size:
                    logger.info(f"Resizing model embeddings from {model.config.vocab_size} to {tokenizer_vocab_size}")
                    model.resize_token_embeddings(tokenizer_vocab_size, mean_resizing=False)
                    logger.info("✓ Model embeddings resized (preserving adapter's trained embeddings)")
            elif has_chat_template and vocab_already_extended:
                # Tokenizer already has chat format configured from training and vocab was extended
                logger.info("✓ Chat format already configured in tokenizer (skipping setup_chat_format)")
                logger.info(f"  Tokenizer vocab: {tokenizer_vocab_size}, Model vocab: {model.config.vocab_size}")
                logger.info(f"  Vocab size difference: {tokenizer_vocab_size - model.config.vocab_size} special tokens")
                logger.info(f"  Has chat template: {has_chat_template}, Has chat tokens: {has_chat_tokens}")

                # Resize model embeddings to match tokenizer if needed
                # This ensures the base model has the correct embedding size before loading LoRA
                # CRITICAL: Use mean_resizing=False to avoid reinitializing embeddings
                # The adapter already has the correct embeddings trained
                if tokenizer_vocab_size != model.config.vocab_size:
                    logger.info(f"Resizing model embeddings from {model.config.vocab_size} to {tokenizer_vocab_size}")
                    model.resize_token_embeddings(tokenizer_vocab_size, mean_resizing=False)
                    logger.info("✓ Model embeddings resized (preserving adapter's trained embeddings)")
            else:
                # Apply setup_chat_format for backwards compatibility or if special tokens not present
                logger.info("⚠️  Applying chat format setup (special tokens not found in tokenizer)...")
                logger.info(f"  has_chat_template: {has_chat_template}, vocab_already_extended: {vocab_already_extended}")
                logger.info(f"  has_chat_tokens: {has_chat_tokens}")
                logger.warning("This may cause embedding mismatch with the trained adapter!")

                # Handle force_chat_setup flag
                if force_chat_setup and has_chat_template:
                    logger.warning("Force chat setup enabled - setting existing chat template to None")
                    tokenizer.chat_template = None

                model, tokenizer = setup_chat_format(model, tokenizer)
                new_vocab_size = len(tokenizer)
                logger.info(f"✓ Chat format applied (new vocab size: {new_vocab_size})")

                if new_vocab_size != tokenizer_vocab_size:
                    logger.warning(f"  Added {new_vocab_size - tokenizer_vocab_size} NEW special tokens - this will likely cause 0% accuracy!")

            # Step 5: Load PEFT adapter
            logger.info(f"Loading LoRA adapter from {adapter_path}...")
            # Use the original adapter_path (string) - works for both local and Hub
            model = PeftModel.from_pretrained(
                model,
                adapter_path,  # Use string, not Path object
                torch_dtype=torch.bfloat16,
            )
            logger.info("✓ Adapter loaded successfully")

            # Step 5.5: Verify the adapter has embeddings for extended vocabulary
            # The adapter should have saved the new token embeddings if vocab was extended during training
            if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
                base_model = model.base_model.model
                adapter_vocab_size = base_model.get_input_embeddings().weight.shape[0]
                logger.info(f"Adapter model embedding size: {adapter_vocab_size}")
                logger.info(f"Tokenizer vocabulary size: {len(tokenizer)}")

                if adapter_vocab_size != len(tokenizer):
                    logger.warning(f"⚠️  Mismatch: Adapter embeddings ({adapter_vocab_size}) != Tokenizer vocab ({len(tokenizer)})")
                    logger.warning("This indicates the extended embeddings were not saved with the adapter.")
                else:
                    logger.info("✓ Adapter embeddings match tokenizer vocabulary")

            # Step 6: Set padding token if needed
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = tokenizer.eos_token_id
                logger.info("✓ Padding token configured")

            logger.info("✅ Model loaded successfully for evaluation")
            return model, tokenizer

        except Exception as e:
            logger.error(f"❌ Failed to load trained model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise


def initialize_model_for_training(
    model_id: str,
    use_flash_attention: bool = True,
    max_seq_length: int = 2048,
    lora_config: LoraConfig = None,
    setup_chat_format: bool = True,
    force_chat_setup: bool = False
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, LoraConfig]:
    """
    Convenience function to initialize model for training.

    Args:
        model_id: Hugging Face model ID
        use_flash_attention: Whether to attempt Flash Attention 2
        max_seq_length: Maximum sequence length for the model
        lora_config: Optional pre-configured LoRA config
        setup_chat_format: Whether to setup chat format (default: True)
                          Set to False for models that already have chat templates (e.g., Instruct models)
        force_chat_setup: If True, force chat setup even if template exists (default: False)

    Returns:
        Tuple of (model, tokenizer, lora_config)
    """
    setup = ModelSetup()

    # Load model and tokenizer
    model, tokenizer = setup.load_model_and_tokenizer(
        model_id,
        use_flash_attention,
        max_seq_length
    )

    # Setup chat format (conditionally)
    if setup_chat_format:
        model, tokenizer = setup.setup_for_chat(model, tokenizer, force_setup=force_chat_setup)
    else:
        logger.info("Skipping chat format setup (setup_chat_format=False)")

    # Create LoRA config if not provided
    if lora_config is None:
        lora_config = setup.create_lora_config()

    return model, tokenizer, lora_config
