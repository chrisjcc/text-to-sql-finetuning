"""
Model setup module for loading and configuring models for fine-tuning.
"""

import logging
from typing import Tuple

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from trl import setup_chat_format
from peft import LoraConfig, PeftConfig

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
        tokenizer: PreTrainedTokenizer
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Setup model and tokenizer for chat format.
        
        Args:
            model: Pre-trained model
            tokenizer: Pre-trained tokenizer
            
        Returns:
            Tuple of (configured model, configured tokenizer)
        """
        logger.info("Setting up chat format for model and tokenizer")
        try:
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
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )
    
    @staticmethod
    def load_trained_model(
        model_path: str,
        device_map: str = "auto"
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a trained PEFT model for inference.
        
        Args:
            model_path: Path to the trained model
            device_map: Device mapping strategy
            
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading trained model from {model_path}")
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                torch_dtype=torch.float16
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info("Trained model loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load trained model: {e}")
            raise


def initialize_model_for_training(
    model_id: str,
    use_flash_attention: bool = True,
    max_seq_length: int = 2048,
    lora_config: LoraConfig = None
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, LoraConfig]:
    """
    Convenience function to initialize model for training.
    
    Args:
        model_id: Hugging Face model ID
        use_flash_attention: Whether to attempt Flash Attention 2
        max_seq_length: Maximum sequence length for the model
        lora_config: Optional pre-configured LoRA config
        
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
    
    # Setup chat format
    model, tokenizer = setup.setup_for_chat(model, tokenizer)
    
    # Create LoRA config if not provided
    if lora_config is None:
        lora_config = setup.create_lora_config()
    
    return model, tokenizer, lora_config
