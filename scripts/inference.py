"""
Simple inference script for testing the fine-tuned model.

Usage:
    python scripts/inference.py --question "Show all customers" --schema "CREATE TABLE..."
    python scripts/inference.py --interactive
"""

import sys
import argparse
from pathlib import Path

import torch
from transformers import pipeline

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model_setup import ModelSetup
from src.utils import setup_logging, check_gpu_availability
from config.config import Config


def load_model_pipeline(model_path: str):
    """
    Load the trained model and create a pipeline.
    
    Args:
        model_path: Path to the trained model
        
    Returns:
        Hugging Face pipeline
    """
    print(f"Loading model from {model_path}...")
    model, tokenizer = ModelSetup.load_trained_model(model_path)
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    
    print("Model loaded successfully!")
    return pipe


def generate_sql(
    pipe,
    question: str,
    schema: str,
    max_tokens: int = 256,
    temperature: float = 0.1
) -> str:
    """
    Generate SQL query from natural language question.
    
    Args:
        pipe: Hugging Face pipeline
        question: Natural language question
        schema: Database schema
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Generated SQL query
    """
    system_message = f"""You are an text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA.
SCHEMA:
{schema}"""
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": question}
    ]
    
    prompt = pipe.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    outputs = pipe(
        prompt,
        max_new_tokens=max_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_k=50,
        top_p=0.95,
        eos_token_id=pipe.tokenizer.eos_token_id,
        pad_token_id=pipe.tokenizer.pad_token_id
    )
    
    generated_text = outputs[0]['generated_text'][len(prompt):].strip()
    return generated_text


def interactive_mode(pipe):
    """Run interactive inference mode."""
    print("\n" + "="*80)
    print("Interactive Text-to-SQL Mode")
    print("="*80)
    print("Commands:")
    print("  /schema <schema> - Set the database schema")
    print("  /clear          - Clear the current schema")
    print("  /exit           - Exit interactive mode")
    print("="*80 + "\n")
    
    current_schema = None
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
                
            if user_input == "/exit":
                print("Goodbye!")
                break
                
            if user_input == "/clear":
                current_schema = None
                print("Schema cleared.")
                continue
                
            if user_input.startswith("/schema "):
                current_schema = user_input[8:].strip()
                print("Schema updated.")
                continue
            
            if current_schema is None:
                print("Error: Please set a schema first using /schema")
                continue
            
            # Generate SQL
            print("\nGenerating SQL...")
            sql = generate_sql(pipe, user_input, current_schema)
            
            print("\nGenerated SQL:")
            print("-" * 80)
            print(sql)
            print("-" * 80)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def single_query_mode(pipe, question: str, schema: str, temperature: float):
    """Run single query inference."""
    print("\n" + "="*80)
    print("Single Query Mode")
    print("="*80)
    print(f"\nQuestion: {question}")
    print(f"\nSchema:\n{schema}")
    print("\nGenerating SQL...")
    
    sql = generate_sql(pipe, question, schema, temperature=temperature)
    
    print("\nGenerated SQL:")
    print("-" * 80)
    print(sql)
    print("-" * 80)


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description="Generate SQL queries from natural language"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to the trained model (default: from config)"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Natural language question"
    )
    parser.add_argument(
        "--schema",
        type=str,
        help="Database schema"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Check GPU
    check_gpu_availability()
    
    # Load configuration
    config = Config.load()
    model_path = args.model_path or config.evaluation.model_path
    
    # Load model
    pipe = load_model_pipeline(model_path)
    
    # Run inference
    if args.interactive:
        interactive_mode(pipe)
    elif args.question and args.schema:
        single_query_mode(pipe, args.question, args.schema, args.temperature)
    else:
        print("Error: Either use --interactive or provide both --question and --schema")
        parser.print_help()


if __name__ == "__main__":
    main()
