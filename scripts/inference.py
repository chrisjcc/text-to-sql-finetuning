"""
Simple inference script for testing the fine-tuned model.

Usage:
    Interactive mode:
        python -m scripts.inference \
          --adapter_path chrisjcc/Meta-Llama-3.1-8B-text2sql-adapter \
          --interactive
    Single query:
        python -m scripts.inference \
          --adapter_path chrisjcc/Meta-Llama-3.1-8B-text2sql-adapter \
          --question "List all employees hired after 2020." \
          --context "CREATE TABLE employees(id INT, name TEXT, hire_date DATE);"
    Batch inference:
        python -m scripts.inference \
          --adapter_path chrisjcc/Meta-Llama-3.1-8B-text2sql-adapter \
          --batch_file data/test.jsonl \
          --output_file results.jsonl
"""

import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

from config.config import Config
from src.utils import setup_logging, check_gpu_availability

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


def generate_sql(model, tokenizer, question, context, max_new_tokens=512):
    """
    Generate an SQL query from a natural-language question and schema context.

    Args:
        model (transformers.PreTrainedModel): The language model used for inference.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer corresponding to the model.
        question (str): The natural language question to translate into SQL.
        context (str): The database schema or table context (e.g., CREATE TABLE statements).
        max_new_tokens (int, optional): The maximum number of tokens to generate. Defaults to 512.

    Returns:
        str: The generated SQL query as a string.
    """
    prompt = f"### Context:\n{context}\n\n### Question:\n{question}\n\n### SQL:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # Deterministic SQL generation, uses greedy decoding, which is usually preferred
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def interactive_mode(model, tokenizer):
    """
    Run an interactive REPL-style interface for text-to-SQL inference.

    Allows users to iteratively provide natural-language questions and optional
    schema contexts (e.g., CREATE TABLE statements). Supports basic commands
    to manage the current schema context.

    Commands:
        /schema <schema> - Set or update the current schema context
        /clear            - Clear the current schema
        /exit             - Exit interactive mode

    Args:
        model: The language model used for SQL generation.
        tokenizer: The tokenizer associated with the model.
    """
    print("\n" + "=" * 80)
    print("Interactive Text-to-SQL Mode")
    print("=" * 80)
    print("Commands:")
    print("  /schema <schema> - Set or update the database schema")
    print("  /clear           - Clear the current schema")
    print("  /exit            - Exit interactive mode")
    print("=" * 80 + "\n")

    context = ""  # persist schema between questions

    while True:
        try:
            question = input("\nNatural language question: ").strip()
            if not question:
                continue

            # Command handling
            if question.lower() in {"/exit", "exit", "quit"}:
                print("Goodbye!")
                break
            elif question.startswith("/schema "):
                context = question[len("/schema "):].strip()
                print("Schema updated.")
                continue
            elif question.startswith("/clear"):
                context = ""
                print("Schema cleared.")
                continue

            # Ask for schema if none stored
            if not context:
                context = input("Schema / context (CREATE TABLE...): ").strip()
                if not context:
                    print("Warning: empty context. Proceeding without schema context.")

            # Generate SQL
            sql = generate_sql(model, tokenizer, question, context)

            print("\nGenerated SQL:")
            print("-" * 80)
            print(sql)
            print("-" * 80)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")

def single_query_mode(model, tokenizer, question, context):
    """
    Run inference for a single natural-language question and its schema context.

    This mode generates an SQL query for a single input pair and prints it
    to the console. It is intended for quick, non-interactive testing or
    script-based use.

    Args:
        model: The language model used for SQL generation.
        tokenizer: The tokenizer associated with the model.
        question (str): The natural-language question.
        context (str): The database schema or contextual information
            (e.g., CREATE TABLE statements).
    """
    print("\n" + "=" * 80)
    print("Single Query Mode")
    print("=" * 80)
    print(f"\nQuestion:\n{question}")
    print(f"\nContext:\n{context if context else '(none)'}")
    print("\nGenerating SQL...")

    try:
        sql = generate_sql(model, tokenizer, question, context)
        print("\nGenerated SQL:")
        print("-" * 80)
        print(sql)
        print("-" * 80)
    except Exception as e:
        print(f"\nError during generation: {e}")

def batch_query_mode(model, tokenizer, batch_file, output_file="inference_outputs.jsonl"):
    """
    Run batched text-to-SQL inference over a JSONL dataset file.

    Each line in the input file must be a valid JSON object containing:
        {
            "question": "<natural-language question>",
            "context": "<database schema or CREATE TABLE statements>",
            "query": "<gold SQL query (optional)>"
        }

    The function will generate SQL predictions for each example and write
    results to an output JSONL file. Each output line includes:
        - question
        - context
        - pred_sql (model-generated SQL)
        - gold_sql (if available)

    Args:
        model: The language model used for SQL generation.
        tokenizer: The tokenizer associated with the model.
        batch_file (str): Path to the input JSONL dataset file.
        output_file (str): Path to save inference results. Defaults to 'inference_outputs.jsonl'.
    """
    results = []

    try:
        with open(batch_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Could not find batch file '{batch_file}'.")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSONL file '{batch_file}' — {e}")
        return

    for example in tqdm(lines, desc="Running batch inference"):
        question = example.get("question")
        context = example.get("context", "")
        gold_sql = example.get("query")

        if not question:
            print("⚠️  Skipping example with missing 'question' field.")
            continue

        try:
            pred_sql = generate_sql(model, tokenizer, question, context)
        except Exception as e:
            print(f"Error generating SQL for question '{question[:50]}...': {e}")
            pred_sql = None

        results.append({
            "question": question,
            "context": context,
            "pred_sql": pred_sql,
            "gold_sql": gold_sql,
        })

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n✅ Batch inference completed. Results saved to '{output_file}'.")
    except Exception as e:
        print(f"Error: Failed to write output file '{output_file}' — {e}")

def load_model_and_tokenizer(base_model: str, adapter_path: str | None = None):
    """
    Load a language model and tokenizer, optionally merging a PEFT (LoRA/QLoRA) adapter.

    If an adapter path is provided, the function will:
        1. Load the PEFT configuration to determine the correct base model.
        2. Load the corresponding base model and tokenizer.
        3. Merge the adapter weights dynamically for inference.

    If no adapter is provided, only the base model is loaded.

    Args:
        base_model (str): Name or local path of the base pretrained model
            (e.g., "meta-llama/Meta-Llama-3-8B").
        adapter_path (Optional[str]): Hugging Face Hub ID or local path of the
            trained PEFT adapter (e.g., "chrisjcc/Meta-Llama-3.1-8B-text2sql-adapter").

    Returns:
        Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
            The loaded model and tokenizer, ready for inference.
    """
    try:
        if adapter_path:
            print(f"🔹 Loading PEFT adapter from '{adapter_path}'...")
            # Load PEFT config to locate base model
            peft_config = PeftConfig.from_pretrained(adapter_path)
            base_name = peft_config.base_model_name_or_path

            tokenizer = AutoTokenizer.from_pretrained(base_name)
            base = AutoModelForCausalLM.from_pretrained(base_name, device_map="auto")
            model = PeftModel.from_pretrained(base, adapter_path)

            print(f"✅ Adapter successfully loaded on top of '{base_name}'")
        else:
            print(f"🔹 Loading base model '{base_model}' (no adapter)...")
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
            print("✅ Base model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model or tokenizer: {e}")
        raise

    return model, tokenizer


def main():
    """
    Entry point for the Text-to-SQL inference script.

    Supports three modes:
        1. Interactive mode (--interactive): REPL-style SQL generation.
        2. Single-query mode (--question and --context): Generate SQL for one example.
        3. Batch mode (--batch_file): Process a JSONL dataset file.

    Optionally loads a PEFT (LoRA/QLoRA) adapter on top of a base model
    for inference, as specified by --adapter_path.

    Command-line Arguments:
        --base_model     : Base model name or path (default: meta-llama/Meta-Llama-3-8B)
        --adapter_path   : Hugging Face ID or local path of PEFT adapter
        --interactive    : Run in interactive text-to-SQL REPL mode
        --question       : Single NL question (used with --context)
        --context        : Schema or CREATE TABLE statements for single query
        --batch_file     : Path to JSONL file for batch inference
        --output_file    : Path to save batch results (default: inference_outputs.jsonl)
    """
    parser = argparse.ArgumentParser(
        description="Text-to-SQL inference script with optional PEFT adapter support"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="Base model name or path (used if no adapter is provided)",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        help="Hugging Face ID or local path of PEFT adapter (e.g., 'chrisjcc/Meta-Llama-3.1-8B-text2sql-adapter')",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Single question for single-query mode",
    )
    parser.add_argument(
        "--context",
        type=str,
        help="Schema/context for single-query mode",
    )
    parser.add_argument(
        "--batch_file",
        type=str,
        help="Path to JSONL file for batch inference",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="inference_outputs.jsonl",
        help="Output file path for batch results",
    )

    args = parser.parse_args()

    # Initialize logging and environment
    setup_logging()
    check_gpu_availability()

    # Load configuration (optional external config)
    try:
        config = Config.load()
        print("✅ Configuration loaded successfully.")
    except Exception:
        config = None
        print("⚠️  No configuration file found or failed to load; proceeding with CLI args only.")

    # Load model and tokenizer (adapter-aware)
    try:
        model, tokenizer = load_model_and_tokenizer(args.base_model, args.adapter_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Select inference mode
    if args.interactive:
        interactive_mode(model, tokenizer)
    elif args.batch_file:
        batch_query_mode(model, tokenizer, args.batch_file, args.output_file)
    elif args.question and args.context:
        single_query_mode(model, tokenizer, args.question, args.context)
    else:
        parser.print_help()
        print(
            "\n⚠️  Please specify one of: "
            "--interactive, --batch_file, or both --question and --context."
        )


if __name__ == "__main__":
    main()
