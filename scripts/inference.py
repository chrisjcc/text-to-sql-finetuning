"""
Inference script for text-to-SQL model using Hydra configuration.

Usage:
    # Interactive mode
    python scripts/inference.py inference.mode=interactive

    # Single query mode
    python scripts/inference.py inference.mode=single inference.question="What are all users?" inference.context="CREATE TABLE users (id INT, name TEXT)"

    # Batch mode
    python scripts/inference.py inference.mode=batch inference.batch_file=data/test_queries.jsonl inference.output_file=results.jsonl
"""

import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import get_original_cwd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import setup_logging, check_gpu_availability, extract_sql
from src.model_setup import ModelSetup


def generate_sql(model, tokenizer, question, context, max_new_tokens=512, temperature=0.0):
    """
    Generate SQL query from natural language question and database context.

    Args:
        model: The loaded language model
        tokenizer: The tokenizer
        question: Natural language question
        context: Database schema (CREATE TABLE statements)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 = greedy)

    Returns:
        Generated SQL query string
    """
    # Construct prompt using chat template if available
    system_msg = "You are a helpful assistant that converts natural language questions into SQL queries."
    user_msg = f"### Context:\n{context}\n\n### Question:\n{question}\n\n### SQL:"

    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_msg},
             {"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        prompt = f"{system_msg}\n\n{user_msg}"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    raw_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return extract_sql(raw_sql)


def interactive_mode(model, tokenizer, max_new_tokens=512, temperature=0.0):
    """
    Run interactive inference mode where user can input questions.

    Commands:
        /schema <schema>  - Set/update the database schema
        /clear            - Clear the current schema
        /exit, exit, quit - Exit interactive mode
    """
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("\nCommands:")
    print("  /schema <schema>  - Set/update database schema")
    print("  /clear            - Clear current schema")
    print("  /exit             - Exit interactive mode")
    print("\n" + "="*80 + "\n")

    context = ""

    while True:
        try:
            question = input("\nNatural language question: ").strip()

            if not question:
                continue

            if question.lower() in {"/exit", "exit", "quit"}:
                print("\nExiting interactive mode.")
                break

            elif question.startswith("/schema "):
                context = question[len("/schema "):].strip()
                print("✅ Schema updated.")
                continue

            elif question.startswith("/clear"):
                context = ""
                print("✅ Schema cleared.")
                continue

            if not context:
                context = input("Database schema (CREATE TABLE...): ").strip()

            print("\nGenerating SQL...")
            sql = generate_sql(model, tokenizer, question, context, max_new_tokens, temperature)
            print(f"\nGenerated SQL:\n{'-'*80}\n{sql}\n{'-'*80}")

        except KeyboardInterrupt:
            print("\n\nExiting interactive mode.")
            break

        except Exception as e:
            print(f"\n❌ Error: {e}")


def single_query_mode(model, tokenizer, question, context, max_new_tokens=512, temperature=0.0):
    """
    Run inference on a single question-context pair.
    """
    print("\n" + "="*80)
    print("SINGLE QUERY MODE")
    print("="*80)
    print(f"\nQuestion: {question}")
    print(f"Context: {context[:100]}..." if len(context) > 100 else f"Context: {context}")
    print("\nGenerating SQL...")

    sql = generate_sql(model, tokenizer, question, context, max_new_tokens, temperature)

    print(f"\nGenerated SQL:\n{'-'*80}\n{sql}\n{'-'*80}\n")


def parse_batch_example(example):
    """
    Parse a batch example into question, context, and ground truth SQL.
    Supports two formats:
    1. Standard format: {"question": "...", "context": "...", "query": "..."}
    2. Messages format: {"messages": [{"role": "...", "content": "..."}]}

    Returns:
        Tuple of (question, context, ground_truth_sql) or None if invalid
    """
    # Format 1: Standard format
    if "question" in example:
        question = example.get("question")
        context = example.get("context", "")
        ground_truth = example.get("query")

        if question:
            return question, context, ground_truth

    # Format 2: Messages format (from prepared dataset)
    elif "messages" in example:
        messages = example.get("messages", [])

        if len(messages) < 2:
            return None

        # Extract context from system message or user message
        context = ""
        question = ""
        ground_truth = None

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "system":
                # System message often contains schema/context
                context = content
            elif role == "user":
                # User message contains the question
                # May also contain context if not in system message
                question = content
                # If context is in user message, try to extract it
                if "### Context:" in content or "CREATE TABLE" in content:
                    parts = content.split("### Question:")
                    if len(parts) == 2:
                        context = parts[0].replace("### Context:", "").strip()
                        question = parts[1].replace("### SQL:", "").strip()
                    else:
                        # Just use the whole thing as question
                        question = content
            elif role == "assistant":
                # Assistant message contains the ground truth SQL
                ground_truth = content

        if question:
            return question, context, ground_truth

    return None


def batch_query_mode(model, tokenizer, batch_file, output_file="inference_outputs.jsonl",
                     max_new_tokens=512, temperature=0.0):
    """
    Run inference on a batch of questions from a JSON or JSONL file.

    Supports two formats:
    1. JSONL with {"question": "...", "context": "...", "query": "..."}
    2. JSON with messages format: {"messages": [{"role": "...", "content": "..."}]}
    """
    print("\n" + "="*80)
    print("BATCH QUERY MODE")
    print("="*80)
    print(f"\nInput file: {batch_file}")
    print(f"Output file: {output_file}")

    results = []

    # Load batch file
    batch_path = Path(batch_file)
    if not batch_path.exists():
        print(f"❌ Error: Batch file not found: {batch_file}")
        return

    # Determine if it's JSON or JSONL
    with open(batch_file, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == '[':
            # Standard JSON array
            data = json.load(f)
            if not isinstance(data, list):
                print(f"❌ Error: Expected JSON array, got {type(data)}")
                return
            examples = data
        else:
            # JSONL format
            examples = [json.loads(line) for line in f if line.strip()]

    print(f"\nLoaded {len(examples)} examples from file")
    print(f"Processing queries...")

    # Process each example
    skipped = 0
    for example in tqdm(examples, desc="Running batch inference"):
        parsed = parse_batch_example(example)

        if parsed is None:
            skipped += 1
            continue

        question, context, ground_truth = parsed

        try:
            pred_sql = generate_sql(model, tokenizer, question, context, max_new_tokens, temperature)
        except Exception as e:
            print(f"\n❌ Error processing question: {e}")
            pred_sql = None

        results.append({
            "question": question,
            "context": context,
            "predicted_sql": pred_sql,
            "ground_truth_sql": ground_truth
        })

    if skipped > 0:
        print(f"\n⚠️  Skipped {skipped} invalid examples")

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✅ Batch inference completed. Results saved to '{output_file}'.")
    print(f"   Processed: {len(results)} queries")
    if skipped > 0:
        print(f"   Skipped: {skipped} invalid examples")


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Hydra-based inference entry point."""
    load_dotenv()
    print("\n📊 Starting inference...")
    print("\nConfiguration:\n", OmegaConf.to_yaml(cfg))

    # Setup logging
    log_file = Path(get_original_cwd()) / cfg.logging.log_dir / "inference.log"
    setup_logging(log_file)

    # Check GPU
    check_gpu_availability()

    # Load model + tokenizer
    model_path = cfg.evaluation.model_path
    adapter_path = cfg.evaluation.adapter_path

    print("\n" + "="*80)
    model_desc = f"with adapter: {adapter_path}" if adapter_path else "base model only"
    print(f"Loading model {model_desc}...")
    print("="*80)

    try:
        model, tokenizer = ModelSetup.load_trained_model(
            model_path=model_path,
            adapter_path=adapter_path
        )
        model.eval()
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Get inference configuration
    inference_cfg = cfg.get('inference', {})
    mode = inference_cfg.get('mode', 'interactive')
    max_new_tokens = inference_cfg.get('max_new_tokens', cfg.evaluation.max_new_tokens)
    temperature = inference_cfg.get('temperature', cfg.evaluation.temperature)

    # Run appropriate inference mode
    if mode == 'interactive':
        interactive_mode(model, tokenizer, max_new_tokens, temperature)

    elif mode == 'batch':
        batch_file = inference_cfg.get('batch_file')
        if not batch_file:
            print("❌ Error: batch_file must be specified for batch mode")
            print("Usage: python scripts/inference.py inference.mode=batch inference.batch_file=<path>")
            return

        output_file = inference_cfg.get('output_file', 'inference_outputs.jsonl')
        batch_query_mode(model, tokenizer, batch_file, output_file, max_new_tokens, temperature)

    elif mode == 'single':
        question = inference_cfg.get('question')
        context = inference_cfg.get('context')

        if not question or not context:
            print("❌ Error: Both question and context must be specified for single mode")
            print("Usage: python scripts/inference.py inference.mode=single inference.question='...' inference.context='...'")
            return

        single_query_mode(model, tokenizer, question, context, max_new_tokens, temperature)

    else:
        print(f"❌ Unknown mode: {mode}")
        print("Valid modes: interactive, batch, single")
        print("\nExamples:")
        print("  python scripts/inference.py inference.mode=interactive")
        print("  python scripts/inference.py inference.mode=single inference.question='...' inference.context='...'")
        print("  python scripts/inference.py inference.mode=batch inference.batch_file=data/test.jsonl")

    print("\n" + "="*80)
    print("INFERENCE COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
