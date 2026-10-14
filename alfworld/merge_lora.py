"""Merge a PEFT LoRA adapter into its base model for vLLM-loadable evaluation.

The training runs save LoRA adapters (``global_step_N/actor/lora_adapter``),
but the benchmark harness loads a plain Hugging Face model directory. This
merges one adapter into the base weights and writes the result, with the
base tokenizer, as safetensors.

Usage::

    python merge_lora.py <base_model_dir> <lora_adapter_dir> <output_dir>

The output is a full copy of the model (several GB); the benchmark launcher
writes it to node-local storage and deletes it after scoring.
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("base_model_dir", help="Base (student) model dir.")
    parser.add_argument("lora_adapter_dir", help="PEFT adapter dir.")
    parser.add_argument("output_dir", help="Where to write the merged model.")
    return parser.parse_args()


def merge(base_model_dir: str, lora_adapter_dir: str, output_dir: str) -> None:
    """Merge the adapter into the base weights; save model and tokenizer.

    Args:
        base_model_dir: Hugging Face directory of the base model.
        lora_adapter_dir: Directory holding the PEFT adapter.
        output_dir: Destination directory; created if missing.
    """
    print(
        f"[merge] base={base_model_dir}\n"
        f"[merge] adapter={lora_adapter_dir}\n"
        f"[merge] out={output_dir}",
        flush=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir, torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(base_model, lora_adapter_dir)
    model = model.merge_and_unload()
    model.save_pretrained(output_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_model_dir).save_pretrained(output_dir)
    print("MERGE_DONE", output_dir, flush=True)


def main() -> None:
    args = parse_args()
    merge(args.base_model_dir, args.lora_adapter_dir, args.output_dir)


if __name__ == "__main__":
    main()
