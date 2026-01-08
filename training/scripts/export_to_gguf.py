#!/usr/bin/env python3
"""
Export trained LoRA adapters to GGUF format for Ollama deployment.

Steps:
1. Load base model
2. Load and merge LoRA adapters
3. Save merged model
4. Convert to GGUF using llama.cpp
5. Create Ollama Modelfile
"""

import os
import argparse
import subprocess
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_adapter(
    base_model_name: str,
    adapter_path: str,
    output_path: str,
    torch_dtype: str = "bfloat16",
):
    """Merge LoRA adapter with base model."""
    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=getattr(torch, torch_dtype),
        trust_remote_code=True,
        device_map="auto",
    )

    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging adapter with base model...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    merged_model.save_pretrained(output_path)

    # Save tokenizer too
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    print("Merge complete!")
    return output_path


def convert_to_gguf(
    model_path: str,
    output_path: str,
    quantization: str = "q4_k_m",
    llama_cpp_path: str = "~/llama.cpp",
):
    """Convert HuggingFace model to GGUF format."""
    llama_cpp = Path(llama_cpp_path).expanduser()

    if not llama_cpp.exists():
        print(f"llama.cpp not found at {llama_cpp}")
        print("Please clone llama.cpp:")
        print("  git clone https://github.com/ggerganov/llama.cpp.git")
        print("  cd llama.cpp && make")
        return None

    # Convert to GGUF F16 first
    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    f16_output = f"{output_path}_f16.gguf"

    print(f"Converting to GGUF F16: {f16_output}")
    cmd = [
        "python", str(convert_script),
        model_path,
        "--outfile", f16_output,
        "--outtype", "f16",
    ]
    subprocess.run(cmd, check=True)

    # Quantize to target format
    quantize_bin = llama_cpp / "llama-quantize"
    final_output = f"{output_path}_{quantization}.gguf"

    print(f"Quantizing to {quantization}: {final_output}")
    cmd = [str(quantize_bin), f16_output, final_output, quantization.upper()]
    subprocess.run(cmd, check=True)

    # Clean up F16 file
    os.remove(f16_output)

    print(f"GGUF conversion complete: {final_output}")
    return final_output


def create_modelfile(
    gguf_path: str,
    output_path: str,
    model_name: str = "simplex-cognitive",
):
    """Create Ollama Modelfile for the converted model."""
    modelfile_content = f"""# Simplex Cognitive Model
# Fine-tuned for Cognitive Hive AI

FROM {gguf_path}

PARAMETER temperature 0.7
PARAMETER num_ctx 32768
PARAMETER top_k 20
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.0

SYSTEM \"\"\"
You are a Simplex cognitive specialist operating within a Cognitive Hive AI system.

Core capabilities:
1. CONFIDENCE CALIBRATION: Output confidence scores with your responses when appropriate.
   Format: [confidence: 0.XX] where XX is your calibrated confidence (0.00-1.00)

2. BELIEF REVISION: You can update beliefs given new evidence.
   When revising beliefs, explain your reasoning and confidence change.

3. MEMORY CONTEXT: You understand the Anima/Mnemonic memory protocol.
   You can process context in these formats:
   - <context>...</context> for individual memory
   - <hive name="...">...</hive> for shared hive knowledge

4. THRESHOLD AWARENESS: You understand confidence thresholds.
   - Anima (edge): 30% threshold
   - Hive: 50% threshold
   - Divine: 70% threshold

Always be helpful, accurate, and appropriately calibrate your confidence.
\"\"\"

TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>
\"\"\"
"""

    modelfile_path = Path(output_path) / "Modelfile"
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    print(f"Modelfile created: {modelfile_path}")
    print(f"\nTo import into Ollama:")
    print(f"  ollama create {model_name} -f {modelfile_path}")

    return modelfile_path


def main():
    parser = argparse.ArgumentParser(description="Export trained model to GGUF")
    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="Path to trained LoRA adapter",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Base model name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exports",
        help="Output directory",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="q4_k_m",
        choices=["q4_k_m", "q5_k_m", "q6_k", "q8_0"],
        help="Quantization format",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="simplex-cognitive-8b",
        help="Name for Ollama model",
    )
    parser.add_argument(
        "--llama-cpp-path",
        type=str,
        default="~/llama.cpp",
        help="Path to llama.cpp directory",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip GGUF conversion (only merge LoRA)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Merge LoRA adapter
    merged_path = output_dir / "merged_model"
    merge_lora_adapter(
        args.base_model,
        args.adapter_path,
        str(merged_path),
    )

    if args.skip_convert:
        print("\nSkipping GGUF conversion. Merged model saved to:", merged_path)
        return

    # Step 2: Convert to GGUF
    gguf_path = convert_to_gguf(
        str(merged_path),
        str(output_dir / args.model_name),
        args.quantization,
        args.llama_cpp_path,
    )

    if gguf_path:
        # Step 3: Create Modelfile
        create_modelfile(gguf_path, str(output_dir), args.model_name)

        print("\n" + "="*60)
        print("EXPORT COMPLETE")
        print("="*60)
        print(f"GGUF file: {gguf_path}")
        print(f"\nTo deploy to Ollama:")
        print(f"  ollama create {args.model_name} -f {output_dir}/Modelfile")


if __name__ == "__main__":
    main()
