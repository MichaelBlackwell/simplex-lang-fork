#!/usr/bin/env python3
"""
Full Training Pipeline Runner

Orchestrates all training stages for Simplex Cognitive models:
1. Context Protocol Training
2. Confidence Calibration
3. Belief Revision
4. (Optional) Specialist LoRA Adapters

Usage:
    python run_full_training.py --all              # Run all stages
    python run_full_training.py --stage 1         # Run only stage 1
    python run_full_training.py --generate-only   # Only generate training data
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


def run_command(cmd: list, description: str):
    """Run a command and handle output."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60 + "\n")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\nERROR: {description} failed with code {result.returncode}")
        sys.exit(1)

    print(f"\nCompleted: {description}")


def main():
    parser = argparse.ArgumentParser(description="Simplex Cognitive Training Pipeline")
    parser.add_argument("--all", action="store_true", help="Run all training stages")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4, 5], help="Run specific stage")
    parser.add_argument("--generate-only", action="store_true", help="Only generate training data")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--local-test", action="store_true", help="Run in local test mode")
    parser.add_argument("--base-output", type=str, default="outputs")

    args = parser.parse_args()

    # Determine which stages to run
    stages_to_run = []
    if args.all:
        stages_to_run = [1, 2, 3, 4, 5]
    elif args.stage:
        stages_to_run = [args.stage]
    else:
        print("Please specify --all or --stage N")
        sys.exit(1)

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = Path(args.base_output) / f"run_{timestamp}"

    print("\n" + "="*60)
    print("SIMPLEX COGNITIVE MODEL TRAINING PIPELINE")
    print("="*60)
    print(f"Stages to run: {stages_to_run}")
    print(f"Output directory: {base_output}")
    print(f"Config file: {args.config}")
    print(f"Local test mode: {args.local_test}")
    print("="*60)

    # Stage definitions
    stages = {
        1: {
            "name": "Context Protocol Training",
            "script": "scripts/train_context_protocol.py",
            "output": "context_protocol",
        },
        2: {
            "name": "Confidence Calibration",
            "script": "scripts/train_confidence_calibration.py",
            "output": "confidence_calibration",
        },
        3: {
            "name": "Belief Revision",
            "script": "scripts/train_belief_revision.py",
            "output": "belief_revision",
        },
        4: {
            "name": "Neural IR/Gates Compatibility",
            "script": "scripts/train_neural_ir_gates.py",
            "output": "neural_ir_gates",
        },
        5: {
            "name": "Specialist LoRA Adapters",
            "script": "scripts/train_specialists.py",
            "output": "specialists",
        },
    }

    for stage_num in stages_to_run:
        stage = stages[stage_num]
        output_dir = base_output / stage["output"]

        # Build command
        cmd = [
            sys.executable,
            stage["script"],
            "--config", args.config,
            "--output-dir", str(output_dir),
            "--data-path", "data",
            "--generate-data",  # Always generate fresh data
        ]

        if args.local_test:
            cmd.append("--local-test")

        if args.generate_only:
            # Only run data generation part
            cmd = [
                sys.executable,
                stage["script"],
                "--config", args.config,
                "--data-path", "data",
                "--generate-data",
                "--num-examples", "100" if args.local_test else "10000",
            ]
            # Run the script but only generate data
            print(f"\nGenerating data for Stage {stage_num}: {stage['name']}")
            # We need a way to just generate data - for now, run full script
            # In a real implementation, we'd have a separate data generation mode

        run_command(cmd, f"Stage {stage_num}: {stage['name']}")

    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETE")
    print("="*60)
    print(f"\nOutputs saved to: {base_output}")
    print("\nNext steps:")
    print("1. Evaluate model on test set")
    print("2. Convert LoRA to merged model")
    print("3. Quantize and export to GGUF")
    print("4. Deploy to Ollama")


if __name__ == "__main__":
    main()
