#!/usr/bin/env python3
"""
Test script to validate data generation for all training stages.
Runs locally without GPU - only tests data generation, not training.
"""

import sys
import json
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def test_context_protocol_data():
    """Test context protocol data generation."""
    print("\n" + "="*60)
    print("Testing Context Protocol Data Generation")
    print("="*60)

    from train_context_protocol import create_context_protocol_example

    examples = []
    for i in range(10):
        example = create_context_protocol_example()
        examples.append(example)

        if i < 2:  # Show first 2 examples
            print(f"\n--- Example {i+1} ---")
            print(example["text"][:500] + "..." if len(example["text"]) > 500 else example["text"])

    # Validate structure
    for i, ex in enumerate(examples):
        assert "text" in ex, f"Example {i} missing 'text' key"
        assert len(ex["text"]) > 100, f"Example {i} text too short"
        assert "confidence" in ex["text"].lower(), f"Example {i} missing confidence"

    print(f"\n✓ Generated {len(examples)} valid context protocol examples")
    return True


def test_confidence_calibration_data():
    """Test confidence calibration data generation."""
    print("\n" + "="*60)
    print("Testing Confidence Calibration Data Generation")
    print("="*60)

    from train_confidence_calibration import generate_calibration_example

    qa_types = ["factual", "ambiguous", "unknowable", "threshold_comparison"]
    examples = []

    for qa_type in qa_types:
        for i in range(3):
            example = generate_calibration_example(qa_type)
            examples.append(example)

            if i == 0:  # Show first example of each type
                print(f"\n--- {qa_type.upper()} Example ---")
                print(example["text"][:400] + "..." if len(example["text"]) > 400 else example["text"])

    # Validate structure
    for i, ex in enumerate(examples):
        assert "text" in ex, f"Example {i} missing 'text' key"
        assert "[confidence:" in ex["text"], f"Example {i} missing confidence marker"

    print(f"\n✓ Generated {len(examples)} valid calibration examples")
    return True


def test_belief_revision_data():
    """Test belief revision data generation."""
    print("\n" + "="*60)
    print("Testing Belief Revision Data Generation")
    print("="*60)

    from train_belief_revision import generate_belief_revision_example, generate_resistance_example

    examples = []

    # Normal revisions
    for i in range(5):
        example = generate_belief_revision_example()
        examples.append(example)
        if i < 2:
            print(f"\n--- Belief Revision Example {i+1} ---")
            print(example["text"][:400] + "...")

    # Resistance examples
    for i in range(3):
        example = generate_resistance_example()
        examples.append(example)
        if i == 0:
            print(f"\n--- Resistance Example ---")
            print(example["text"][:400] + "...")

    # Validate structure
    for i, ex in enumerate(examples):
        assert "text" in ex, f"Example {i} missing 'text' key"
        assert "belief" in ex["text"].lower(), f"Example {i} missing belief reference"

    print(f"\n✓ Generated {len(examples)} valid belief revision examples")
    return True


def test_neural_ir_gates_data():
    """Test Neural IR/Gates data generation."""
    print("\n" + "="*60)
    print("Testing Neural IR/Gates Data Generation")
    print("="*60)

    from train_neural_ir_gates import (
        generate_temperature_aware_example,
        generate_soft_logic_example,
        generate_probability_output_example,
        generate_straight_through_example,
        generate_gumbel_softmax_example,
    )

    generators = [
        ("Temperature Aware", generate_temperature_aware_example),
        ("Soft Logic", generate_soft_logic_example),
        ("Probability Output", generate_probability_output_example),
        ("Straight Through", generate_straight_through_example),
        ("Gumbel Softmax", generate_gumbel_softmax_example),
    ]

    examples = []
    for name, generator in generators:
        example = generator()
        examples.append(example)
        print(f"\n--- {name} Example ---")
        print(example["text"][:400] + "..." if len(example["text"]) > 400 else example["text"])

    # Validate structure
    for i, ex in enumerate(examples):
        assert "text" in ex, f"Example {i} missing 'text' key"
        assert len(ex["text"]) > 100, f"Example {i} text too short"

    print(f"\n✓ Generated {len(examples)} valid Neural IR/Gates examples")
    return True


def test_full_data_generation():
    """Test full data generation pipeline."""
    print("\n" + "="*60)
    print("Testing Full Data Generation Pipeline")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        from train_context_protocol import generate_training_data as gen_context
        from train_confidence_calibration import generate_calibration_data as gen_calibration
        from train_belief_revision import generate_training_data as gen_belief
        from train_neural_ir_gates import generate_training_data as gen_neural

        print("\nGenerating small test datasets...")

        # Context Protocol
        context_file = gen_context(100, tmpdir)
        with open(context_file) as f:
            context_count = sum(1 for _ in f)
        print(f"  Context Protocol: {context_count} examples")

        # Confidence Calibration
        calib_file = gen_calibration(100, tmpdir)
        with open(calib_file) as f:
            calib_count = sum(1 for _ in f)
        print(f"  Confidence Calibration: {calib_count} examples")

        # Belief Revision
        belief_file = gen_belief(100, tmpdir)
        with open(belief_file) as f:
            belief_count = sum(1 for _ in f)
        print(f"  Belief Revision: {belief_count} examples")

        # Neural IR/Gates
        neural_file = gen_neural(100, tmpdir)
        with open(neural_file) as f:
            neural_count = sum(1 for _ in f)
        print(f"  Neural IR/Gates: {neural_count} examples")

        total = context_count + calib_count + belief_count + neural_count
        print(f"\n✓ Total: {total} examples generated successfully")

    return True


def main():
    """Run all data generation tests."""
    print("="*60)
    print("SIMPLEX COGNITIVE TRAINING - DATA GENERATION TESTS")
    print("="*60)

    tests = [
        ("Context Protocol", test_context_protocol_data),
        ("Confidence Calibration", test_confidence_calibration_data),
        ("Belief Revision", test_belief_revision_data),
        ("Neural IR/Gates", test_neural_ir_gates_data),
        ("Full Pipeline", test_full_data_generation),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"\n✗ {name} FAILED: {e}")

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed

    for name, success, error in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")
        if error:
            print(f"         Error: {error}")

    print(f"\nTotal: {passed}/{len(results)} passed")

    if failed > 0:
        sys.exit(1)
    else:
        print("\n✓ All data generation tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
