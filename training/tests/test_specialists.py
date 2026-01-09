#!/usr/bin/env python3
"""
Test specialist data generation.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from train_specialists import (
    SPECIALISTS,
    GENERATORS,
    generate_training_data,
)


def test_all_specialists():
    """Test all specialist generators produce valid output."""
    print("="*60)
    print("SPECIALIST DATA GENERATION TESTS")
    print("="*60)

    results = []

    for name, config in SPECIALISTS.items():
        print(f"\n--- Testing: {name} ---")
        print(f"  Description: {config.description}")
        print(f"  LoRA Rank: {config.lora_rank}")
        print(f"  License: {config.license_status}")

        try:
            # Generate a few examples
            generator = GENERATORS[name]
            examples = [generator() for _ in range(5)]

            # Validate
            for i, ex in enumerate(examples):
                assert "text" in ex, f"Example {i} missing 'text'"
                assert len(ex["text"]) > 50, f"Example {i} too short"
                assert "confidence" in ex["text"].lower(), f"Example {i} missing confidence"

            print(f"  Generated: {len(examples)} examples")
            print(f"  Sample ({len(examples[0]['text'])} chars):")
            print(f"    {examples[0]['text'][:200]}...")
            print(f"  ✓ PASS")
            results.append((name, True, None))

        except Exception as e:
            print(f"  ✗ FAIL: {e}")
            results.append((name, False, str(e)))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, success, _ in results if success)
    for name, success, error in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
        if error:
            print(f"    Error: {error}")

    print(f"\nTotal: {passed}/{len(results)} passed")

    return passed == len(results)


if __name__ == "__main__":
    success = test_all_specialists()
    sys.exit(0 if success else 1)
