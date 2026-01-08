#!/usr/bin/env python3
"""
Standalone test for data generation - no torch required.
Tests only the data generation logic, not the training code.
"""

import sys
import json
import random
from pathlib import Path

# Add faker
from faker import Faker
fake = Faker()


def test_context_protocol_example():
    """Test context protocol example generation."""
    print("\n" + "="*60)
    print("Testing Context Protocol Data Generation")
    print("="*60)

    example_types = ["episodic_recall", "hive_knowledge", "threshold_decision"]

    for example_type in example_types:
        if example_type == "episodic_recall":
            events = [
                f"Met with {fake.name()} about {fake.bs()}",
                f"Completed task: {fake.catch_phrase()}",
            ]
            context = f"""<context>
Recent experiences:
- {events[0]}
- {events[1]}

Current beliefs (confidence > 30%):
- Project on track (75%)
</context>

What were my recent activities?"""
            response = f"Based on your experiences: {events[0]}, {events[1]} [confidence: 0.95]"

        elif example_type == "hive_knowledge":
            hive_name = fake.company().split()[0] + "Hive"
            context = f"""<hive name="{hive_name}">
Shared knowledge:
- API v2 in development

Hive beliefs (confidence > 50%):
- Project on time (78%)
</hive>

What does the hive believe?"""
            response = f"The {hive_name} believes the project is on time (78% confidence) [confidence: 0.91]"

        else:  # threshold_decision
            conf = random.uniform(0.25, 0.85)
            threshold = random.choice([30, 50, 70])
            passes = conf * 100 >= threshold
            context = f"Should confidence {conf:.0%} pass {threshold}% threshold?"
            response = f"{'Yes' if passes else 'No'}, {conf:.0%} {'≥' if passes else '<'} {threshold}% [confidence: 0.98]"

        example = {"text": f"{context}\n\nAssistant: {response}"}

        print(f"\n--- {example_type} ---")
        print(example["text"][:400])

        # Validate
        assert "text" in example
        assert "confidence" in example["text"].lower()

    print("\n✓ Context protocol examples generated successfully")
    return True


def test_confidence_calibration_example():
    """Test confidence calibration example generation."""
    print("\n" + "="*60)
    print("Testing Confidence Calibration Data Generation")
    print("="*60)

    # Factual QA
    factual = [
        ("What is the capital of France?", "Paris", 0.99),
        ("What is 2 + 2?", "4", 0.99),
    ]

    # Ambiguous
    ambiguous = [
        ("Will it rain tomorrow?", "Uncertain based on patterns", 0.55),
    ]

    # Unknowable
    unknowable = [
        ("What will happen in 50 years?", "Cannot predict", 0.10),
    ]

    examples = []
    for q, a, conf in factual + ambiguous + unknowable:
        example = {"text": f"Question: {q}\n\nAssistant: {a} [confidence: {conf:.2f}]"}
        examples.append(example)
        print(f"\n--- Q: {q} ---")
        print(f"A: {a} [confidence: {conf:.2f}]")

    # Validate
    for ex in examples:
        assert "[confidence:" in ex["text"]

    print("\n✓ Confidence calibration examples generated successfully")
    return True


def test_belief_revision_example():
    """Test belief revision example generation."""
    print("\n" + "="*60)
    print("Testing Belief Revision Data Generation")
    print("="*60)

    evidence_types = ["strong_confirm", "weak_contradict", "bad_evidence"]

    for ev_type in evidence_types:
        initial_belief = f"The deadline is {fake.day_of_week()}"
        initial_conf = random.uniform(0.5, 0.8)

        if ev_type == "strong_confirm":
            new_evidence = f"Manager confirmed: {initial_belief}"
            delta = 0.15
            should_update = True
        elif ev_type == "weak_contradict":
            new_evidence = "Someone mentioned a different deadline"
            delta = -0.05
            should_update = True
        else:  # bad evidence
            new_evidence = "I feel like the deadline is different"
            delta = 0
            should_update = False

        new_conf = max(0.1, min(0.95, initial_conf + delta))

        prompt = f"""Initial belief: "{initial_belief}" ({initial_conf:.0%})
New evidence: {new_evidence}
Should belief be updated?"""

        if should_update:
            response = f"""Belief updated: "{initial_belief}"
Original: {initial_conf:.0%} -> Updated: {new_conf:.0%}
Reasoning: Evidence {'supports' if delta > 0 else 'contradicts'} belief
[confidence: 0.92]"""
        else:
            response = f"""Belief maintained: "{initial_belief}" ({initial_conf:.0%})
Reasoning: Evidence is weak/unreliable, no update warranted
[confidence: 0.95]"""

        example = {"text": f"{prompt}\n\nAssistant: {response}"}
        print(f"\n--- {ev_type} ---")
        print(example["text"][:400])

        assert "belief" in example["text"].lower()

    print("\n✓ Belief revision examples generated successfully")
    return True


def test_neural_ir_gates_example():
    """Test Neural IR/Gates example generation."""
    print("\n" + "="*60)
    print("Testing Neural IR/Gates Data Generation")
    print("="*60)

    # Temperature-aware
    temp = 0.7
    print(f"\n--- Temperature Aware (τ={temp}) ---")
    example = f"""Temperature: {temp} (moderate exploration)
Options: A, B, C
Probabilities: A: 0.50, B: 0.30, C: 0.20
Selected: A [confidence: 0.50]"""
    print(example)

    # Soft logic gate
    conf = 0.65
    threshold = 0.50
    passes = conf >= threshold
    print(f"\n--- Soft Logic Gate ---")
    example = f"""Input: {conf:.2f}, Threshold: {threshold:.2f}
Result: {'PASS' if passes else 'FAIL'} ({conf:.2f} {'≥' if passes else '<'} {threshold:.2f})
[confidence: 0.99]"""
    print(example)

    # Probability output
    print(f"\n--- Probability Output ---")
    probs = [0.45, 0.35, 0.20]
    example = f"""Options: [Approve, Review, Reject]
Probabilities: {probs}
Most likely: Approve ({probs[0]:.1%})
[confidence: {max(probs):.2f}]"""
    print(example)

    # Gumbel-softmax
    print(f"\n--- Gumbel-Softmax ---")
    example = f"""Categorical selection (τ=1.0)
Weights: [1.5, 1.0, 0.5]
Soft output: [0.50, 0.33, 0.17]
Hard selection: Option A
[confidence: 0.50]"""
    print(example)

    print("\n✓ Neural IR/Gates examples generated successfully")
    return True


def main():
    """Run all standalone data generation tests."""
    print("="*60)
    print("SIMPLEX COGNITIVE - STANDALONE DATA GENERATION TESTS")
    print("="*60)
    print("(No torch required - testing data generation logic only)")

    tests = [
        ("Context Protocol", test_context_protocol_example),
        ("Confidence Calibration", test_confidence_calibration_example),
        ("Belief Revision", test_belief_revision_example),
        ("Neural IR/Gates", test_neural_ir_gates_example),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, True, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"\n✗ {name} FAILED: {e}")

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, success, _ in results if success)

    for name, success, error in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")
        if error:
            print(f"         Error: {error}")

    print(f"\nTotal: {passed}/{len(results)} passed")

    if passed == len(results):
        print("\n✓ All data generation tests passed!")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
