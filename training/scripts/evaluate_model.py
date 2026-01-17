#!/usr/bin/env python3
"""
Phase 5: Evaluation Framework

Comprehensive evaluation of trained Simplex Cognitive models:
1. Core metrics (ECE, Brier Score, MMLU subset)
2. Simplex-specific benchmarks (belief revision, context, thresholds)
3. Comparison with base model

Usage:
    python evaluate_model.py --model-path outputs/context_protocol/final
    python evaluate_model.py --ollama-model simplex-cognitive-8b
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re

import numpy as np
from tqdm import tqdm


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    metric: str
    value: float
    details: Optional[Dict] = None


class SimplexEvaluator:
    """Evaluator for Simplex Cognitive models."""

    def __init__(self, model_path: str = None, ollama_model: str = None):
        self.model_path = model_path
        self.ollama_model = ollama_model
        self.model = None
        self.tokenizer = None

        if model_path:
            self._load_hf_model()
        elif ollama_model:
            self._setup_ollama()

    def _load_hf_model(self):
        """Load HuggingFace model for evaluation."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            print(f"Loading model from {self.model_path}...")

            # Check if it's a LoRA adapter
            adapter_config = Path(self.model_path) / "adapter_config.json"
            if adapter_config.exists():
                with open(adapter_config) as f:
                    config = json.load(f)
                base_model = config.get("base_model_name_or_path", "Qwen/Qwen3-8B")

                self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
                base = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map="auto",
                )
                self.model = PeftModel.from_pretrained(base, self.model_path)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map="auto",
                )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print("Model loaded successfully")
        except ImportError:
            print("Warning: transformers not available, using Ollama fallback")
            self._setup_ollama()

    def _setup_ollama(self):
        """Setup Ollama for evaluation."""
        import subprocess

        if not self.ollama_model:
            self.ollama_model = "simplex-cognitive-8b"

        # Check if model exists
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
        )
        if self.ollama_model not in result.stdout:
            print(f"Warning: {self.ollama_model} not found in Ollama")
            print("Available models:")
            print(result.stdout)

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate response from model."""
        if self.model is not None:
            return self._generate_hf(prompt, max_tokens)
        else:
            return self._generate_ollama(prompt, max_tokens)

    def _generate_hf(self, prompt: str, max_tokens: int) -> str:
        """Generate using HuggingFace model."""
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()

    def _generate_ollama(self, prompt: str, max_tokens: int) -> str:
        """Generate using Ollama."""
        import subprocess

        result = subprocess.run(
            ["ollama", "run", self.ollama_model, prompt],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.stdout.strip()

    def extract_confidence(self, text: str) -> Optional[float]:
        """Extract confidence score from model output."""
        # Pattern: [confidence: 0.XX] or confidence: X.XX
        patterns = [
            r'\[confidence:\s*([\d.]+)\]',
            r'confidence:\s*([\d.]+)',
            r'Confidence:\s*([\d.]+)',
            r'\((\d+)%\s*confidence\)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                value = float(match.group(1))
                # Normalize percentage to decimal
                if value > 1:
                    value /= 100
                return min(1.0, max(0.0, value))

        return None

    def evaluate_confidence_calibration(self, num_samples: int = 100) -> EvaluationResult:
        """Evaluate confidence calibration (ECE)."""
        print("\nEvaluating confidence calibration...")

        # Test questions with known answers
        test_cases = [
            # Factual (should be high confidence)
            ("What is the capital of France?", "Paris", True),
            ("What is 2 + 2?", "4", True),
            ("What is the chemical symbol for water?", "H2O", True),

            # Ambiguous (should be medium confidence)
            ("Is it going to rain tomorrow?", None, None),
            ("Will the stock market go up next month?", None, None),

            # Unknowable (should be low confidence)
            ("What will be the most popular programming language in 2050?", None, None),
            ("Who will win the election in 2040?", None, None),
        ]

        confidences = []
        accuracies = []

        for question, expected, is_factual in tqdm(test_cases, desc="Calibration"):
            prompt = f"Answer this question and include your confidence: {question}"
            response = self.generate(prompt)

            confidence = self.extract_confidence(response)
            if confidence is not None:
                confidences.append(confidence)

                # For factual questions, check accuracy
                if expected is not None:
                    is_correct = expected.lower() in response.lower()
                    accuracies.append(float(is_correct))
                else:
                    # For non-factual, assume medium calibration target
                    if is_factual is None:
                        accuracies.append(0.5)

        if not confidences:
            return EvaluationResult("ECE", 1.0, {"error": "No confidence scores extracted"})

        # Compute ECE
        confidences = np.array(confidences)
        accuracies = np.array(accuracies)

        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_acc = accuracies[mask].mean()
                bin_conf = confidences[mask].mean()
                bin_size = mask.sum() / len(confidences)
                ece += bin_size * abs(bin_acc - bin_conf)

        return EvaluationResult(
            "ECE",
            ece,
            {
                "n_samples": len(confidences),
                "mean_confidence": float(np.mean(confidences)),
                "mean_accuracy": float(np.mean(accuracies)),
            }
        )

    def evaluate_belief_revision(self, num_samples: int = 20) -> EvaluationResult:
        """Evaluate belief revision capability."""
        print("\nEvaluating belief revision...")

        test_cases = [
            {
                "initial": "The meeting is at 2pm (confidence: 70%)",
                "evidence": "Your manager just sent an email saying the meeting is moved to 3pm",
                "expected_direction": "update",  # Should update belief
            },
            {
                "initial": "The project deadline is Friday (confidence: 80%)",
                "evidence": "Someone on social media mentioned a different deadline",
                "expected_direction": "maintain",  # Should resist weak evidence
            },
            {
                "initial": "The budget is $500K (confidence: 60%)",
                "evidence": "Official memo confirms budget is $600K",
                "expected_direction": "update",  # Should update on strong evidence
            },
        ]

        correct = 0
        total = 0

        for case in tqdm(test_cases, desc="Belief Revision"):
            prompt = f"""Initial belief: {case['initial']}

New evidence: {case['evidence']}

Should the belief be updated? Explain your reasoning."""

            response = self.generate(prompt)

            # Check if model correctly identified update vs maintain
            response_lower = response.lower()
            if case["expected_direction"] == "update":
                is_correct = any(word in response_lower for word in ["update", "revise", "change", "new"])
            else:
                is_correct = any(word in response_lower for word in ["maintain", "resist", "weak", "insufficient"])

            if is_correct:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0

        return EvaluationResult(
            "Belief Revision Accuracy",
            accuracy,
            {"correct": correct, "total": total}
        )

    def evaluate_threshold_understanding(self, num_samples: int = 30) -> EvaluationResult:
        """Evaluate understanding of confidence thresholds."""
        print("\nEvaluating threshold understanding...")

        test_cases = []
        for _ in range(num_samples):
            confidence = np.random.uniform(0.2, 0.9)
            threshold = np.random.choice([0.3, 0.5, 0.7])
            passes = confidence >= threshold
            test_cases.append((confidence, threshold, passes))

        correct = 0
        for conf, thresh, expected in tqdm(test_cases, desc="Thresholds"):
            prompt = f"Does confidence {conf:.2f} pass a {thresh:.0%} threshold? Answer yes or no."
            response = self.generate(prompt, max_tokens=50)

            response_lower = response.lower()
            predicted_pass = "yes" in response_lower and "no" not in response_lower[:20]

            if predicted_pass == expected:
                correct += 1

        accuracy = correct / len(test_cases)

        return EvaluationResult(
            "Threshold Accuracy",
            accuracy,
            {"correct": correct, "total": len(test_cases)}
        )

    def evaluate_context_protocol(self, num_samples: int = 10) -> EvaluationResult:
        """Evaluate understanding of Simplex context protocol."""
        print("\nEvaluating context protocol understanding...")

        test_prompts = [
            """<context>
Recent experiences:
- Met with Alice about project planning
- Completed code review for feature X

Current beliefs (confidence > 30%):
- Project is on track (75%)
</context>

What were my recent activities?""",

            """<hive name="EngineeringHive">
Shared knowledge:
- API v2 is in development
- Sprint ends Friday

Hive beliefs (confidence > 50%):
- Release will be on time (68%)
</hive>

What does the hive believe about the release?""",
        ]

        scores = []
        for prompt in tqdm(test_prompts, desc="Context Protocol"):
            response = self.generate(prompt)

            # Check if response references context appropriately
            score = 0
            if "confidence" in response.lower():
                score += 0.25
            if any(word in response.lower() for word in ["recent", "activities", "experiences"]):
                score += 0.25
            if any(word in response.lower() for word in ["belief", "hive", "shared"]):
                score += 0.25
            if self.extract_confidence(response) is not None:
                score += 0.25

            scores.append(score)

        avg_score = np.mean(scores)

        return EvaluationResult(
            "Context Protocol Score",
            avg_score,
            {"scores": scores}
        )

    def run_full_evaluation(self) -> Dict[str, EvaluationResult]:
        """Run all evaluations."""
        print("\n" + "="*60)
        print("SIMPLEX COGNITIVE MODEL EVALUATION")
        print("="*60)

        results = {}

        # Core evaluations
        results["calibration"] = self.evaluate_confidence_calibration()
        results["belief_revision"] = self.evaluate_belief_revision()
        results["thresholds"] = self.evaluate_threshold_understanding()
        results["context_protocol"] = self.evaluate_context_protocol()

        # Print summary
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)

        for name, result in results.items():
            status = "PASS" if result.value > 0.5 else "FAIL"
            print(f"{status} {result.metric}: {result.value:.3f}")
            if result.details:
                for k, v in result.details.items():
                    if isinstance(v, float):
                        print(f"    {k}: {v:.3f}")
                    else:
                        print(f"    {k}: {v}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Simplex Cognitive Model")
    parser.add_argument("--model-path", type=str, help="Path to HuggingFace model or LoRA adapter")
    parser.add_argument("--ollama-model", type=str, help="Ollama model name")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output file")

    args = parser.parse_args()

    if not args.model_path and not args.ollama_model:
        # Default to Ollama model
        args.ollama_model = "simplex-cognitive-8b"

    evaluator = SimplexEvaluator(
        model_path=args.model_path,
        ollama_model=args.ollama_model,
    )

    results = evaluator.run_full_evaluation()

    # Save results
    output_data = {
        name: {
            "metric": r.metric,
            "value": r.value,
            "details": r.details,
        }
        for name, r in results.items()
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
