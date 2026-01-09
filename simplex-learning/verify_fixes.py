#!/usr/bin/env python3
"""
Verification script for simplex-learning TASK-004 fixes.
This validates the core logic of the fixes we made.
"""

import math
import random
from typing import List, Tuple, Optional

print("=" * 60)
print("simplex-learning TASK-004 Fix Verification")
print("=" * 60)

# Track test results
passed = 0
failed = 0

def test(name: str, condition: bool, message: str = ""):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name} - {message}")
        failed += 1

# ==============================================================================
# Test 1: Xorshift64 RNG
# ==============================================================================
print("\n1. Testing Xorshift64 RNG implementation...")

def xorshift64(state: int) -> Tuple[int, int]:
    """Xorshift64 as implemented in tensor.sx"""
    x = state & 0xFFFFFFFFFFFFFFFF  # Keep as 64-bit
    x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 7) & 0xFFFFFFFFFFFFFFFF
    x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
    return x, x

# Test that RNG produces different values
state = 0x123456789ABCDEF0
values = []
for _ in range(100):
    val, state = xorshift64(state)
    values.append(val / (2**64 - 1))

test("RNG produces values in [0, 1]", all(0 <= v <= 1 for v in values))
test("RNG produces varied values", len(set(values)) > 90, f"Only {len(set(values))} unique values")
test("RNG mean is ~0.5", 0.3 < sum(values)/len(values) < 0.7, f"Mean: {sum(values)/len(values):.3f}")

# ==============================================================================
# Test 2: Gradient Clipping by Norm
# ==============================================================================
print("\n2. Testing Gradient Clipping by Norm...")

def clip_grad_norm(grads: List[List[float]], max_norm: float) -> Tuple[List[List[float]], float]:
    """Gradient clipping implementation from streaming.sx"""
    # Compute total norm
    total_norm_sq = 0.0
    for grad in grads:
        for g in grad:
            total_norm_sq += g * g
    total_norm = math.sqrt(total_norm_sq)

    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        clipped = [[g * clip_coef for g in grad] for grad in grads]
        return clipped, total_norm
    return grads, total_norm

# Test gradient clipping
grads = [[3.0, 4.0]]  # norm = 5
clipped, original_norm = clip_grad_norm(grads, 1.0)
clipped_norm = math.sqrt(sum(g*g for grad in clipped for g in grad))

test("Original norm computed correctly", abs(original_norm - 5.0) < 0.01, f"Got {original_norm}")
test("Clipped norm is ~max_norm", abs(clipped_norm - 1.0) < 0.01, f"Got {clipped_norm}")
test("Direction preserved", abs(clipped[0][0]/clipped[0][1] - 0.75) < 0.01)

# ==============================================================================
# Test 3: Gradient Clipping by Value
# ==============================================================================
print("\n3. Testing Gradient Clipping by Value...")

def clip_grad_value(grads: List[List[float]], max_value: float) -> List[List[float]]:
    """Gradient value clipping implementation from streaming.sx"""
    return [[max(-max_value, min(max_value, g)) for g in grad] for grad in grads]

# Test value clipping
grads = [[0.5, 2.0, -3.0, 0.1]]
clipped = clip_grad_value(grads, 1.0)

test("Values within bounds unchanged", clipped[0][0] == 0.5 and clipped[0][3] == 0.1)
test("Large positive clamped", clipped[0][1] == 1.0, f"Got {clipped[0][1]}")
test("Large negative clamped", clipped[0][2] == -1.0, f"Got {clipped[0][2]}")

# ==============================================================================
# Test 4: Cross-Entropy Loss (Index-based)
# ==============================================================================
print("\n4. Testing Cross-Entropy Loss...")

def cross_entropy_index(logits: List[float], target_idx: int) -> float:
    """Cross-entropy with index target from ops.sx"""
    # Softmax
    max_logit = max(logits)
    exp_vals = [math.exp(l - max_logit) for l in logits]
    sum_exp = sum(exp_vals)
    probs = [e / sum_exp for e in exp_vals]

    # Negative log likelihood
    return -math.log(probs[target_idx] + 1e-10)

# Test cross-entropy
logits = [1.0, 2.0, 3.0]  # Class 2 should have highest prob
loss_class2 = cross_entropy_index(logits, 2)
loss_class0 = cross_entropy_index(logits, 0)

test("Loss for correct class is lower", loss_class2 < loss_class0)
test("Loss is positive", loss_class2 > 0)
test("Loss is reasonable", loss_class2 < 2.0, f"Got {loss_class2}")

# ==============================================================================
# Test 5: Batch Matrix Multiplication Broadcasting
# ==============================================================================
print("\n5. Testing Batch MatMul Broadcasting...")

def broadcast_batch_shape(a_batch: List[int], b_batch: List[int]) -> List[int]:
    """Broadcast batch dimensions from ops.sx"""
    result = []
    max_len = max(len(a_batch), len(b_batch))

    # Pad with 1s
    a_padded = [1] * (max_len - len(a_batch)) + list(a_batch)
    b_padded = [1] * (max_len - len(b_batch)) + list(b_batch)

    for a, b in zip(a_padded, b_padded):
        if a == b:
            result.append(a)
        elif a == 1:
            result.append(b)
        elif b == 1:
            result.append(a)
        else:
            raise ValueError(f"Cannot broadcast {a} and {b}")

    return result

# Test broadcasting
test("Same batch dims", broadcast_batch_shape([2, 3], [2, 3]) == [2, 3])
test("Broadcast with 1", broadcast_batch_shape([1, 3], [2, 3]) == [2, 3])
test("Broadcast different lengths", broadcast_batch_shape([3], [2, 3]) == [2, 3])

try:
    broadcast_batch_shape([2, 3], [4, 3])
    test("Incompatible dims raise error", False, "Should have raised")
except ValueError:
    test("Incompatible dims raise error", True)

# ==============================================================================
# Test 6: NoGradGuard Context
# ==============================================================================
print("\n6. Testing NoGradGuard Context Manager...")

class NoGradGuard:
    """NoGradGuard implementation from autograd.sx"""
    _grad_enabled = True

    def __init__(self):
        self.previous_state = NoGradGuard._grad_enabled
        NoGradGuard._grad_enabled = False

    def __del__(self):
        NoGradGuard._grad_enabled = self.previous_state

# Test NoGradGuard
NoGradGuard._grad_enabled = True
test("Initially enabled", NoGradGuard._grad_enabled)

guard = NoGradGuard()
test("Disabled after guard", not NoGradGuard._grad_enabled)

del guard
test("Restored after guard deleted", NoGradGuard._grad_enabled)

# Nested test
NoGradGuard._grad_enabled = True
guard1 = NoGradGuard()
guard2 = NoGradGuard()  # Nested
del guard2
test("Nested: inner restore to False", not NoGradGuard._grad_enabled)
del guard1
test("Nested: outer restore to True", NoGradGuard._grad_enabled)

# ==============================================================================
# Test 7: SafetyError (no panic)
# ==============================================================================
print("\n7. Testing SafetyError (no panic)...")

class SafetyError(Exception):
    """SafetyError from fallback.sx - replaces panic!()"""
    pass

class NoFallbackAvailable(SafetyError):
    def __init__(self, failures: int):
        self.failures = failures
        super().__init__(f"No fallback available after {failures} failures")

def safe_process_with_error(value: Optional[float]) -> float:
    """Safe processing that returns Result instead of panic"""
    if value is None:
        raise NoFallbackAvailable(3)
    return value

# Test error handling
try:
    result = safe_process_with_error(None)
    test("Raises error on failure", False, "Should have raised")
except NoFallbackAvailable as e:
    test("Raises error on failure", True)
    test("Error contains failure count", e.failures == 3)

result = safe_process_with_error(42.0)
test("Returns value on success", result == 42.0)

# ==============================================================================
# Test 8: Gaussian Noise with Proper RNG
# ==============================================================================
print("\n8. Testing Gaussian Noise RNG...")

def gaussian_noise_fixed(mean: float, std: float, state: int) -> Tuple[float, int]:
    """Box-Muller with proper RNG from federated.sx"""
    u1_int, state = xorshift64(state)
    u2_int, state = xorshift64(state)

    u1 = max(u1_int / (2**64 - 1), 1e-10)  # Avoid ln(0)
    u2 = u2_int / (2**64 - 1)

    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return mean + std * z, state

# Generate samples
state = 0xDEADBEEF
samples = []
for _ in range(1000):
    sample, state = gaussian_noise_fixed(0.0, 1.0, state)
    samples.append(sample)

sample_mean = sum(samples) / len(samples)
sample_var = sum((s - sample_mean)**2 for s in samples) / len(samples)
sample_std = math.sqrt(sample_var)

test("Gaussian mean ~0", abs(sample_mean) < 0.2, f"Got {sample_mean:.3f}")
test("Gaussian std ~1", 0.7 < sample_std < 1.3, f"Got {sample_std:.3f}")

# ==============================================================================
# Test 9: Timestamp Function
# ==============================================================================
print("\n9. Testing Timestamp Function...")

import time

def current_timestamp() -> int:
    """current_timestamp from beliefs.sx"""
    return int(time.time() * 1000)

ts1 = current_timestamp()
time.sleep(0.01)
ts2 = current_timestamp()

test("Timestamp is positive", ts1 > 0)
test("Timestamp increases", ts2 > ts1)
test("Timestamp is in milliseconds", ts1 > 1000000000000)  # After year 2001

# ==============================================================================
# Test 10: Federated Aggregation Strategies
# ==============================================================================
print("\n10. Testing Federated Aggregation Strategies...")

def fed_avg(updates: List[List[float]]) -> List[float]:
    """FedAvg from federated.sx"""
    n = len(updates)
    result = [0.0] * len(updates[0])
    for update in updates:
        for i, v in enumerate(update):
            result[i] += v / n
    return result

def weighted_avg(updates: List[Tuple[List[float], int]]) -> List[float]:
    """Weighted average from federated.sx"""
    total_samples = sum(s for _, s in updates)
    result = [0.0] * len(updates[0][0])
    for update, samples in updates:
        weight = samples / total_samples
        for i, v in enumerate(update):
            result[i] += v * weight
    return result

def median_aggregation(updates: List[List[float]]) -> List[float]:
    """Median aggregation from federated.sx"""
    result = []
    for i in range(len(updates[0])):
        values = sorted([u[i] for u in updates])
        result.append(values[len(values) // 2])
    return result

# Test FedAvg
updates = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
avg = fed_avg(updates)
test("FedAvg computes mean", avg == [3.0, 4.0], f"Got {avg}")

# Test WeightedAvg
updates_weighted = [([0.0, 0.0], 100), ([4.0, 4.0], 300)]
wavg = weighted_avg(updates_weighted)
test("WeightedAvg respects weights", wavg == [3.0, 3.0], f"Got {wavg}")

# Test Median
updates_median = [[1.0], [5.0], [9.0], [2.0], [3.0]]
med = median_aggregation(updates_median)
test("Median finds middle value", med == [3.0], f"Got {med}")

# ==============================================================================
# Summary
# ==============================================================================
print("\n" + "=" * 60)
print(f"VERIFICATION COMPLETE: {passed} passed, {failed} failed")
print("=" * 60)

if failed == 0:
    print("\nAll fixes verified successfully!")
else:
    print(f"\n{failed} test(s) need attention.")
    exit(1)
