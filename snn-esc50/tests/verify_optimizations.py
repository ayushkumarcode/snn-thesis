"""
Verification that performance optimizations produce IDENTICAL results.

Tests:
1. Non-TET loss: old loop vs new vectorized F.cross_entropy
2. TET loss: old loop vs new vectorized with reduction='none'
3. Eval loss: old loop (sum without /T) vs new vectorized
4. total_spikes .item() deferral correctness
5. set_to_none=True safety check
6. DataLoader num_workers/pin_memory safety analysis (printed, not testable numerically)

Run: python -m tests.verify_optimizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

PASS = 0
FAIL = 0


def report(name, passed, detail=""):
    global PASS, FAIL
    if passed:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} -- {detail}")


# ============================================================
# TEST 1: Non-TET training loss equivalence
# ============================================================
def test_non_tet_loss():
    """
    Old code:
        loss = torch.zeros(1, device=device)
        for step in range(T_steps):
            loss += criterion(mem_out[step], targets)
        loss = loss / T_steps

    New code:
        T, B, C = mem_out.shape
        loss = F.cross_entropy(
            mem_out.reshape(T * B, C),
            targets.unsqueeze(0).expand(T, -1).reshape(-1),
        )

    Math check:
    - Old: (1/T) * sum_t[ (1/B) * sum_b( -log(softmax(mem_out[t,b]))[targets[b]] ) ]
         = (1/(T*B)) * sum_t sum_b ( -log(softmax(mem_out[t,b]))[targets[b]] )
    - New: F.cross_entropy with reduction='mean' over T*B samples
         = (1/(T*B)) * sum_i ( -log(softmax(logits[i]))[labels[i]] )
         where logits[i] = mem_out[t,b] and labels[i] = targets[b] for i = t*B + b
    - These are IDENTICAL.
    """
    print("\n=== TEST 1: Non-TET training loss ===")

    torch.manual_seed(42)
    T, B, C = 25, 32, 50  # timesteps, batch, classes
    mem_out = torch.randn(T, B, C)
    targets = torch.randint(0, C, (B,))
    criterion = nn.CrossEntropyLoss()

    # OLD method
    device = mem_out.device
    loss_old = torch.zeros(1, device=device)
    for step in range(T):
        loss_old += criterion(mem_out[step], targets)
    loss_old = loss_old / T

    # NEW method
    loss_new = F.cross_entropy(
        mem_out.reshape(T * B, C),
        targets.unsqueeze(0).expand(T, -1).reshape(-1),
    )

    diff = abs(loss_old.item() - loss_new.item())
    report("Non-TET loss: old loop vs vectorized",
           diff < 1e-6,
           f"old={loss_old.item():.10f}, new={loss_new.item():.10f}, diff={diff:.2e}")
