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

    # Also test with different T and B to be thorough
    for T2, B2 in [(1, 1), (5, 8), (25, 64), (100, 16)]:
        torch.manual_seed(123)
        m = torch.randn(T2, B2, C)
        t = torch.randint(0, C, (B2,))

        old = torch.zeros(1)
        for s in range(T2):
            old += criterion(m[s], t)
        old = old / T2

        new = F.cross_entropy(
            m.reshape(T2 * B2, C),
            t.unsqueeze(0).expand(T2, -1).reshape(-1),
        )

        diff = abs(old.item() - new.item())
        report(f"  Non-TET T={T2}, B={B2}",
               diff < 1e-5,
               f"diff={diff:.2e}")


# ============================================================
# TEST 2: TET loss equivalence
# ============================================================
def test_tet_loss():
    """
    Old code:
        per_step_losses = []
        for step in range(T_steps):
            per_step_losses.append(criterion(mem_out[step], targets))
        per_step = torch.stack(per_step_losses)
        mean_loss = per_step.mean()
        var_loss = ((per_step - mean_loss) ** 2).mean()
        loss = mean_loss + lambda_tet * var_loss

    New code:
        per_sample = F.cross_entropy(
            mem_out.reshape(T * B, C),
            targets.unsqueeze(0).expand(T, -1).reshape(-1),
            reduction='none'
        ).reshape(T, B)
        per_step = per_sample.mean(dim=1)  # (T,)
        mean_loss = per_step.mean()
        var_loss = ((per_step - mean_loss) ** 2).mean()
        loss = mean_loss + lambda_tet * var_loss

    Math check:
    - Old per_step[t] = CE(mem_out[t], targets) with reduction='mean'
                       = (1/B) * sum_b( -log(softmax(mem_out[t,b]))[targets[b]] )
    - New per_step[t] = per_sample[t].mean(dim=1)
                       = (1/B) * sum_b( CE_individual(mem_out[t,b], targets[b]) )
    - These are IDENTICAL.
    """
    print("\n=== TEST 2: TET loss ===")

    torch.manual_seed(42)
    T, B, C = 25, 32, 50
    mem_out = torch.randn(T, B, C)
    targets = torch.randint(0, C, (B,))
    criterion = nn.CrossEntropyLoss()
    lambda_tet = 1.0

    # OLD method
    per_step_old = []
    for step in range(T):
        per_step_old.append(criterion(mem_out[step], targets))
    per_step_old = torch.stack(per_step_old)
    mean_loss_old = per_step_old.mean()
    var_loss_old = ((per_step_old - mean_loss_old) ** 2).mean()
    loss_old = mean_loss_old + lambda_tet * var_loss_old

    # NEW method
    per_sample_new = F.cross_entropy(
        mem_out.reshape(T * B, C),
        targets.unsqueeze(0).expand(T, -1).reshape(-1),
        reduction='none'
    ).reshape(T, B)
    per_step_new = per_sample_new.mean(dim=1)
    mean_loss_new = per_step_new.mean()
    var_loss_new = ((per_step_new - mean_loss_new) ** 2).mean()
    loss_new = mean_loss_new + lambda_tet * var_loss_new

    # Compare per-step values
    per_step_diff = (per_step_old - per_step_new).abs().max().item()
    report("TET per-step losses match",
           per_step_diff < 1e-6,
           f"max per-step diff={per_step_diff:.2e}")

    loss_diff = abs(loss_old.item() - loss_new.item())
    report("TET total loss matches",
           loss_diff < 1e-6,
           f"old={loss_old.item():.10f}, new={loss_new.item():.10f}, diff={loss_diff:.2e}")

    # Test with different lambda_tet
    for lam in [0.0, 0.5, 2.0, 10.0]:
        old = mean_loss_old + lam * var_loss_old
        new = mean_loss_new + lam * var_loss_new
        diff = abs(old.item() - new.item())
        report(f"  TET lambda={lam}",
               diff < 1e-6,
               f"diff={diff:.2e}")


# ============================================================
# TEST 3: Eval loss -- OLD vs NEW (CRITICAL: different semantics!)
# ============================================================
def test_eval_loss():
    """
    OLD eval code:
        loss = torch.zeros(1, device=device)
        for step in range(mem_out.shape[0]):
            loss += criterion(mem_out[step], targets)
        total_loss += loss.item()
        ...
        return total_loss / len(loader), ...

    This means: per-batch loss = sum_t[ CE(mem_out[t], targets) ]
                                = sum_t[ (1/B) * sum_b(loss_tb) ]
                                = (T/1) * (1/(T*B)) * sum_all
                                = (1/B) * sum_all

    NEW eval code:
        T, B, C = mem_out.shape
        loss = F.cross_entropy(
            mem_out.reshape(T * B, C),
            targets.unsqueeze(0).expand(T, -1).reshape(-1),
        )
        total_loss += loss.item()

    This means: per-batch loss = (1/(T*B)) * sum_all

    DIFFERENCE: Old = T * New

    The old code does NOT divide by T in eval (but the old training code DOES).
    The new code divides by T*B in both train and eval.

    IMPACT: The eval loss is used for:
    1. ReduceLROnPlateau scheduler.step(test_loss)
    2. Display only

    The SCALE of the loss changes by factor T=25.
    ReduceLROnPlateau uses a relative threshold (default factor=0.5, threshold=1e-4 in 'rel' mode).
    Since it compares (best - current) / best, a uniform scale factor cancels out.
    So the scheduler behavior is IDENTICAL (relative mode is scale-invariant).
    """
    print("\n=== TEST 3: Eval loss (CRITICAL difference check) ===")

    torch.manual_seed(42)
    T, B, C = 25, 32, 50
    mem_out = torch.randn(T, B, C)
    targets = torch.randint(0, C, (B,))
    criterion = nn.CrossEntropyLoss()

    # OLD eval: sum without dividing by T
    loss_old = torch.zeros(1)
    for step in range(T):
        loss_old += criterion(mem_out[step], targets)
    loss_old_val = loss_old.item()

    # NEW eval: F.cross_entropy with reduction='mean' over T*B
    loss_new = F.cross_entropy(
        mem_out.reshape(T * B, C),
        targets.unsqueeze(0).expand(T, -1).reshape(-1),
    )
    loss_new_val = loss_new.item()

