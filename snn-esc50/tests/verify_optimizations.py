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

    ratio = loss_old_val / loss_new_val
    report(f"Eval loss ratio = T = {T}",
           abs(ratio - T) < 1e-4,
           f"ratio={ratio:.6f}, expected={T}")

    print(f"    Old eval loss: {loss_old_val:.6f}")
    print(f"    New eval loss: {loss_new_val:.6f}")
    print(f"    Ratio (old/new): {ratio:.6f}")
    print(f"    Expected ratio: {T} (T timesteps)")
    print(f"    >>> EVAL LOSS SCALE CHANGED by factor {T}x <<<")

    # Verify ReduceLROnPlateau is scale-invariant
    # Simulate two runs with different scale
    print("\n    Verifying ReduceLROnPlateau scale-invariance...")
    losses_raw = [5.0, 4.8, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5]

    # Old scale (T * raw)
    opt1 = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
    sched1 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt1, mode="min", factor=0.5, patience=5)
    lrs1 = []
    for l in losses_raw:
        sched1.step(l * T)
        lrs1.append(opt1.param_groups[0]['lr'])

    # New scale (raw)
    opt2 = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
    sched2 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2, mode="min", factor=0.5, patience=5)
    lrs2 = []
    for l in losses_raw:
        sched2.step(l)
        lrs2.append(opt2.param_groups[0]['lr'])

    lrs_match = all(abs(a - b) < 1e-10 for a, b in zip(lrs1, lrs2))
    report("ReduceLROnPlateau identical with scaled loss",
           lrs_match,
           f"lrs1={lrs1}, lrs2={lrs2}")

    # ALSO check: does the old TRAINING code match the new TRAINING code?
    # Old training: sum / T, New training: mean over T*B
    # Already tested in TEST 1. But let's confirm they're the same.
    train_old = loss_old_val / T
    train_new = loss_new_val
    train_diff = abs(train_old - train_new)
    report("Training loss consistent (old sum/T == new mean)",
           train_diff < 1e-5,
           f"old_train={train_old:.10f}, new_train={train_new:.10f}, diff={train_diff:.2e}")

    # Flag: old code was INCONSISTENT between train and eval!
    print(f"\n    NOTE: Old code had train loss = sum/T = {train_old:.6f}")
    print(f"          Old code had eval loss  = sum   = {loss_old_val:.6f}")
    print(f"          Old eval was {T}x larger than old train for same data!")
    print(f"          New code is CONSISTENT: train and eval both use mean = {loss_new_val:.6f}")
    print(f"          >>> The old code had a TRAIN/EVAL INCONSISTENCY that the new code FIXES <<<")


# ============================================================
# TEST 4: .item() deferral
# ============================================================
def test_item_deferral():
    """
    Old: total_spikes += spk4.sum().item()  (adds Python float each step)
    New: total_spikes = total_spikes + spk4.sum()  (accumulates on GPU)
         ... then total_spikes.item() at end

    Edge case: total_spikes starts as 0 (Python int).
    First iteration: 0 + spk4.sum() = tensor (int + tensor = tensor). OK.
    Subsequent: tensor + tensor. OK.
    """
    print("\n=== TEST 4: .item() deferral ===")

    torch.manual_seed(42)

    # Simulate old approach
    total_old = 0
    tensors = [torch.randn(32, 50) for _ in range(25)]  # 25 timesteps

    for t in tensors:
        total_old += t.sum().item()

    # Simulate new approach
    total_new = 0
    for t in tensors:
        total_new = total_new + t.sum()

    # Convert at end
    total_new_val = total_new.item() if isinstance(total_new, torch.Tensor) else total_new

    diff = abs(total_old - total_new_val)
    report("Spike count: deferred .item()",
           diff < 1e-3,  # Allow small floating point difference from accumulation order
           f"old={total_old:.6f}, new={total_new_val:.6f}, diff={diff:.2e}")

    # Edge case: what if all spikes are zero?
    total_zero = 0
    for _ in range(25):
        total_zero = total_zero + torch.zeros(32, 50).sum()
    val = total_zero.item() if isinstance(total_zero, torch.Tensor) else total_zero
    report("Spike count: all zeros edge case",
           val == 0.0,
           f"val={val}")

    # Edge case: int 0 + tensor
    x = 0
    x = x + torch.tensor(5.0)
    report("int 0 + tensor = tensor",
           isinstance(x, torch.Tensor) and x.item() == 5.0,
           f"type={type(x)}, val={x}")


# ============================================================
# TEST 5: set_to_none=True safety
# ============================================================
def test_set_to_none():
    """
    optimizer.zero_grad(set_to_none=True) sets .grad to None instead of zero tensor.
    This is safe for Adam and standard training (no gradient accumulation).
    PyTorch treats None grad as zero during optimizer.step().
    """
    print("\n=== TEST 5: set_to_none=True ===")

    torch.manual_seed(42)
    # Create a simple model and compare training with both approaches
    model1 = nn.Linear(10, 5)
    model2 = nn.Linear(10, 5)
    model2.load_state_dict(model1.state_dict())

    opt1 = torch.optim.Adam(model1.parameters(), lr=1e-3, weight_decay=1e-4)
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3, weight_decay=1e-4)

    data = torch.randn(8, 10)
    target = torch.randint(0, 5, (8,))

    # Train 10 steps with zero_grad() vs zero_grad(set_to_none=True)
    for _ in range(10):
        opt1.zero_grad()
        loss1 = F.cross_entropy(model1(data), target)
        loss1.backward()
        opt1.step()

        opt2.zero_grad(set_to_none=True)
        loss2 = F.cross_entropy(model2(data), target)
        loss2.backward()
        opt2.step()

    # Compare final weights
    w1 = model1.weight.data
    w2 = model2.weight.data
    max_diff = (w1 - w2).abs().max().item()
    report("set_to_none=True produces identical weights (Adam, 10 steps)",
           max_diff < 1e-6,
           f"max weight diff={max_diff:.2e}")


# ============================================================
# TEST 6: BF16 autocast analysis (informational)
# ============================================================
def test_bf16_analysis():
    """
    BF16 has 8 exponent bits (same as FP32) but only 7 mantissa bits (vs 23).
    This means ~3 decimal digits of precision vs ~7 for FP32.

    Risks:
    - LIF membrane potentials: typically O(1), safe
    - Spike thresholds: typically 1.0, safe
    - Cross-entropy loss: gradient magnitudes can vary, but autocast handles this
    - BN running stats: accumulated in FP32 by default even under autocast

    Key safety: autocast only applies to matmuls and convolutions in BF16.
    Loss computation, BN stats, and optimizer step remain FP32.
    The `enabled=use_amp` guard ensures CPU/MPS remain FP32.
    """
    print("\n=== TEST 6: BF16 autocast analysis ===")

    # Test that autocast doesn't affect loss computation
    torch.manual_seed(42)
    T, B, C = 25, 32, 50
    mem_out = torch.randn(T, B, C)
    targets = torch.randint(0, C, (B,))

    # Without autocast
    loss_fp32 = F.cross_entropy(
        mem_out.reshape(T * B, C),
        targets.unsqueeze(0).expand(T, -1).reshape(-1),
    )

    # Note: on CPU, autocast to bfloat16 may not work on all ops.
    # We test that the guard `enabled=use_amp` (False on CPU) means no change.
    use_amp = False  # Simulating CPU/MPS
    with torch.amp.autocast('cpu', dtype=torch.bfloat16, enabled=use_amp):
        loss_no_amp = F.cross_entropy(
            mem_out.reshape(T * B, C),
            targets.unsqueeze(0).expand(T, -1).reshape(-1),
        )

    report("BF16 disabled on CPU (use_amp=False)",
           loss_fp32.item() == loss_no_amp.item(),
           f"fp32={loss_fp32.item()}, no_amp={loss_no_amp.item()}")

    # Test BF16 precision for typical membrane potential values
    mem_typical = torch.tensor([0.0, 0.5, 1.0, -0.5, 1.5, 0.001])
    mem_bf16 = mem_typical.to(torch.bfloat16).to(torch.float32)
    max_err = (mem_typical - mem_bf16).abs().max().item()
    report(f"BF16 rounding error on typical membrane values",
           max_err < 0.01,
           f"max_err={max_err:.6f}")
    print(f"    FP32:  {mem_typical.tolist()}")
    print(f"    BF16:  {mem_bf16.tolist()}")
    print(f"    NOTE: BF16 is standard for A100 training.")
    print(f"    NOTE: use_amp=False on CPU/MPS, so local runs are FP32 (bit-identical).")


# ============================================================
# TEST 7: Gradient equivalence through vectorized loss
# ============================================================
def test_gradient_equivalence():
    """
    The most critical test: do the OLD and NEW loss computations produce
    identical gradients through the model? Even if loss values match,
    gradient computation could differ due to different computational graphs.
    """
    print("\n=== TEST 7: Gradient equivalence ===")

    torch.manual_seed(42)
    T, B, C = 25, 32, 50
