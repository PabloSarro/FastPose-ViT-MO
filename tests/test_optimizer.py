"""SingleDeviceMuonWithAuxAdam: the batched step (foreach + one Newton-Schulz per shape) gives the
same parameters as the original per-parameter loop over muon_update / adam_update.

- Optimizer state (momentum, Adam moments): bit-identical.
- Parameters: bit-identical on the CPU. On the GPU, Muon is bit-identical too, but the final Adam
  `p += -lr * update` of the multi-tensor kernel may round differently by one float32 ulp
  (measured 7e-9); the test allows float32 rounding there and bf16 rounding for Muon.
"""

import pytest
import torch

from src.optimizer import SingleDeviceMuonWithAuxAdam, adam_update, muon_update

# The kinds of parameters FastPoseViT has: square, tall and wide matrices, a conv filter, 3D
# tokens, and vectors; several of the same shape so that the batched path is used
SHAPES = [(16, 16)] * 3 + [(48, 16), (16, 48)] * 2 + [(16, 3, 4, 4), (4, 1, 16), (1, 9, 16), (5, 16)]
VECTOR_SHAPES = [(16,), (48,), (16,), (5,)]
STEPS = 4


def reference_step(params, state, lr_muon, lr_adam, momentum=0.95, betas=(0.9, 0.95), eps=1e-10):
    """The original SingleDeviceMuonWithAuxAdam.step, one parameter at a time."""
    for p in params["muon"]:
        st = state.setdefault(p, {"momentum_buffer": torch.zeros_like(p)})
        update = muon_update(p.grad, st["momentum_buffer"], beta=momentum)
        p.add_(update.reshape(p.shape), alpha=-lr_muon)
    for p in params["adam"]:
        st = state.setdefault(
            p, {"exp_avg": torch.zeros_like(p), "exp_avg_sq": torch.zeros_like(p), "step": 0}
        )
        st["step"] += 1
        update = adam_update(p.grad, st["exp_avg"], st["exp_avg_sq"], st["step"], betas, eps)
        p.add_(update, alpha=-lr_adam)


def _params(device):
    g = torch.Generator().manual_seed(0)
    make = lambda s: torch.randn(s, generator=g).to(device)  # noqa: E731
    return [make(s) for s in SHAPES], [make(s) for s in VECTOR_SHAPES]


def _run(device):
    muon0, adam0 = _params(device)
    ref = {"muon": [p.clone() for p in muon0], "adam": [p.clone() for p in adam0]}
    new_muon = [p.clone().requires_grad_(False) for p in muon0]
    new_adam = [p.clone() for p in adam0]
    opt = SingleDeviceMuonWithAuxAdam(
        [
            {"params": new_muon, "use_muon": True, "lr": 0.02},
            {"params": new_adam, "use_muon": False, "lr": 0.002, "betas": (0.9, 0.95)},
        ]
    )
    ref_state = {}
    g = torch.Generator().manual_seed(1)
    for _ in range(STEPS):
        for r, n in zip(ref["muon"] + ref["adam"], new_muon + new_adam):
            grad = torch.randn(r.shape, generator=g).to(device)
            r.grad, n.grad = grad.clone(), grad.clone()
        reference_step(ref, ref_state, 0.02, 0.002)
        opt.step()
    return ref, new_muon, new_adam, ref_state, opt


def _check(device, muon_atol, adam_exact):
    ref, new_muon, new_adam, ref_state, opt = _run(device)
    for r, n in zip(ref["adam"], new_adam):
        if adam_exact:
            assert torch.equal(r, n)
        else:
            torch.testing.assert_close(n, r, rtol=1e-6, atol=1e-7)
        for k in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(ref_state[r][k], opt.state[n][k])
        assert opt.state[n]["step"] == ref_state[r]["step"] == STEPS
    for r, n in zip(ref["muon"], new_muon):
        torch.testing.assert_close(n, r, rtol=0, atol=muon_atol)
        torch.testing.assert_close(
            opt.state[n]["momentum_buffer"], ref_state[r]["momentum_buffer"], rtol=0, atol=0
        )


def test_batched_step_matches_per_parameter_loop_cpu():
    _check(torch.device("cpu"), muon_atol=0, adam_exact=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_batched_step_matches_per_parameter_loop_gpu():
    # 4 steps of lr 0.02 with bf16 Newton-Schulz: allow one bf16 rounding step per update
    _check(torch.device("cuda"), muon_atol=4 * 0.02 * 2**-7, adam_exact=False)


def test_state_dict_format_unchanged():
    _, new_muon, new_adam, _, opt = _run(torch.device("cpu"))
    sd = opt.state_dict()
    keys = {frozenset(s.keys()) for s in sd["state"].values()}
    assert keys == {frozenset({"momentum_buffer"}), frozenset({"exp_avg", "exp_avg_sq", "step"})}


def _newton_schulz_float64(G, steps=5):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.double()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return X.mT if G.size(-2) > G.size(-1) else X


def _newton_schulz_unfused(G, steps=5):
    """The previous bf16 implementation (b*A + c*A@A and a*X + B@X as separate operations)."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return X.mT if G.size(-2) > G.size(-1) else X


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "shape", [(64, 64), (3, 96, 32), (2, 32, 96), (4, 1, 64), (2, 4, 1, 64), (2, 256, 256)]
)
def test_newton_schulz_at_least_as_accurate_as_before(shape, device):
    # The fused multiply-adds round once per product instead of three times: compared with the
    # exact (float64) iteration, the error must not grow (measured on the GPU with the ViT-B
    # shapes: 1.1 % vs 2.3 % relative Frobenius error)
    from src.optimizer import zeropower_via_newtonschulz5

    G = torch.randn(shape, generator=torch.Generator().manual_seed(0)).to(device)
    X = zeropower_via_newtonschulz5(G, steps=5)
    assert X.shape == G.shape and X.dtype == torch.bfloat16
    ref = _newton_schulz_float64(G)
    err = ((X.double() - ref).norm() / ref.norm()).item()
    err_before = ((_newton_schulz_unfused(G).double() - ref).norm() / ref.norm()).item()
    assert err <= 1.05 * err_before and err < 3e-2, (err, err_before)
