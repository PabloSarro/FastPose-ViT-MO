# Custom optimizer implementations
# Inspired by https://github.com/KellerJordan/Muon/blob/master/muon.py

import torch


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Compute orthogonalization of G using Newton-Schulz iteration.

    Uses a quintic iteration with coefficients selected to maximize the slope at zero.
    This iteration produces an approximation to UV^T where USV^T = G is the SVD.

    Args:
        G: Input tensor to orthogonalize, must have ndim >= 2.
        steps: Number of Newton-Schulz iterations to perform.

    Returns:
        Orthogonalized tensor in bfloat16 format.
    """
    assert (
        G.ndim >= 2
    )  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
) -> torch.Tensor:
    """Compute Muon optimizer update with orthogonalization.

    Args:
        grad: Gradient tensor.
        momentum: Momentum buffer tensor (updated in-place).
        beta: Momentum coefficient.
        ns_steps: Number of Newton-Schulz iterations.
        nesterov: Whether to use Nesterov momentum.

    Returns:
        Orthogonalized update tensor.
    """
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


def adam_update(
    grad: torch.Tensor,
    buf1: torch.Tensor,
    buf2: torch.Tensor,
    step: int,
    betas: tuple[float, float],
    eps: float,
) -> torch.Tensor:
    """Compute Adam optimizer update.

    Args:
        grad: Gradient tensor.
        buf1: First moment buffer (updated in-place).
        buf2: Second moment buffer (updated in-place).
        step: Current optimization step.
        betas: Beta coefficients for exponential moving averages.
        eps: Epsilon for numerical stability.

    Returns:
        Adam update tensor.
    """
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    MuonWithAuxAdam optimizer without weight decay.

    This optimizer combines Muon (for matrix parameters) with Adam (for other parameters).
    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. For efficient orthogonalization we use a Newton-Schulz iteration, which has the
    advantage that it can be stably run in bfloat16 on the GPU.

    Muon should only be used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases should be optimized using a standard method such as Adam.
    Hidden convolutional weights can be trained using Muon by viewing them as 2D and then
    collapsing their last 3 dimensions.
    """

    def __init__(self, param_groups: list[dict]) -> None:
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # Muon defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                assert set(group.keys()) == set(["params", "lr", "momentum", "use_muon"])
            else:
                # Adam defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None) -> torch.Tensor | None:
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss (optional).

        Returns:
            The loss value if closure is provided, otherwise None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.add_(update, alpha=-group["lr"])

        return loss
