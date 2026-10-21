"""DecoupledTranslationLoss: L1 on (U_x, U_y) plus lambda_z times L1 on U_z."""

import torch

from src.losses import DecoupledTranslationLoss


def test_value_and_weights():
    pred = torch.tensor([[0.1, -0.2, 1.1], [0.0, 0.0, 0.9]])
    target = torch.tensor([[0.0, 0.0, 1.0], [0.05, 0.0, 1.0]])
    # sample 1: 0.1 + 0.2 + 2 * 0.1 = 0.5; sample 2: 0.05 + 0 + 2 * 0.1 = 0.25
    loss = DecoupledTranslationLoss(lambda_z=2.0)(pred, target)
    assert torch.isclose(loss, torch.tensor(0.375))


def test_zero_at_target_and_gradient_flows():
    target = torch.rand(8, 3)
    pred = target.clone().requires_grad_()
    assert DecoupledTranslationLoss()(pred, target) == 0
    (DecoupledTranslationLoss()(pred + 0.1, target)).backward()
    assert pred.grad is not None
