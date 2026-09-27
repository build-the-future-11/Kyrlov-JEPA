"""Model forward / EMA / gradient tests."""

import torch

from spectral_krylov_jepa.models.downstream import DownstreamGroundStateModel
from spectral_krylov_jepa.models.ema_utils import update_ema
from spectral_krylov_jepa.models.field_jepa import FieldJEPA
from spectral_krylov_jepa.models.krylov_jepa import KrylovJEPA
from spectral_krylov_jepa.models.operator_jepa import OperatorJEPA
from spectral_krylov_jepa.physics.grid import cell_area
from spectral_krylov_jepa.physics.grid import GridSpec


def test_field_jepa_forward_and_ema():
    model = FieldJEPA(img_size=16, size="smoke")
    v = torch.randn(2, 16, 16)
    out = model(v)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    # Online has grads; target does not
    online_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.online.parameters())
    assert online_grad
    for p in model.target.parameters():
        assert p.grad is None
        assert not p.requires_grad
    # Perturb online weights so EMA is observable (backward alone does not step weights)
    with torch.no_grad():
        for p in model.online.parameters():
            p.add_(0.1)
    before = next(model.target.parameters()).clone()
    update_ema(model.online, model.target, 0.5)
    after = next(model.target.parameters())
    assert not torch.allclose(before, after)


def test_krylov_jepa_forward():
    model = KrylovJEPA(img_size=16, size="smoke", context_steps=2)
    v = torch.randn(2, 16, 16)
    q_ctx = torch.randn(2, 2, 256)
    q_ctx = q_ctx / q_ctx.norm(dim=-1, keepdim=True)
    q_tgt = torch.randn(2, 256)
    q_tgt = q_tgt / q_tgt.norm(dim=-1, keepdim=True)
    alpha = torch.randn(2, 3)
    beta = torch.rand(2, 3) + 0.1
    out = model(v, q_ctx, q_tgt, alpha=alpha, beta=beta)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert any(p.grad is not None for p in model.potential_enc.parameters())


def test_operator_and_downstream_shapes():
    op = OperatorJEPA(img_size=16, size="smoke")
    v = torch.randn(2, 16, 16)
    q = torch.randn(2, 256)
    hq = torch.randn(2, 256)
    hq = hq / hq.norm(dim=-1, keepdim=True)
    out = op(v, q, hq)
    assert out.z_pred.shape == (2, op.embed_dim)

    g = GridSpec(n_interior=16)
    down = DownstreamGroundStateModel(img_size=16, size="smoke", cell_area=cell_area(g))
    o = down(v)
    assert o.psi.shape == (2, 16, 16)
    assert o.energy.shape == (2,)
