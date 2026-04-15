import torch
import torch.nn as nn


def sinkhorn_knopp(W: torch.Tensor, n_iters: int = 50) -> torch.Tensor:
    """Project W onto Birkhoff Polytope (doubly stochastic matrices)."""
    W = W.abs() + 1e-8
    for _ in range(n_iters):
        W = W / W.sum(dim=-1, keepdim=True)
        W = W / W.sum(dim=-2, keepdim=True)
    return W


class mHCResidual(nn.Module):
    def __init__(self, n_streams: int = 2, n_iters: int = 50):
        super().__init__()
        self.N = n_streams
        self.n_iters = n_iters
        # Initialize close to identity so early training is stable
        self.W_raw = nn.Parameter(
            torch.eye(n_streams) * 2.0 + 0.01 * torch.randn(n_streams, n_streams)
        )

    def forward(self, residual: torch.Tensor, sublayer_out: torch.Tensor) -> torch.Tensor:
        W = sinkhorn_knopp(self.W_raw, self.n_iters)
        stream_0 = W[0, 0] * residual + W[0, 1] * sublayer_out
        stream_1 = W[1, 0] * residual + W[1, 1] * sublayer_out
        return stream_0 + stream_1


if __name__ == "__main__":
    torch.manual_seed(42)
    mhc = mHCResidual(n_streams=2)
    x = torch.randn(2, 10, 512)
    out = mhc(x, torch.randn(2, 10, 512))
    assert out.shape == x.shape, f"Shape mismatch: {out.shape}"

    W = sinkhorn_knopp(mhc.W_raw)
    row_sums = W.sum(-1)
    col_sums = W.sum(-2)
    print("Row sums:", row_sums.detach().numpy())   # should be [1.0, 1.0]
    print("Col sums:", col_sums.detach().numpy())   # should be [1.0, 1.0]

    assert torch.allclose(row_sums.detach(), torch.ones(2), atol=5e-3), f"Row sums off: {row_sums}"
    assert torch.allclose(col_sums.detach(), torch.ones(2), atol=5e-3), f"Col sums off: {col_sums}"
    print("mHC PASSED — output shape:", out.shape)
    print("W (doubly stochastic):\n", W.detach().numpy())