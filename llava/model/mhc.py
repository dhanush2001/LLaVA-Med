import torch
import torch.nn as nn


def row_stochastic(W: torch.Tensor) -> torch.Tensor:
    """Normalize each row into a probability distribution."""
    return torch.softmax(W, dim=-1)


class mHCResidual(nn.Module):
    def __init__(self, n_streams: int = 2, n_iters: int = 50):
        super().__init__()
        if n_streams < 1:
            raise ValueError("n_streams must be >= 1")

        self.N = n_streams
        self.n_iters = n_iters  # kept for backward compatibility with callers

        # Each stream mixes two sources: residual and sublayer output.
        init = torch.zeros(n_streams, 2)
        init[:, 0] = 2.0  # start closer to residual connection for stability
        self.W_raw = nn.Parameter(init + 0.01 * torch.randn(n_streams, 2))

        # Learn how to combine the stream outputs into one tensor.
        self.output_logits = nn.Parameter(torch.zeros(n_streams))

    def forward(self, residual: torch.Tensor, sublayer_out: torch.Tensor) -> torch.Tensor:
        mix_weights = row_stochastic(self.W_raw)  # [N, 2]
        sources = torch.stack((residual, sublayer_out), dim=0)  # [2, ...]
        mixed_streams = torch.einsum("ns,s...->n...", mix_weights, sources)  # [N, ...]

        stream_gates = torch.softmax(self.output_logits, dim=0)  # [N]
        return torch.einsum("n,n...->...", stream_gates, mixed_streams)


if __name__ == "__main__":
    torch.manual_seed(42)
    mhc = mHCResidual(n_streams=2)
    x = torch.randn(2, 10, 512)
    out = mhc(x, torch.randn(2, 10, 512))
    assert out.shape == x.shape, f"Shape mismatch: {out.shape}"

    W = row_stochastic(mhc.W_raw)
    row_sums = W.sum(-1)
    gate_sums = torch.softmax(mhc.output_logits, dim=0).sum()
    print("Row sums:", row_sums.detach().numpy())   # should be [1.0, ..., 1.0]
    print("Gate sum:", gate_sums.item())            # should be 1.0

    assert torch.allclose(row_sums.detach(), torch.ones_like(row_sums), atol=5e-3), f"Row sums off: {row_sums}"
    assert torch.allclose(gate_sums.detach(), torch.tensor(1.0), atol=1e-6), f"Gate sum off: {gate_sums}"
    print("mHC PASSED — output shape:", out.shape)
    print("W (row-stochastic):\n", W.detach().numpy())