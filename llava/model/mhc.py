import torch
import torch.nn as nn




def sinkhorn_normalize(log_W: torch.Tensor, n_iters: int = 20) -> torch.Tensor:

    W = log_W.exp()
    for _ in range(n_iters):
        W = W / (W.sum(dim=-1, keepdim=True) + 1e-8)   # row normalise
        W = W / (W.sum(dim=-2, keepdim=True) + 1e-8)   # col normalise
    return W




class mHCResidual(nn.Module):
   
    def __init__(self, n_streams: int = 2, n_iters: int = 20):
        super().__init__()
        if n_streams != 2:
            raise ValueError(
                "mHCResidual requires n_streams == 2 to keep W square "
                "(Birkhoff polytope is only defined for square matrices)."
            )

        self.N = n_streams
        self.n_iters = n_iters

        # Log-domain [2, 2] mixing matrix — Sinkhorn maps this to doubly stochastic.
        # Diagonal init (2.0) → after Sinkhorn ≈ [[0.88, 0.12], [0.12, 0.88]]
        # so each stream starts mostly aligned with one source.
        init = torch.eye(n_streams) * 2.0
        self.log_W = nn.Parameter(init + 0.01 * torch.randn(n_streams, n_streams))

        # Scalar logits for combining the two streams.
        # zeros → uniform softmax (0.5 / 0.5) at init.
        self.stream_logits = nn.Parameter(torch.zeros(n_streams))

    @property
    def mixing_matrix(self) -> torch.Tensor:
        """Doubly stochastic W projected onto the Birkhoff polytope."""
        return sinkhorn_normalize(self.log_W, self.n_iters)

    def forward(self, residual: torch.Tensor, sublayer_out: torch.Tensor) -> torch.Tensor:

        W = self.mixing_matrix                                  
        sources = torch.stack([residual, sublayer_out], dim=0)  


        streams = torch.einsum("ns,s...->n...", W, sources)     

        gates = torch.softmax(self.stream_logits, dim=0)        

        return self.N * torch.einsum("n,n...->...", gates, streams)   




if __name__ == "__main__":
    torch.manual_seed(42)
    mhc = mHCResidual(n_streams=2, n_iters=20)

    x   = torch.randn(2, 10, 512)
    f_x = torch.randn(2, 10, 512)
    out = mhc(x, f_x)

    assert out.shape == x.shape, f"Shape mismatch: {out.shape}"

    W = mhc.mixing_matrix
    row_sums = W.sum(dim=-1)
    col_sums = W.sum(dim=-2)
    gate_sum = torch.softmax(mhc.stream_logits, dim=0).sum()

    print("W (doubly stochastic):\n", W.detach().numpy())
    print("Row sums:", row_sums.detach().numpy())   # should be [1, 1]
    print("Col sums:", col_sums.detach().numpy())   # should be [1, 1]
    print("Gate sum:", gate_sum.item())             # should be 1.0

    assert torch.allclose(row_sums.detach(), torch.ones(2), atol=1e-5), f"Row sums off: {row_sums}"
    assert torch.allclose(col_sums.detach(), torch.ones(2), atol=1e-5), f"Col sums off: {col_sums}"
    assert torch.allclose(gate_sum.detach(), torch.tensor(1.0),  atol=1e-6), f"Gate sum off: {gate_sum}"
    print("mHC PASSED — output shape:", out.shape)
