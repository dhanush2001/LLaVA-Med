import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Birkhoff-polytope projection via differentiable Sinkhorn-Knopp
# ---------------------------------------------------------------------------

def sinkhorn_normalize(log_W: torch.Tensor, n_iters: int = 20) -> torch.Tensor:
    """
    Project a square log-domain matrix onto the Birkhoff polytope
    (set of doubly stochastic matrices) via Sinkhorn-Knopp normalization.

    Operates entirely in probability space (after exp) with alternating
    row / column normalization.  Gradients flow back through all iterations.

    Args:
        log_W:  [n, n] unconstrained parameter tensor (log-domain)
        n_iters: number of alternating normalization steps (20 is enough for
                 n=2; increase for larger n)

    Returns:
        W: [n, n] doubly stochastic matrix  (row sums = col sums = 1)
    """
    W = log_W.exp()
    for _ in range(n_iters):
        W = W / (W.sum(dim=-1, keepdim=True) + 1e-8)   # row normalise
        W = W / (W.sum(dim=-2, keepdim=True) + 1e-8)   # col normalise
    return W


# ---------------------------------------------------------------------------
# mHC residual module
# ---------------------------------------------------------------------------

class mHCResidual(nn.Module):
    """
    Multi-Head Connection (mHC) residual — drop-in replacement for the
    standard  ``y = residual + sublayer_out``  in transformer residual paths.

    Architecture (DeepSeek mHC, §3-4)
    ----------------------------------
    Given two sources  s0 = residual,  s1 = sublayer_out  (both [..., D]):

      1.  Mixing matrix  W ∈ R^{2×2}  is projected onto the Birkhoff polytope
          via Sinkhorn-Knopp so that W is doubly stochastic:
              row sums  = 1  (each stream draws a proper distribution over sources)
              col sums  = 1  (each source is "consumed" in a balanced way)

      2.  N=2 streams are formed:
              stream_i = W[i,0] * residual + W[i,1] * sublayer_out

      3.  Streams are blended by a learned softmax gate:
              output = Σ_i  gate_i * stream_i

    The doubly stochastic constraint is the key inductive bias: it enforces
    that neither source is ignored and information flow is conserved across
    streams.  The gate then selects which mixture is most useful.

    Initialization
    --------------
    log_W  starts near the identity permutation matrix (diagonal = 2.0 in
    log-space ≈ [[0.88, 0.12], [0.12, 0.88]] after Sinkhorn), so the module
    behaves close to an equal-weight residual at the start of training.
    stream_logits starts at zero → uniform gate → both streams weighted 0.5.

    Args:
        n_streams: must be 2 (equals the number of sources so W is square)
        n_iters:   Sinkhorn iterations for Birkhoff projection (default 20)
    """

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
        """
        Args:
            residual:     [..., D]  pre-sublayer hidden states
            sublayer_out: [..., D]  output of the sublayer (attn or MLP)

        Returns:
            output: [..., D]  mixed result replacing ``residual + sublayer_out``
        """
        W = self.mixing_matrix                                  # [2, 2]  doubly stochastic
        sources = torch.stack([residual, sublayer_out], dim=0)  # [2, ..., D]

        # streams[i] = W[i,0]*residual + W[i,1]*sublayer_out
        streams = torch.einsum("ns,s...->n...", W, sources)     # [2, ..., D]

        # Weighted combination of streams
        gates = torch.softmax(self.stream_logits, dim=0)        # [2]
        return torch.einsum("n,n...->...", gates, streams)      # [..., D]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

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
