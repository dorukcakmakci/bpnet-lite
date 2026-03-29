# losses.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
# Adapted from code by Alex Tseng

"""
This module contains the losses used by BPNet for training.
"""

import torch

class CoarseMNLL(torch.nn.Module):
    """
    Coarse multinomial NLL with configurable binning.

    Receives base-resolution signals and performs binning internally.

    IMPORTANT: Assumes logps are log-softmax outputs (sum to 1 in probability space).
    This is guaranteed by MixtureLoss which calls torch.log_softmax(logits, dim=-1)
    before passing to this loss.

    Args:
        bin_size: int - size of bins in base pairs (e.g., 8 for 8bp bins)
        eps: float - numerical stability constant (default 1e-8)

    Forward:
        logps: (B, T, L) - base-resolution predicted log-probabilities
        counts: (B, T, L) - base-resolution ground truth counts

    Returns:
        loss: scalar tensor
        stats: dict with {"coarse_mnll": loss.detach()}
    """

    def __init__(self, bin_size: int = 8, eps: float = 1e-8):
        super().__init__()
        self.bin_size = int(bin_size)
        self.eps = float(eps)

    def forward(
        self,
        logps: torch.Tensor,      # (B, T, L) log-probs
        counts: torch.Tensor,     # (B, T, L) counts
    ):
        if logps.shape != counts.shape:
            raise ValueError(
                f"logps and counts must have same shape, got "
                f"{tuple(logps.shape)} vs {tuple(counts.shape)}"
            )

        B, T, L = logps.shape
        bin_size = self.bin_size
        num_bins = L // bin_size

        # Reshape to (B, T, num_bins, bin_size)
        logps_binned = logps.view(B, T, num_bins, bin_size)
        counts_binned = counts.view(B, T, num_bins, bin_size)

        # Sum counts within each bin: (B, T, num_bins)
        binned_counts = counts_binned.sum(dim=-1)

        # Pool log-probs using logsumexp for numerical stability
        binned_logps = torch.logsumexp(logps_binned, dim=-1)  # (B, T, num_bins)

        # Renormalize to ensure sum of probabilities = 1
        binned_logps_normalized = binned_logps - torch.logsumexp(binned_logps, dim=-1, keepdim=True)

        # Compute loss: -sum_j (binned_counts_j * log P_j)
        loss_per_example = -(binned_counts * binned_logps_normalized).sum(dim=-1)  # (B, T)
        loss = loss_per_example.mean()

        return loss


class ChIPSeqCoarseMNLL(torch.nn.Module):
    """
    Coarse multinomial NLL for ChIP-seq with strand-aware binning.

    Critical difference from CoarseMNLL above:
    - CoarseMNLL: Assumes input is already log-softmax'd, re-normalizes after binning
    - ChIPSeqCoarseMNLL: Takes raw logits, applies log-softmax across BOTH strands
                         concatenated (matching BPNet's behavior), bins WITHOUT
                         re-normalization

    This matches BPNet's strand handling where "A single log softmax is applied
    across both strands such that the logsumexp of both strands together is 0"
    (see bpnet.py lines 172-175).

    Args:
        bin_size: int - size of bins in base pairs (e.g., 20 for 20bp bins)
        eps: float - numerical stability constant (default 1e-8)

    Forward:
        logits: (B, T, L) - base-resolution raw logits (NOT log-probs)
        counts: (B, T, L) - base-resolution ground truth counts

        where T=2 for ChIP-seq (plus and minus strands)

    Returns:
        loss: scalar tensor
    """

    def __init__(self, bin_size: int = 20, eps: float = 1e-8):
        super().__init__()
        self.bin_size = int(bin_size)
        self.eps = float(eps)

    def forward(
        self,
        logits: torch.Tensor,  # (B, T, L) raw logits
        counts: torch.Tensor,  # (B, T, L) counts
    ):
        if logits.shape != counts.shape:
            raise ValueError(
                f"logits and counts must have same shape, got "
                f"{tuple(logits.shape)} vs {tuple(counts.shape)}"
            )

        B, T, L = logits.shape
        bin_size = self.bin_size

        # Validation: check divisibility
        if L % bin_size != 0:
            raise ValueError(
                f"Output length {L} must be divisible by bin_size {bin_size}. "
                f"Got L={L}, bin_size={bin_size}, remainder={L % bin_size}"
            )

        num_bins = L // bin_size

        # Step 1: Apply log-softmax across BOTH strands (flatten T*L)
        # This matches BPNet's behavior: both strands normalized together
        logits_flat = logits.reshape(B, T * L)  # (B, T*L)
        logps_flat = torch.nn.functional.log_softmax(logits_flat, dim=-1)  # (B, T*L)
        logps = logps_flat.reshape(B, T, L)  # (B, T, L) with preserved cross-strand normalization

        # Step 2: Bin the log-probabilities and counts
        logps_binned = logps.view(B, T, num_bins, bin_size)  # (B, T, num_bins, bin_size)
        counts_binned = counts.view(B, T, num_bins, bin_size)  # (B, T, num_bins, bin_size)

        # Sum counts within each bin
        binned_counts = counts_binned.sum(dim=-1)  # (B, T, num_bins)

        # Pool log-probs using logsumexp for numerical stability
        binned_logps = torch.logsumexp(logps_binned, dim=-1)  # (B, T, num_bins)

        # Step 3: CRITICAL - NO re-normalization!
        # The cross-strand probability distribution is already correct from Step 1
        # Re-normalizing would break the cross-strand normalization

        # Step 4: Compute multinomial NLL loss
        loss_per_example = -(binned_counts * binned_logps).sum(dim=(1, 2))  # (B,)
        loss = loss_per_example.mean()

        return loss
