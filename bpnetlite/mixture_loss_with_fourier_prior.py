from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn

from .losses import MNLLLoss


def _smooth_tensor_1d(input_tensor: torch.Tensor, smooth_sigma: int) -> torch.Tensor:
    """Smooth a (B, L) tensor along the sequence axis using a Gaussian kernel."""
    if smooth_sigma == 0:
        return input_tensor

    sigma = int(smooth_sigma)
    kernel_size = 1 + (2 * sigma)
    base = np.zeros(kernel_size, dtype=np.float32)
    base[sigma] = 1.0
    kernel = scipy.ndimage.gaussian_filter(base, sigma=sigma, truncate=1)
    kernel_t = torch.tensor(kernel, dtype=input_tensor.dtype, device=input_tensor.device)

    x = input_tensor.unsqueeze(1)  # (B, 1, L)
    w = kernel_t.unsqueeze(0).unsqueeze(0)  # (1, 1, K)
    y = torch.nn.functional.conv1d(x, w, padding=sigma)
    return y.squeeze(1)


class FourierAttributionPriorLoss(nn.Module):
    """
    Fourier attribution prior operating on gradient×input attributions.

    Expects input_grads in shape (B, L, D), where D is input channels (DNA=4).
    """

    def __init__(
        self,
        *,
        freq_limit: int = 200,
        limit_softness: Optional[float] = 0.2,
        smooth_sigma: int = 3,
    ):
        super().__init__()
        self.freq_limit = int(freq_limit)
        self.limit_softness = limit_softness
        self.smooth_sigma = int(smooth_sigma)

    def forward(
        self,
        input_grads: torch.Tensor,
        status: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if input_grads.dim() != 3:
            raise ValueError(
                f"input_grads must have shape (B, L, D), got {tuple(input_grads.shape)}"
            )

        device = input_grads.device
        dtype = input_grads.dtype
        stats: Dict[str, torch.Tensor] = {}

        # (B, L, D) -> (B, L)
        abs_grads = torch.sum(torch.abs(input_grads), dim=2)
        grads_smooth = _smooth_tensor_1d(abs_grads, self.smooth_sigma)

        if status is not None:
            pos_mask = (status == 1)
            pos_grads = grads_smooth[pos_mask]
        else:
            pos_grads = grads_smooth

        if pos_grads.numel() == 0:
            zero = torch.zeros((), device=device, dtype=dtype)
            stats["fourier/loss"] = zero.detach()
            stats["fourier/num_positives"] = torch.tensor(0, device=device)
            stats["fourier/mean_score"] = zero.detach()
            return zero, stats

        pos_fft = torch.fft.rfft(pos_grads, dim=1)
        pos_mags = torch.abs(pos_fft)

        mag_sum = torch.sum(pos_mags, dim=1, keepdim=True)
        mag_sum = torch.where(mag_sum == 0, torch.ones_like(mag_sum), mag_sum)
        pos_mags_norm = pos_mags / mag_sum

        # Drop DC component.
        pos_mags_norm = pos_mags_norm[:, 1:]
        num_freqs = pos_mags_norm.size(1)

        weights = torch.ones(num_freqs, device=device, dtype=pos_mags_norm.dtype)
        if self.freq_limit < num_freqs:
            if self.limit_softness is None:
                weights[self.freq_limit:] = 0.0
            else:
                decay_len = num_freqs - self.freq_limit
                x = torch.arange(1, decay_len + 1, device=device, dtype=weights.dtype)
                weights[self.freq_limit:] = 1.0 / (1.0 + torch.pow(x, self.limit_softness))

        score = torch.sum(pos_mags_norm * weights.unsqueeze(0), dim=1)
        loss = torch.mean(1.0 - score)

        stats["fourier/loss"] = loss.detach()
        stats["fourier/num_positives"] = torch.tensor(pos_grads.size(0), device=device)
        stats["fourier/mean_score"] = torch.mean(score).detach()
        return loss, stats


def _gradient_input_for_profile(input_seqs_bcl: torch.Tensor, profile_logits: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient×input attributions for profile logits.

    input_seqs_bcl: (B, 4, L_dna)
    profile_logits: (B, T, L_profile)
    returns: (B, L_dna, 4)
    """
    norm_logits = profile_logits - torch.mean(profile_logits, dim=-1, keepdim=True)
    probs = torch.softmax(profile_logits, dim=-1).detach()
    weighted_logits = norm_logits * probs

    input_grads, = torch.autograd.grad(
        outputs=weighted_logits,
        inputs=input_seqs_bcl,
        grad_outputs=torch.ones_like(weighted_logits),
        retain_graph=True,
        create_graph=True,
    )

    input_grads = input_grads * input_seqs_bcl
    return input_grads.transpose(1, 2).contiguous()


class MNLLFourierPriorLoss(nn.Module):
    """Total loss = MNLL + lambda * Fourier attribution prior."""

    def __init__(
        self,
        *,
        fourier_loss_weight: float = 0.0,
        fourier_freq_limit: int = 200,
        fourier_limit_softness: Optional[float] = 0.2,
        fourier_smooth_sigma: int = 3,
    ):
        super().__init__()
        self.fourier_loss_weight = float(fourier_loss_weight)
        self.fourier_prior = FourierAttributionPriorLoss(
            freq_limit=fourier_freq_limit,
            limit_softness=fourier_limit_softness,
            smooth_sigma=fourier_smooth_sigma,
        )

    def forward(
        self,
        y_true: torch.Tensor,
        y_hat_logits: torch.Tensor,
        input_seqs_bcl: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        y_true: (B, T, L)
        y_hat_logits: (B, T, L)
        input_seqs_bcl: (B, 4, L_dna), must require grad if Fourier enabled
        labels: (B,), 1 for peaks and 0 for negatives (optional)
        """
        device = y_hat_logits.device
        dtype = y_hat_logits.dtype

        y_logps = torch.nn.functional.log_softmax(
            y_hat_logits.reshape(y_hat_logits.shape[0], -1), dim=-1
        )
        y_flat = y_true.reshape(y_true.shape[0], -1)

        if labels is not None:
            labels = labels.to(device)
            pos_mask = labels == 1
            if torch.any(pos_mask):
                mnll_loss = MNLLLoss(y_logps[pos_mask], y_flat[pos_mask]).mean()
            else:
                mnll_loss = torch.zeros((), device=device, dtype=dtype)
        else:
            mnll_loss = MNLLLoss(y_logps, y_flat).mean()

        fourier_loss = torch.zeros((), device=device, dtype=dtype)
        fourier_stats: Dict[str, torch.Tensor] = {
            "fourier/loss": fourier_loss.detach(),
            "fourier/num_positives": torch.tensor(0, device=device),
            "fourier/mean_score": torch.zeros((), device=device, dtype=dtype),
        }

        if self.fourier_loss_weight > 0:
            if not input_seqs_bcl.requires_grad:
                raise ValueError(
                    "input_seqs_bcl must require gradients when Fourier prior is enabled."
                )

            grad_x = _gradient_input_for_profile(input_seqs_bcl, y_hat_logits)
            # Keep Fourier path in float32 for numerical stability under autocast.
            grad_x = grad_x.float()
            status = labels if labels is not None else None
            fourier_loss, fourier_stats = self.fourier_prior(grad_x, status=status)
            fourier_loss = fourier_loss.to(dtype=mnll_loss.dtype)

        total_loss = mnll_loss + (self.fourier_loss_weight * fourier_loss)

        stats: Dict[str, torch.Tensor] = {
            "mnll_loss": mnll_loss.detach(),
            "fourier_loss": fourier_loss.detach(),
            "loss_total": total_loss.detach(),
        }
        stats.update(fourier_stats)

        return mnll_loss.item(), fourier_loss.item(), total_loss, stats


__all__ = [
    "FourierAttributionPriorLoss",
    "MNLLFourierPriorLoss",
]
