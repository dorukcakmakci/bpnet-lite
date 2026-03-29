from __future__ import annotations

import time

import numpy
import torch

from tangermeme.predict import predict

from .bpnet import BPNet
from .logging import Logger
from .losses import MNLLLoss
from .mixture_loss_with_fourier_prior import MNLLFourierPriorLoss
from .performance import calculate_performance_measures


torch.backends.cudnn.benchmark = True


class BPNetWithFourierPrior(BPNet):
    """
    Isolated BPNet variant with MNLL + Fourier attribution prior.

    Core BPNet implementation remains untouched. This subclass overrides only
    the training objective and fit loop behavior.
    """

    def __init__(
        self,
        n_filters: int = 64,
        n_layers: int = 8,
        n_outputs: int = 2,
        n_control_tracks: int = 2,
        count_loss_weight: float = 1.0,
        profile_output_bias: bool = True,
        count_output_bias: bool = True,
        name: str | None = None,
        trimming: int | None = None,
        verbose: bool = True,
        fourier_loss_weight: float = 0.0,
        fourier_freq_limit: int = 200,
        fourier_limit_softness: float | None = 0.2,
        fourier_smooth_sigma: int = 3,
    ):
        super().__init__(
            n_filters=n_filters,
            n_layers=n_layers,
            n_outputs=n_outputs,
            n_control_tracks=n_control_tracks,
            count_loss_weight=count_loss_weight,
            profile_output_bias=profile_output_bias,
            count_output_bias=count_output_bias,
            name=name,
            trimming=trimming,
            verbose=verbose,
        )

        self.training_loss = MNLLFourierPriorLoss(
            fourier_loss_weight=fourier_loss_weight,
            fourier_freq_limit=fourier_freq_limit,
            fourier_limit_softness=fourier_limit_softness,
            fourier_smooth_sigma=fourier_smooth_sigma,
        )
        self.fourier_loss_weight = float(fourier_loss_weight)

        # Keep logging isolated to this training variant.
        self.logger = Logger(
            [
                "Epoch",
                "Iteration",
                "Training Time",
                "Validation Time",
                "Training MNLL",
                "Training Fourier",
                "Validation MNLL",
                "Validation Profile Pearson",
                "Saved?",
            ],
            verbose=verbose,
        )

    def fit(
        self,
        training_data,
        optimizer,
        scheduler=None,
        X_valid=None,
        X_ctl_valid=None,
        y_valid=None,
        max_epochs: int = 100,
        batch_size: int = 64,
        dtype: str | torch.dtype = "float32",
        device: str = "cuda",
        early_stopping: int | None = None,
    ):
        """Fit using MNLL + Fourier prior only (no count-head loss term)."""
        print(
            "Warning: BPNet and ChromBPNet models trained using bpnet-lite may "
            "underperform those trained using the official repositories. See the "
            "GitHub README for further documentation."
        )

        if X_valid is None or y_valid is None:
            raise ValueError("X_valid and y_valid must be provided for this fit variant.")

        if X_ctl_valid is not None:
            X_ctl_valid = (X_ctl_valid,)

        dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

        iteration = 0
        early_stop_count = 0
        best_loss = float("inf")
        self.logger.start()

        for epoch in range(max_epochs):
            tic = time.time()

            for data in training_data:
                X, y, labels = data[0], data[-2], data[-1]
                X_ctl = data[1].to(device) if len(data) == 4 else None

                X = X.to(device).float()
                if self.fourier_loss_weight > 0:
                    X.requires_grad_(True)

                y = y.to(device)
                labels = labels.to(device) if torch.is_tensor(labels) else torch.tensor(labels, device=device)

                optimizer.zero_grad()
                self.train()

                with torch.autocast(device_type=device, dtype=dtype):
                    y_hat_logits, _ = self(X, X_ctl)
                    training_mnll_, training_fourier_, loss, _ = self.training_loss(
                        y_true=y,
                        y_hat_logits=y_hat_logits,
                        input_seqs_bcl=X,
                        labels=labels,
                    )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                optimizer.step()

                iteration += 1

            train_time = time.time() - tic

            with torch.no_grad():
                self.eval()
                tic = time.time()

                y_hat_logits, y_hat_logcounts = predict(
                    self,
                    X_valid,
                    args=X_ctl_valid,
                    batch_size=batch_size,
                    dtype=dtype,
                    device=device,
                )

                y_hat_logps = torch.nn.functional.log_softmax(
                    y_hat_logits.reshape(y_hat_logits.shape[0], -1),
                    dim=-1,
                )
                y_valid_flat = y_valid.reshape(y_valid.shape[0], -1).to(y_hat_logps.device)
                valid_mnll = MNLLLoss(y_hat_logps, y_valid_flat).mean()

                measures = calculate_performance_measures(
                    y_hat_logits,
                    y_valid,
                    y_hat_logcounts,
                    kernel_sigma=7,
                    kernel_width=81,
                    measures=["profile_pearson"],
                )
                valid_profile_corr = numpy.nan_to_num(measures["profile_pearson"])
                valid_time = time.time() - tic

                valid_loss_value = valid_mnll.item()

                self.logger.add(
                    [
                        epoch,
                        iteration,
                        train_time,
                        valid_time,
                        training_mnll_,
                        training_fourier_,
                        valid_loss_value,
                        valid_profile_corr.mean(),
                        valid_loss_value < best_loss,
                    ]
                )

                self.logger.save(f"{self.name}.log")

                if valid_loss_value < best_loss:
                    torch.save(self, f"{self.name}.torch")
                    best_loss = valid_loss_value
                    early_stop_count = -1

            if scheduler is not None:
                scheduler.step(valid_mnll)

            early_stop_count += 1
            if early_stopping is not None and early_stop_count >= early_stopping:
                break

        torch.save(self, f"{self.name}.final.torch")


__all__ = ["BPNetWithFourierPrior"]
