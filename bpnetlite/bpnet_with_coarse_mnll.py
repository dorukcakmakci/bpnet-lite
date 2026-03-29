from __future__ import annotations

import time

import numpy
import torch

from tangermeme.predict import predict

from .bpnet import BPNet
from .logging import Logger
from .low_res_losses import ChIPSeqCoarseMNLL
from .performance import calculate_performance_measures


torch.backends.cudnn.benchmark = True


class BPNetWithCoarseMNLL(BPNet):
    """
    BPNet variant with CoarseMNLL loss for low-resolution training.

    Uses binned multinomial NLL instead of base-resolution MNLL. This allows
    training at coarser resolutions (e.g., 20bp bins) while still predicting
    at base resolution (1bp).

    Key differences from base BPNet:
    - Uses ChIPSeqCoarseMNLL loss (binned MNLL) instead of standard MNLL
    - No count-head loss (profile-only training)
    - Configurable bin size for resolution control

    The coarse MNLL loss bins base-resolution predictions (e.g., 1000bp) into
    coarser bins (e.g., 50 bins of 20bp each) and computes multinomial NLL on
    the binned predictions. This provides more stable training by reducing
    sensitivity to position-level noise.
    """

    def __init__(
        self,
        n_filters: int = 64,
        n_layers: int = 8,
        n_outputs: int = 2,
        n_control_tracks: int = 2,
        profile_output_bias: bool = True,
        count_output_bias: bool = True,
        name: str | None = None,
        trimming: int | None = None,
        verbose: bool = True,
        coarse_bin_size: int = 20,
    ):
        """
        Initialize BPNetWithCoarseMNLL.

        Parameters
        ----------
        n_filters : int, optional
            The number of filters in each convolutional layer. Default is 64.

        n_layers : int, optional
            The number of dilated residual convolutional layers. Default is 8.

        n_outputs : int, optional
            The number of output profiles. For ChIP-seq, this is typically 2
            (plus and minus strands). Default is 2.

        n_control_tracks : int, optional
            The number of control tracks to include. For ChIP-seq with strand-
            specific controls, this is typically 2. Default is 2.

        profile_output_bias : bool, optional
            Whether to include a bias term in the profile prediction head.
            Default is True.

        count_output_bias : bool, optional
            Whether to include a bias term in the count prediction head.
            Default is True.

        name : str or None, optional
            The name to use when saving models. If None, will use a default
            name. Default is None.

        trimming : int or None, optional
            The amount to trim from the ends of the input. If None, will be
            calculated based on n_layers. Default is None.

        verbose : bool, optional
            Whether to print verbose output during training. Default is True.

        coarse_bin_size : int, optional
            The size of bins in base pairs for coarse resolution binning.
            For example, with bin_size=20, a 1000bp profile is binned into
            50 bins of 20bp each. Default is 20.
        """
        super().__init__(
            n_filters=n_filters,
            n_layers=n_layers,
            n_outputs=n_outputs,
            n_control_tracks=n_control_tracks,
            count_loss_weight=0.0,  # No count loss for this variant
            profile_output_bias=profile_output_bias,
            count_output_bias=count_output_bias,
            name=name,
            trimming=trimming,
            verbose=verbose,
        )

        self.coarse_mnll = ChIPSeqCoarseMNLL(bin_size=coarse_bin_size)
        self.coarse_bin_size = int(coarse_bin_size)

        # Update logger for coarse MNLL specific metrics
        self.logger = Logger(
            [
                "Epoch",
                "Iteration",
                "Training Time",
                "Validation Time",
                "Training Coarse MNLL",
                "Validation Coarse MNLL",
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
        """
        Fit the model using CoarseMNLL loss (no count-head loss).

        This method trains the model using binned multinomial NLL at coarse
        resolution instead of base-resolution MNLL. The count-head is not
        trained (no count loss).

        Parameters
        ----------
        training_data : iterable
            An iterable that yields batches of training data. Each batch should
            be a tuple of (X, y, labels) or (X, X_ctl, y, labels) if control
            tracks are provided.

        optimizer : torch.optim.Optimizer
            The optimizer to use for training.

        scheduler : torch.optim.lr_scheduler or None, optional
            A learning rate scheduler. If provided, will be stepped after each
            epoch based on validation loss. Default is None.

        X_valid : torch.Tensor or None, optional
            Validation input sequences. Must be provided. Shape: (n, 4, L).

        X_ctl_valid : torch.Tensor or None, optional
            Validation control tracks. Shape: (n, n_control_tracks, L_out).
            Default is None.

        y_valid : torch.Tensor or None, optional
            Validation target profiles. Must be provided. Shape: (n, n_outputs, L_out).

        max_epochs : int, optional
            Maximum number of training epochs. Default is 100.

        batch_size : int, optional
            Batch size for validation. Default is 64.

        dtype : str or torch.dtype, optional
            Data type for computations ('float32', 'float16', etc.). Default is
            'float32'.

        device : str, optional
            Device to train on ('cuda' or 'cpu'). Default is 'cuda'.

        early_stopping : int or None, optional
            Number of epochs without improvement after which to stop training.
            If None, no early stopping. Default is None.
        """
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
                y = y.to(device)

                optimizer.zero_grad()
                self.train()

                with torch.autocast(device_type=device, dtype=dtype):
                    y_hat_logits, _ = self(X, X_ctl)
                    # Compute coarse MNLL loss (no count loss)
                    training_coarse_mnll = self.coarse_mnll(y_hat_logits, y)

                training_coarse_mnll.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                optimizer.step()

                iteration += 1

            train_time = time.time() - tic

            # Validation
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

                # Compute validation coarse MNLL
                valid_coarse_mnll = self.coarse_mnll(
                    y_hat_logits.to(device),
                    y_valid.to(device)
                )

                # Compute validation Pearson correlation
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

                valid_loss_value = valid_coarse_mnll.item()

                self.logger.add(
                    [
                        epoch,
                        iteration,
                        train_time,
                        valid_time,
                        training_coarse_mnll.item(),
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
                scheduler.step(valid_coarse_mnll)

            early_stop_count += 1
            if early_stopping is not None and early_stop_count >= early_stopping:
                break

        torch.save(self, f"{self.name}.final.torch")


__all__ = ["BPNetWithCoarseMNLL"]
