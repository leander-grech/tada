"""Running value-target statistics for critic normalization."""

import torch
from torch import nn


class RunningMeanStd(nn.Module):
    """Tracks running mean and standard deviation via bias-corrected EMA.

    Uses Adam-style bias correction so the first few updates converge
    quickly to the true statistics instead of lagging near zero.
    """

    mean_acc: torch.Tensor
    sq_mean_acc: torch.Tensor
    count: torch.Tensor

    def __init__(self, beta: float = 0.99):
        super().__init__()
        self.beta = beta
        # Use float64 to avoid catastrophic cancellation in Var = E[x²] - E[x]²
        self.register_buffer("mean_acc", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("sq_mean_acc", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(0.0, dtype=torch.float64))

    @property
    def mean(self) -> torch.Tensor:
        if self.count == 0:
            return torch.zeros_like(self.mean_acc)
        return self.mean_acc / (1.0 - self.beta**self.count)

    @property
    def std(self) -> torch.Tensor:
        if self.count == 0:
            return torch.ones_like(self.sq_mean_acc)
        bias = 1.0 - self.beta**self.count
        corrected_mean = self.mean_acc / bias
        corrected_sq_mean = self.sq_mean_acc / bias
        # Var = E[x²] - E[x]²; clamp to avoid negative values from float precision
        variance = (corrected_sq_mean - corrected_mean**2).clamp(min=0.0)
        return torch.sqrt(variance).clamp(min=1e-6)

    def update(self, values: torch.Tensor) -> None:
        batch_var, batch_mean = torch.var_mean(values.detach(), correction=0)
        self.mean_acc = self.beta * self.mean_acc + (1 - self.beta) * batch_mean
        # Track second moment E[x²] = Var(x) + E[x]² for correct population variance
        self.sq_mean_acc = self.beta * self.sq_mean_acc + (1 - self.beta) * (
            batch_var + batch_mean**2
        )
        self.count += 1

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return ((values - self.mean) / self.std).to(values.dtype)

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        return (values * self.std + self.mean).to(values.dtype)
