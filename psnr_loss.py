import torch
import torch.nn as nn

class PSNRLoss(nn.Module):
    """
    Differentiable –PSNR loss for inputs that were Z-score-normalized
    (x ← (x – mean) / std).  `data_range` is computed per-sample from
    the *normalized* target tensor, so no hard-coded 1.0 is assumed.

    Args:
        reduction: 'mean' | 'sum' | 'none'
        eps: numerical stability term
    """
    def __init__(self, reduction: str = 'mean', eps: float = 1e-8, offset: float = 50.0):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
        self._log10 = 1.0 / torch.log(torch.tensor(10.0))
        self.offset = offset

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(1, pred.ndim))               # all but batch dim

        mse = torch.mean((pred - target) ** 2, dim=dims) + self.eps

        # dynamic range of Z-score-normalized target, per sample
        flat = target.view(target.size(0), -1)
        data_range = (flat.max(dim=1).values - flat.min(dim=1).values + self.eps)

        psnr = 10.0 * self._log10 * torch.log((data_range ** 2) / mse)
        loss = self.offset-psnr                                        # minimise –PSNR

        if   self.reduction == 'mean': return loss.mean()
        elif self.reduction == 'sum':  return loss.sum()
        else:                          return loss         # 'none'
