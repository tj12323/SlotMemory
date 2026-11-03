from abc import ABC, abstractmethod
import torch


class DenoisingLoss(ABC):
    @abstractmethod
    def __call__(
        self, x: torch.Tensor, x_pred: torch.Tensor,
        noise: torch.Tensor, noise_pred: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        timestep: torch.Tensor,
        gradient_mask: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Base class for denoising loss.
        Input:
            - x: the clean data with shape [B, F, C, H, W]
            - x_pred: the predicted clean data with shape [B, F, C, H, W]
            - noise: the noise with shape [B, F, C, H, W]
            - noise_pred: the predicted noise with shape [B, F, C, H, W]
            - alphas_cumprod: the cumulative product of alphas (defining the noise schedule) with shape [T]
            - timestep: the current timestep with shape [B, F]
        """
        pass


class X0PredLoss(DenoisingLoss):
    def __call__(
        self, x: torch.Tensor, x_pred: torch.Tensor,
        noise: torch.Tensor, noise_pred: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        timestep: torch.Tensor,
        gradient_mask: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        err = (x - x_pred) ** 2
        if gradient_mask is not None:
            return err[gradient_mask].mean()
        return err.mean()


class VPredLoss(DenoisingLoss):
    def __call__(
        self, x: torch.Tensor, x_pred: torch.Tensor,
        noise: torch.Tensor, noise_pred: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        timestep: torch.Tensor,
        gradient_mask: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        weights = 1 / (1 - alphas_cumprod[timestep].reshape(*timestep.shape, 1, 1, 1))
        err = weights * (x - x_pred) ** 2
        if gradient_mask is not None:
            return err[gradient_mask].mean()
        return err.mean()


class NoisePredLoss(DenoisingLoss):
    def __call__(
        self, x: torch.Tensor, x_pred: torch.Tensor,
        noise: torch.Tensor, noise_pred: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        timestep: torch.Tensor,
        gradient_mask: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        err = (noise - noise_pred) ** 2
        if gradient_mask is not None:
            return err[gradient_mask].mean()
        return err.mean()


class FlowPredLoss(DenoisingLoss):
    def __call__(
        self, x: torch.Tensor, x_pred: torch.Tensor,
        noise: torch.Tensor, noise_pred: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        timestep: torch.Tensor,
        gradient_mask: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        err = (kwargs["flow_pred"] - (noise - x)) ** 2
        if gradient_mask is not None:
            return err[gradient_mask].mean()
        return err.mean()


NAME_TO_CLASS = {
    "x0": X0PredLoss,
    "v": VPredLoss,
    "noise": NoisePredLoss,
    "flow": FlowPredLoss
}


def get_denoising_loss(loss_type: str) -> DenoisingLoss:
    return NAME_TO_CLASS[loss_type]

import torch
import torch.nn as nn
import torch.nn.functional as F

class Slot_Slot_Contrastive_Loss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.1,
        batch_contrast: bool = True,
    ):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.batch_contrast = batch_contrast

    def forward(self, prev_slots: torch.Tensor, curr_slots: torch.Tensor):
        """
        prev_slots: [B, K, D] 上一个 Chunk 的 Slots
        curr_slots: [B, K, D] 当前 Chunk 的 Slots
        """
        # L2 归一化
        prev_slots = F.normalize(prev_slots, p=2.0, dim=-1)
        curr_slots = F.normalize(curr_slots, p=2.0, dim=-1)
        
        B, K, D = prev_slots.shape

        if self.batch_contrast:
            # [B, K, D] -> [1, B*K, D] 将不同 Batch 的 Slots 视为负样本
            prev_slots = prev_slots.reshape(1, B * K, D)
            curr_slots = curr_slots.reshape(1, B * K, D)
            S = B * K
            batch_size_for_loss = 1
        else:
            S = K
            batch_size_for_loss = B

        # 计算相似度矩阵 [Batch_for_loss, S, S]
        ss = torch.matmul(prev_slots, curr_slots.transpose(-2, -1)) / self.temperature
        
        # 你的原版使用的是 soft-target (torch.eye)，
        # 这里优化为更高效的 class indices 形式 (torch.arange) 供 CrossEntropy 使用
        target = torch.arange(S, device=ss.device).unsqueeze(0).expand(batch_size_for_loss, S)
        
        # 展平计算交叉熵: [Batch_for_loss * S, S], [Batch_for_loss * S]
        loss = self.criterion(ss.reshape(-1, S), target.reshape(-1))
        return loss