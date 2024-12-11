import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', l1_lambda=1e-5, l2_lambda=1e-4):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        
    def forward(self, inputs, targets, model_params):
        """
        Args:
            inputs: 모델 예측값 (N, C)
            targets: 실제 레이블 (N,)
            model_params: 모델의 파라미터들 (정규화에 사용)
        """
        # 기본 Focal Loss 계산
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        # L1 정규화
        l1_reg = torch.tensor(0., requires_grad=True)
        for param in model_params:
            l1_reg = l1_reg + torch.norm(param, 1)
        
        # L2 정규화
        l2_reg = torch.tensor(0., requires_grad=True)
        for param in model_params:
            l2_reg = l2_reg + torch.norm(param, 2)
            
        # 전체 Loss 계산
        if self.reduction == 'mean':
            total_loss = focal_loss.mean() + self.l1_lambda * l1_reg + self.l2_lambda * l2_reg
        elif self.reduction == 'sum':
            total_loss = focal_loss.sum() + self.l1_lambda * l1_reg + self.l2_lambda * l2_reg
        else:  # 'none'
            total_loss = focal_loss + self.l1_lambda * l1_reg + self.l2_lambda * l2_reg
            
        return total_loss