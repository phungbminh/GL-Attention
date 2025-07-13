import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


class BaseFusion(nn.Module, ABC):
    """Base class for all fusion strategies"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
    
    @abstractmethod
    def forward(self, A, L, x_in=None):
        """
        Args:
            A: Global attention features [B, C, H, W]
            L: Local attention features [B, C, H, W] 
            x_in: Original input features [B, C, H, W]
        Returns:
            fused: Fused features [B, C, H, W]
        """
        pass
    
    def get_name(self):
        return self.__class__.__name__


class OriginalFusion(BaseFusion):
    """Original multiplicative fusion: A * L"""
    def __init__(self, channels):
        super().__init__(channels)
        self.to_out_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
    
    def forward(self, A, L, x_in=None):
        fuse = A * L  # element-wise multiplication
        return self.to_out_conv(fuse)


class LearnableGatingFusion(BaseFusion):
    """Learnable gating mechanism inspired by Highway Networks"""
    def __init__(self, channels):
        super().__init__(channels)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid()
        )
        self.to_out_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
    
    def forward(self, A, L, x_in=None):
        concat_features = torch.cat([A, L], dim=1)
        gate = self.gate_conv(concat_features)
        fuse = gate * A + (1 - gate) * L
        return self.to_out_conv(fuse)


class WeightedAdditiveFusion(BaseFusion):
    """Weighted additive fusion with learnable parameters"""
    def __init__(self, channels):
        super().__init__(channels)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.to_out_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
    
    def forward(self, A, L, x_in=None):
        # Normalized weights
        weights_sum = torch.abs(self.alpha) + torch.abs(self.beta) + 1e-8
        alpha_norm = torch.abs(self.alpha) / weights_sum
        beta_norm = torch.abs(self.beta) / weights_sum
        
        # Tri-linear fusion: αA + βL + γ(A⊙L)
        fuse = alpha_norm * A + beta_norm * L + torch.sigmoid(self.gamma) * A * L
        return self.to_out_conv(fuse)


class ChannelAttentionFusion(BaseFusion):
    """Channel-wise attention fusion inspired by SE-Net"""
    def __init__(self, channels, reduction=16):
        super().__init__(channels)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels * 2, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels * 2, 1),
            nn.Sigmoid()
        )
        self.to_out_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
    
    def forward(self, A, L, x_in=None):
        # Channel attention weights
        concat_features = torch.cat([A, L], dim=1)
        attention = self.global_pool(concat_features)
        weights = self.fc(attention)
        w_A, w_L = weights.chunk(2, dim=1)
        
        # Weighted fusion
        fuse = w_A * A + w_L * L
        return self.to_out_conv(fuse)


class SpatialAttentionFusion(BaseFusion):
    """Spatial attention fusion with conv-based attention"""
    def __init__(self, channels, kernel_size=7):
        super().__init__(channels)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.to_out_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
    
    def forward(self, A, L, x_in=None):
        # Spatial attention map
        avg_A = torch.mean(A, dim=1, keepdim=True)
        avg_L = torch.mean(L, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_A, avg_L], dim=1)
        spatial_weight = torch.sigmoid(self.spatial_conv(spatial_input))
        
        # Spatially weighted fusion
        fuse = spatial_weight * A + (1 - spatial_weight) * L
        return self.to_out_conv(fuse)


class CrossAttentionFusion(BaseFusion):
    """Cross-attention between global and local features"""
    def __init__(self, channels, num_heads=8):
        super().__init__(channels)
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, A, L, x_in=None):
        B, C, H, W = A.shape
        
        # Use A as query, L as key,value
        q = self.q_proj(A).view(B, self.num_heads, self.head_dim, H*W).transpose(-1, -2)
        k = self.k_proj(L).view(B, self.num_heads, self.head_dim, H*W).transpose(-1, -2)
        v = self.v_proj(L).view(B, self.num_heads, self.head_dim, H*W).transpose(-1, -2)
        
        # Cross attention
        scale = self.head_dim ** -0.5
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, dim=-1)
        out = torch.matmul(attn, v)
        
        # Reshape and project
        out = out.transpose(-1, -2).contiguous().view(B, C, H, W)
        return self.out_proj(out)


class ResidualStrategy(nn.Module, ABC):
    """Base class for residual connection strategies"""
    @abstractmethod
    def forward(self, x_in, fused_features):
        pass


class OriginalResidual(ResidualStrategy):
    """Original: out = x_in + R"""
    def forward(self, x_in, fused_features):
        return x_in + fused_features


class LearnableGateResidual(ResidualStrategy):
    """Learnable gate: out = gate * x_in + (1-gate) * R"""
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.5)
    
    def forward(self, x_in, fused_features):
        gate = torch.sigmoid(self.gate)
        return gate * x_in + (1 - gate) * fused_features


class ProgressiveResidual(ResidualStrategy):
    """Progressive: out = x_in * (1-α) + R * α"""
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x_in, fused_features):
        alpha = torch.sigmoid(self.alpha)
        return x_in * (1 - alpha) + fused_features * alpha


class DropPathResidual(ResidualStrategy):
    """Stochastic depth inspired residual"""
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x_in, fused_features):
        if self.training and torch.rand(1) < self.drop_prob:
            return x_in  # Skip the fusion branch
        return x_in + fused_features


# Factory function to create fusion strategies
FUSION_STRATEGIES = {
    'original': OriginalFusion,
    'learnable_gate': LearnableGatingFusion,
    'weighted_add': WeightedAdditiveFusion,
    'channel_attn': ChannelAttentionFusion,
    'spatial_attn': SpatialAttentionFusion,
    'cross_attn': CrossAttentionFusion,
}

RESIDUAL_STRATEGIES = {
    'original': OriginalResidual,
    'learnable_gate': LearnableGateResidual,
    'progressive': ProgressiveResidual,
    'drop_path': DropPathResidual,
}


def create_fusion_strategy(strategy_name, channels, **kwargs):
    """Factory function to create fusion strategies"""
    if strategy_name not in FUSION_STRATEGIES:
        raise ValueError(f"Unknown fusion strategy: {strategy_name}")
    return FUSION_STRATEGIES[strategy_name](channels, **kwargs)


def create_residual_strategy(strategy_name, channels=None, **kwargs):
    """Factory function to create residual strategies"""
    if strategy_name not in RESIDUAL_STRATEGIES:
        raise ValueError(f"Unknown residual strategy: {strategy_name}")
    
    if strategy_name == 'original':
        return RESIDUAL_STRATEGIES[strategy_name]()
    else:
        return RESIDUAL_STRATEGIES[strategy_name](channels, **kwargs)