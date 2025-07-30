import torch
import torch.nn as nn
from .CBAM import CBAMBlock
from .fusion_strategies import create_fusion_strategy, create_residual_strategy

class GLSABlock(nn.Module):
    """
    MHA → Stacked Attention block.
    - projects X→Q,K,V via 1×1 conv
    - performs multi-head self-attention
    - residual + LayerNorm
    - followed by one of {CBAM, BAM, scSE}
    """
    def __init__(self, channels, num_heads=8, attn_type='CBAM', reduction_ratio=16, kernel_size=7,
                 fusion_strategy='original', residual_strategy='original', 
                 fusion_kwargs=None, residual_kwargs=None):
        super().__init__()
        self.C = channels
        self.channels = channels
        self.fusion_strategy_name = fusion_strategy
        self.residual_strategy_name = residual_strategy
        
        # Multi-Head Self-Attention components
        self.to_qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.mha = nn.MultiheadAttention(embed_dim=channels,
                                         num_heads=num_heads,
                                         batch_first=True,
                                         bias=False)
        self.norm = nn.LayerNorm(channels)
        
        # Local attention mechanism (only CBAM supported)
        if attn_type == 'CBAM':
            self.attn2 = CBAMBlock(channel=channels, reduction=reduction_ratio, kernel_size=kernel_size)
        elif attn_type == 'none':
            self.attn2 = nn.Identity()
        else:
            raise ValueError(f"Unsupported attn_type: {attn_type}. Only 'CBAM' and 'none' are supported.")
        
        # Configurable fusion strategy
        fusion_kwargs = fusion_kwargs or {}
        if fusion_strategy == 'original':
            # Keep original implementation for backward compatibility
            self.fusion_module = None
            self.to_out_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        else:
            self.fusion_module = create_fusion_strategy(
                fusion_strategy, channels, **fusion_kwargs
            )
            self.to_out_conv = None
        
        # Configurable residual strategy
        residual_kwargs = residual_kwargs or {}
        if residual_strategy == 'original':
            # Keep original implementation
            self.residual_module = None
        else:
            self.residual_module = create_residual_strategy(
                residual_strategy, channels, **residual_kwargs
            )
        
        # Legacy components (for backward compatibility)
        self.bn_out = nn.BatchNorm2d(channels)
        self.gamma = nn.Parameter(torch.zeros(1))

        # Hooks for analysis
        self.hook_x = nn.Identity()
        self.hook_f1 = nn.Identity()
        self.hook_f2 = nn.Identity()
        self.hook_out = nn.Identity()
        self.hook_res = nn.Identity()
        self.hook_global = nn.Identity()
        self.hook_local = nn.Identity()
        self.hook_fused = nn.Identity()

    def forward(self, x):
        # Hook input
        x_in = self.hook_x(x)
        B, C, H, W = x_in.shape
        
        # 1) Multi-Head Self-Attention (Global)
        qkv = self.to_qkv(x_in)  # [B, 3C, H, W]
        q, k, v = qkv.chunk(3, dim=1)  # mỗi [B, C, H, W]
        
        # Flatten và transpose để MultiheadAttention
        q = q.flatten(2).permute(0, 2, 1)  # [B, N, C]
        k = k.flatten(2).permute(0, 2, 1)
        v = v.flatten(2).permute(0, 2, 1)
        
        attn_out, _ = self.mha(q, k, v)  # [B, N, C]
        # Residual + LayerNorm
        attn_out = self.norm(attn_out + q)  # [B, N, C]
        A = attn_out.permute(0, 2, 1).view(B, C, H, W)  # Global features
        A = self.hook_global(A)
        
        # 2) Local Attention
        f1 = x_in + A  # Intermediate fusion
        f1 = self.hook_f1(f1)
        L = self.attn2(f1)  # Local features
        L = self.hook_local(L)
        
        # 3) Fusion Strategy (configurable)
        if self.fusion_module is None:
            # Original fusion: A * L
            fused = A * L
            R = self.to_out_conv(fused)
        else:
            # Use configurable fusion strategy
            R = self.fusion_module(A, L, x_in)
        
        R = self.hook_fused(R)
        R = self.hook_res(R)
        
        # 4) Residual Strategy (configurable)
        if self.residual_module is None:
            # Original residual: x_in + R
            out = x_in + R
        else:
            # Use configurable residual strategy
            out = self.residual_module(x_in, R)
        
        out = self.hook_out(out)
        return out
    
    def get_config_info(self):
        """Return configuration information"""
        return {
            'fusion_strategy': self.fusion_strategy_name,
            'residual_strategy': self.residual_strategy_name,
            'channels': self.channels,
        }
    
    def count_parameters(self):
        """Count trainable parameters"""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        if self.fusion_module is not None:
            fusion = sum(p.numel() for p in self.fusion_module.parameters() if p.requires_grad)
        else:
            fusion = sum(p.numel() for p in self.to_out_conv.parameters() if p.requires_grad)
        
        if self.residual_module is not None:
            residual = sum(p.numel() for p in self.residual_module.parameters() if p.requires_grad)
        else:
            residual = 0
        
        return {
            'total': total,
            'fusion': fusion,
            'residual': residual,
            'base': total - fusion - residual
        }