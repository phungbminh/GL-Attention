import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101, resnet152,
    vgg11, vgg13, vgg16, vgg19
)
from torchvision.models.resnet import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
from torchvision.models.vgg import VGG11_Weights, VGG13_Weights, VGG16_Weights, VGG19_Weights

from attention.GLSABlock import GLSABlock


class BackboneWithGLSA(nn.Module):
    """
    Generic backbone model with GLSA block integration.
    Supports multiple backbone architectures with configurable attention placement.
    """
    
    def __init__(self, 
                 backbone_name='resnet18',
                 pretrained=True, 
                 num_classes=7,
                 attn_type='CBAM',
                 fusion_strategy='original',
                 residual_strategy='original',
                 num_heads=8,
                 reduction_ratio=16):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.attn_type = attn_type
        
        # Get backbone model and integration point
        self.backbone, self.integration_point, self.feature_channels = self._get_backbone(backbone_name, pretrained)
        
        # Add GLSA block if attention is enabled
        if attn_type != 'none':
            self.glsa_block = GLSABlock(
                channels=self.feature_channels,
                num_heads=num_heads,
                attn_type=attn_type,
                fusion_strategy=fusion_strategy,
                residual_strategy=residual_strategy,
                reduction_ratio=reduction_ratio
            )
        else:
            self.glsa_block = None
        
        # Final classifier
        self.classifier = self._create_classifier(num_classes)
        
    def _get_backbone(self, backbone_name, pretrained):
        """Get backbone model and determine integration point"""
        
        # ResNet family
        if backbone_name == 'resnet18':
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_channels = 256  # layer3 output channels
            integration_point = 'layer3'
            
        elif backbone_name == 'resnet34':
            model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_channels = 256
            integration_point = 'layer3'
            
        elif backbone_name == 'resnet50':
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            feature_channels = 1024  # layer3 output channels for ResNet50
            integration_point = 'layer3'
            
        elif backbone_name == 'resnet101':
            model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2 if pretrained else None)
            feature_channels = 1024
            integration_point = 'layer3'
            
        elif backbone_name == 'resnet152':
            model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2 if pretrained else None)
            feature_channels = 1024
            integration_point = 'layer3'
            
        # VGG family
        elif backbone_name == 'vgg11':
            model = vgg11(weights=VGG11_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_channels = 512
            integration_point = 'features'
            
        elif backbone_name == 'vgg13':
            model = vgg13(weights=VGG13_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_channels = 512
            integration_point = 'features'
            
        elif backbone_name == 'vgg16':
            model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_channels = 512
            integration_point = 'features'
            
        elif backbone_name == 'vgg19':
            model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_channels = 512
            integration_point = 'features'
            
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        return model, integration_point, feature_channels
    
    def _create_classifier(self, num_classes):
        """Create final classifier based on backbone type"""
        if 'resnet' in self.backbone_name:
            return nn.Linear(self.backbone.fc.in_features, num_classes)
        elif 'vgg' in self.backbone_name:
            return nn.Sequential(
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes)
            )
        else:
            # Default: Global average pooling + linear
            return nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(self.feature_channels, num_classes)
            )
    
    def forward(self, x):
        """Forward pass with GLSA integration"""
        
        if 'resnet' in self.backbone_name:
            # ResNet forward with GLSA after layer3
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            
            # Apply GLSA after layer3
            if self.glsa_block is not None:
                x = self.glsa_block(x)
            
            x = self.backbone.layer4(x)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            
        elif 'vgg' in self.backbone_name:
            # VGG forward with GLSA before final layers
            features = list(self.backbone.features.children())
            
            # Apply features up to a certain point
            split_point = len(features) - 6  # Before last few layers
            
            for i, layer in enumerate(features):
                x = layer(x)
                if i == split_point and self.glsa_block is not None:
                    x = self.glsa_block(x)
            
            x = self.classifier(x)
            
        
        return x
    
    def get_feature_channels(self):
        """Get the number of feature channels at GLSA integration point"""
        return self.feature_channels


# Convenience functions for creating specific models
def create_model(backbone_name, pretrained=True, num_classes=7, **kwargs):
    """Factory function to create models with GLSA"""
    return BackboneWithGLSA(
        backbone_name=backbone_name,
        pretrained=pretrained,
        num_classes=num_classes,
        **kwargs
    )


# Supported backbone names
SUPPORTED_BACKBONES = [
    # ResNet family
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    # VGG family  
    'vgg11', 'vgg13', 'vgg16', 'vgg19'
]


def get_model_info(backbone_name):
    """Get information about a specific backbone model"""
    model_info = {
        # ResNet family
        'resnet18': {'params': '11.7M', 'top1_acc': '69.8%', 'family': 'ResNet'},
        'resnet34': {'params': '21.8M', 'top1_acc': '73.3%', 'family': 'ResNet'},
        'resnet50': {'params': '25.6M', 'top1_acc': '80.9%', 'family': 'ResNet'},
        'resnet101': {'params': '44.5M', 'top1_acc': '81.9%', 'family': 'ResNet'},
        'resnet152': {'params': '60.2M', 'top1_acc': '82.3%', 'family': 'ResNet'},
        
        # VGG family
        'vgg11': {'params': '132.9M', 'top1_acc': '69.0%', 'family': 'VGG'},
        'vgg13': {'params': '133.0M', 'top1_acc': '69.9%', 'family': 'VGG'},
        'vgg16': {'params': '138.4M', 'top1_acc': '71.6%', 'family': 'VGG'},
        'vgg19': {'params': '143.7M', 'top1_acc': '72.4%', 'family': 'VGG'},
    }
    
    return model_info.get(backbone_name, {'params': 'Unknown', 'top1_acc': 'Unknown', 'family': 'Unknown'})


if __name__ == '__main__':
    # Test different backbones
    print("Testing backbone models:")
    
    test_backbones = ['resnet18', 'resnet50', 'vgg16']
    
    for backbone in test_backbones:
        try:
            model = create_model(backbone, pretrained=False, num_classes=7)
            x = torch.randn(1, 3, 224, 224)
            output = model(x)
            
            total_params = sum(p.numel() for p in model.parameters())
            info = get_model_info(backbone)
            
            print(f"{backbone}:")
            print(f"  Output shape: {output.shape}")
            print(f"  Total params: {total_params:,}")
            print(f"  Feature channels: {model.get_feature_channels()}")
            print(f"  Family: {info['family']}")
            print()
            
        except Exception as e:
            print(f"Error with {backbone}: {e}")