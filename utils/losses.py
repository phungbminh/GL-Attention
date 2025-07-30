import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance.
    
    Paper: "Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002)
    
    Args:
        alpha: Weighting factor for rare class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: Specifies the reduction to apply to the output
        ignore_index: Specifies a target value that is ignored
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: A float tensor of shape [N, C] containing predictions
            targets: A long tensor of shape [N] containing ground truth labels
        
        Returns:
            Loss tensor
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        
        # Compute probabilities
        pt = torch.exp(-ce_loss)
        
        # Compute alpha term
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
        else:
            alpha_t = 1.0
            
        # Compute focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss Based on Effective Number of Samples.
    
    Paper: "Class-Balanced Loss Based on Effective Number of Samples" 
           (https://arxiv.org/abs/1901.05555)
    
    Args:
        samples_per_cls: Number of samples for each class
        beta: Hyperparameter for re-weighting (default: 0.9999)
        gamma: Focusing parameter for focal loss (default: 2.0)
        loss_type: Type of loss ('focal', 'sigmoid', 'softmax')
    """
    
    def __init__(self, samples_per_cls, beta=0.9999, gamma=2.0, loss_type='focal'):
        super(ClassBalancedLoss, self).__init__()
        self.samples_per_cls = samples_per_cls
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        
        # Calculate effective numbers
        effective_num = 1.0 - torch.pow(beta, torch.FloatTensor(samples_per_cls))
        weights = (1.0 - beta) / effective_num
        self.weights = weights / weights.sum() * len(weights)
        
    def forward(self, inputs, targets):
        weights = self.weights.to(inputs.device)
        
        if self.loss_type == 'focal':
            cb_loss = F.cross_entropy(inputs, targets, weight=weights, reduction='none')
            pt = torch.exp(-cb_loss)
            focal_loss = (1 - pt) ** self.gamma * cb_loss
            return focal_loss.mean()
        elif self.loss_type == 'sigmoid':
            targets_one_hot = F.one_hot(targets, num_classes=len(self.samples_per_cls)).float()
            sigmoid_loss = F.binary_cross_entropy_with_logits(inputs, targets_one_hot, weight=weights)
            return sigmoid_loss
        elif self.loss_type == 'softmax':
            return F.cross_entropy(inputs, targets, weight=weights)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss with class weighting support.
    
    Args:
        smoothing: Label smoothing factor (default: 0.1)
        weight: Manual rescaling weight given to each class
        reduction: Specifies the reduction to apply to the output
    """
    
    def __init__(self, smoothing=0.1, weight=None, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        log_prob = F.log_softmax(inputs, dim=-1)
        
        if self.weight is not None:
            weight = self.weight.to(inputs.device)
        else:
            weight = None
            
        # Standard cross entropy
        nll_loss = F.nll_loss(log_prob, targets, weight=weight, reduction='none')
        
        # Smooth loss
        smooth_loss = -log_prob.mean(dim=-1)
        if weight is not None:
            smooth_loss = smooth_loss * weight[targets]
            
        # Combine losses
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def get_loss_function(loss_name, num_classes=None, samples_per_cls=None, **kwargs):
    """
    Factory function to get loss function by name.
    
    Args:
        loss_name: Name of loss function ('focal', 'class_balanced', 'label_smoothing', 'cross_entropy')
        num_classes: Number of classes
        samples_per_cls: List of sample counts per class (for class_balanced loss)
        **kwargs: Additional arguments for loss function
    
    Returns:
        Loss function instance
    """
    if loss_name == 'focal':
        return FocalLoss(**kwargs)
    elif loss_name == 'class_balanced':
        if samples_per_cls is None:
            raise ValueError("samples_per_cls required for class_balanced loss")
        return ClassBalancedLoss(samples_per_cls, **kwargs)
    elif loss_name == 'label_smoothing':
        return LabelSmoothingCrossEntropy(**kwargs)
    elif loss_name == 'cross_entropy':
        weight = kwargs.get('weight', None)
        return nn.CrossEntropyLoss(weight=weight)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")