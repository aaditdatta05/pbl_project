# src/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits (N, C) for classification
        # targets: long (N,)
        ce = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class OrdinalMSE(nn.Module):
    """Simple regression MSE for ordinal labels (labels 0..4)."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, preds, targets):
        return self.mse(preds, targets)

class OrdinalRegressionLoss(nn.Module):
    """
    Ordinal Regression Loss using cumulative logistic regression.
    Treats class prediction as an ordered sequence.
    """
    def __init__(self, num_classes=5, pos_weight=None):
        super().__init__()
        self.num_classes = num_classes
        self.pos_weight = pos_weight
        
    def forward(self, logits, labels):
        """
        Args:
            logits: (batch_size, num_classes-1) - cumulative logits
            labels: (batch_size,) - ordinal labels (0-4)
            
        Returns:
            loss: scalar loss value
        """
        # Convert labels to cumulative binary targets
        # For label=2: targets = [1, 1, 0, 0] (y<=0, y<=1, y<=2, y<=3)
        batch_size = logits.size(0)
        device = logits.device
        
        cum_targets = torch.zeros(
            batch_size, 
            self.num_classes - 1, 
            device=device
        )
        
        for i in range(batch_size):
            label = labels[i].item()
            # Set targets for cumulative probabilities
            cum_targets[i, :label] = 1.0
        
        # Binary cross-entropy loss for each cumulative probability
        pos_weight = self.pos_weight
        if pos_weight is not None:
            pos_weight = pos_weight.to(device=device, dtype=logits.dtype)

        loss = F.binary_cross_entropy_with_logits(
            logits,
            cum_targets,
            pos_weight=pos_weight,
            reduction='mean'
        )
        
        return loss


class CumulativeLinkLoss(nn.Module):
    """
    Alternative: Cumulative Link Model Loss (more principled for ordinal regression)
    Assumes logits correspond to thresholds: P(y=k) = sigmoid(θ_k - η)
    """
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        
    def forward(self, logits, labels):
        """
        Args:
            logits: (batch_size, 1) - single linear predictor η
            labels: (batch_size,) - ordinal labels (0-4)
        """
        batch_size = logits.size(0)
        device = logits.device
        
        # Learnable thresholds (biases)
        thresholds = torch.arange(
            0, 
            self.num_classes - 1, 
            dtype=torch.float32, 
            device=device
        )
        
        # Compute cumulative probabilities
        # P(y <= k) = sigmoid(θ_k - η)
        logits_expanded = logits.expand(-1, self.num_classes - 1)
        cum_probs = torch.sigmoid(thresholds - logits_expanded)
        
        loss = 0.0
        for i in range(batch_size):
            label = labels[i].item()
            
            # P(y = label) = P(y <= label) - P(y <= label-1)
            if label == 0:
                prob = cum_probs[i, 0]
            elif label == self.num_classes - 1:
                prob = 1.0 - cum_probs[i, -1]
            else:
                prob = cum_probs[i, label] - cum_probs[i, label - 1]
            
            loss += -torch.log(prob + 1e-8)
        
        return loss / batch_size
