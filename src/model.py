import torch
import torch.nn as nn
import timm

class DRModel(nn.Module):
    """
    DenseNet backbone with ordinal regression.
    Predicts cumulative probabilities: P(y <= k) for k in [0, 1, 2, 3]
    Actual grade = sum(P(y <= k) > 0.5)
    """
    def __init__(self, backbone_name='densenet121', pretrained=True, dropout=0.3, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=pretrained, 
            features_only=False, 
            num_classes=0
        )
        
        in_features = self.backbone.num_features
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Ordinal regression head
        # Output: (num_classes - 1) cumulative probabilities
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes - 1)  # 4 outputs for 5 classes
        )
        
    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, 3, H, W)
        
        Returns:
            logits: tensor of shape (batch_size, 4) - cumulative logits
        """
        x = self.backbone.forward_features(x)
        if x.ndim == 4:
            x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x
    
    def predict_ordinal(self, x):
        """
        Convert cumulative probabilities to class predictions.
        
        Args:
            x: input tensor
            
        Returns:
            grades: predicted grades (0-4)
            probs: cumulative probabilities
        """
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        
        # Grade = number of cumulative probabilities > 0.5
        grades = (probs > 0.5).sum(dim=1).long()
        
        return grades, probs


class DRModelV2(nn.Module):
    """
    EfficientNetV2-S backbone with ordinal regression.
    Predicts cumulative probabilities: P(y <= k) for k in [0, 1, 2, 3]
    Actual grade = sum(P(y <= k) > 0.5)
    """
    def __init__(self, backbone_name='tf_efficientnetv2_s', pretrained=True, dropout=0.3, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=False,
            num_classes=0
        )

        in_features = self.backbone.num_features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes - 1)
        )

    def forward(self, x):
        x = self.backbone.forward_features(x)
        if x.ndim == 4:
            x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x

    def predict_ordinal(self, x):
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        grades = (probs > 0.5).sum(dim=1).long()
        return grades, probs