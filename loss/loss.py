import torch
import torch.nn as nn
import torch.nn.functional as F

class DS_UNETR_PlusPlus_Loss(nn.Module):
    def __init__(self, num_classes=4, smooth=1e-5):
        super(DS_UNETR_PlusPlus_Loss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, y_pred, y_true, weight=None):
        """
        Loss function according to Nature Scientific Reports (2025) paper:
        L = CrossEntropy + (1 - SoftDice)
        """
        # 1. Weighted Cross Entropy
        # Helps the model orient correct classification of voxels in space
        ce_loss = F.cross_entropy(y_pred, y_true.long(), weight=weight)

        # 2. Soft Dice Loss
        y_pred_soft = torch.softmax(y_pred, dim=1)
        y_true_oh = F.one_hot(y_true.long(), self.num_classes).permute(0, 4, 1, 2, 3).float()

        # Flatten spatial dimensions (B, C, -1)
        y_pred_flat = y_pred_soft.view(y_pred_soft.shape[0], self.num_classes, -1)
        y_true_flat = y_true_oh.view(y_true_oh.shape[0], self.num_classes, -1)

        intersection = (y_pred_flat * y_true_flat).sum(-1)
        # Soft Dice formula: use squaring in denominator to increase gradient stability
        cardinality = (y_pred_flat**2).sum(-1) + (y_true_flat**2).sum(-1)
        
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1 - dice_score.mean()

        # 1:1 ratio as suggested by the research for optimal DSC results
        return ce_loss + dice_loss