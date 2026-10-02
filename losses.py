
import torch
import torch.nn as nn


# I wrote this code using ai
# The thingi understand about dice loss is 
# that instead of just raw logits prediction we care more about 
# How muuch the masks actaully cover the object using unions and intersections etc
# so seems like a good idea
# As per claude the dice loss defined in the metrics 
# Is not very well for lineary differentialbity something
# so we should use this one that is more differentiable
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        # Convert logits to probabilities; do NOT threshold.
        probs = torch.sigmoid(logits)

        # Compute Dice independently for each image.
        dims = (1, 2, 3)

        intersection = (probs * targets).sum(dim=dims)
        denominator = probs.sum(dim=dims) + targets.sum(dim=dims)

        dice = (
            2.0 * intersection + self.eps
        ) / (
            denominator + self.eps
        )

        # Maximize Dice by minimizing 1 - Dice.
        return (1.0 - dice).mean()