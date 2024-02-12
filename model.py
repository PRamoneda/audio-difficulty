
import torch
import torch.nn as nn
import torch.nn.functional as F


class ordinal_loss(nn.Module):
    """Ordinal regression with encoding as in https://arxiv.org/pdf/0704.1028.pdf"""

    def __init__(self, weight_class=False):
        super(ordinal_loss, self).__init__()
        self.weights = weight_class

    def forward(self, predictions, targets):
        # Fill in ordinalCoefficientVariationLoss target function, i.e. 0 -> [1,0,0,...]
        modified_target = torch.zeros_like(predictions)
        for i, target in enumerate(targets):
            modified_target[i, 0:target + 1] = 1

        # if torch tensor is empty, return 0
        if predictions.shape[0] == 0:
            return 0
        # loss
        if self.weights is not None:
            # pdb.set_trace()
            return torch.sum((self.weights * F.mse_loss(predictions, modified_target, reduction="none")).mean(axis=1))
        else:
            return torch.sum((F.mse_loss(predictions, modified_target, reduction="none")).mean(axis=1))

