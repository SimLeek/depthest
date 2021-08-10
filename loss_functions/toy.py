import numpy as np
import torch
import torch.nn.functional as F
from train_images.toy import right_img

_goal: torch.Tensor = torch.ones(right_img.shape) * 20


def toy_loss(inp: torch.Tensor):
    global _goal
    loss = F.smooth_l1_loss(inp, _goal)
    return loss
