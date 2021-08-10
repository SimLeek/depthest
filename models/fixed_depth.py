import torch


class FixedDepthEst(torch.nn.Module):
    def __init__(self, height_width):
        super().__init__()
        h = height_width[0]
        w = height_width[1]
        self.disparity = torch.nn.Parameter(torch.ones((1, 1, h, w)))

    def forward(self):
        return self.disparity