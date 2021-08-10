import torch
import torch.nn.functional as F


def right_from_left(left: torch.Tensor, disparity: torch.Tensor):
    with torch.no_grad():
        h = left.shape[-2]
        w = left.shape[-1]
        l_grid = torch.meshgrid((torch.arange(0, h)/(h/2) - 1),
                                (torch.arange(0, w)/(w/2) - 1))
        l_grid = tuple(l.to(left.device) for l in l_grid)
        norm_pred_disp_l = disparity / (w / 2)
        l_diff_grid = l_grid[1][None, ...].to(disparity.device) + torch.abs(norm_pred_disp_l[0])
        l_grid = torch.stack([l_diff_grid, l_grid[0][None, ...].to(disparity.device)], dim=-1)
        l_grid = l_grid.to(left.dtype)
        right_pred = F.grid_sample(left, l_grid, align_corners=True)

        return right_pred
