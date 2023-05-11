import numpy as np
import torch
import torch.nn.functional as F
from train_images.toy import right_img
from sparsepyramids.recursive_pyramidalize import RecursivePyramidalize2D, apply_2func_to_nested_tensors, \
    apply_func_to_nested_tensors, RecursiveSumDepyramidalize2D
from visualize.right_from_left import right_from_left
from loss_functions.img_diff import basic_loss, remove_right_half, shift_lr, pyr2d, depyr2d


def pyramidal_loss_rand(left_img: torch.Tensor,
                        right_img: torch.Tensor,
                        predicted_right: torch.Tensor,
                        disparity: torch.Tensor):
    with torch.no_grad():
        pre_1right = disparity + torch.rand(disparity.shape)*predicted_right.shape[-1]/2
        pre_1right = right_from_left(left_img, pre_1right)
        pre_1left = disparity - torch.rand(disparity.shape)*predicted_right.shape[-1]/2
        pre_1left = right_from_left(left_img, pre_1left)

        pyr_pre_right = pyr2d.forward(predicted_right)
        pyr_act_right = pyr2d.forward(right_img)
        pyr_pre1r_right = pyr2d.forward(pre_1right)
        pyr_pre1l_right = pyr2d.forward(pre_1left)

        w0 = predicted_right.shape[-1] // 2
        pyr_disp = pyr2d.forward(disparity)
        diff_center = apply_2func_to_nested_tensors(pyr_pre_right, pyr_act_right, torch.sub)
        diff_l = apply_2func_to_nested_tensors(pyr_pre1l_right, pyr_act_right, torch.sub)
        diff_r = apply_2func_to_nested_tensors(pyr_pre1r_right, pyr_act_right, torch.sub)

        diff_center = apply_func_to_nested_tensors(diff_center, torch.abs)
        diff_l = apply_func_to_nested_tensors(diff_l, torch.abs)
        diff_r = apply_func_to_nested_tensors(diff_r, torch.abs)

        l = []
        for i, q in enumerate(
                reversed(list(zip(pyr_disp, diff_center, diff_l, diff_r, pyr_pre1r_right, pyr_pre1l_right)))):
            pyr, p, le, r, p1r, p1l = q
            cur_w = (w0 ** 2 / (i + 1))  # force smaller images to contribute more to loss despite having less pixels
            select_left = torch.logical_and(le < p, le < r)
            pyr[select_left] -= p1l[select_left]
            select_right = torch.logical_and(r < p, r < le)
            pyr[select_right] += p1r[select_right]
            # remove rightmost pixel because it will likely contain errors,
            # which will travel to the left due to pyramidalization:
            pyr[..., -1:] = 0

            l.insert(0, cur_w * pyr)

        l_img = depyr2d.forward(l)

        disp2 = torch.clone(disparity)
        disp2[...] = l_img

    loss = F.smooth_l1_loss(disparity, disp2)

    return loss
