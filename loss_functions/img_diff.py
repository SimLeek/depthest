import numpy as np
import torch
import torch.nn.functional as F
from train_images.toy import right_img
from sparsepyramids.recursive_pyramidalize import RecursivePyramidalize2D, apply_2func_to_nested_tensors, \
    apply_func_to_nested_tensors, RecursiveSumDepyramidalize2D
from visualize.right_from_left import right_from_left


def basic_loss(predicted_right: torch.Tensor, actual_right: torch.Tensor):
    loss = F.smooth_l1_loss(predicted_right, actual_right)
    return loss


pyr2d = RecursivePyramidalize2D(min_size=None)
depyr2d = RecursiveSumDepyramidalize2D()


def remove_right_half(inp: torch.Tensor):
    w0 = inp.shape[-1] // 2
    out = inp[..., :w0]
    return out


def shift_lr(inp: torch.Tensor, shift_amt=1):
    out = torch.zeros_like(inp)
    if shift_amt >= 0:
        out[..., shift_amt:] = inp[..., :shift_amt]
    else:
        out[..., 0:shift_amt] = inp[..., abs(shift_amt):]

    return out


def pyramidal_loss(left_img: torch.Tensor,
                   right_img: torch.Tensor,
                   predicted_right: torch.Tensor,
                   disparity: torch.Tensor):
    with torch.no_grad():
        pre_1right = disparity+1.0
        pre_1right = right_from_left(left_img, pre_1right)
        pre_1left = disparity - 1.0
        pre_1left = right_from_left(left_img, pre_1left)

        #pre_half = remove_right_half(predicted_right)
        #act_half = remove_right_half(right_img)
        #pre_1rhalf = remove_right_half(pre_1right)
        #pre_1lhalf = remove_right_half(pre_1left)

        pyr_pre_right = pyr2d.forward(predicted_right)
        pyr_act_right = pyr2d.forward(right_img)
        pyr_pre1r_right = pyr2d.forward(pre_1right)
        pyr_pre1l_right = pyr2d.forward(pre_1left)

        w0 = predicted_right.shape[-1] // 2
        #disp_half = remove_right_half(disparity)
        pyr_disp = pyr2d.forward(disparity)
        diff_center = apply_2func_to_nested_tensors(pyr_pre_right, pyr_act_right, torch.sub)
        diff_l = apply_2func_to_nested_tensors(pyr_pre1l_right, pyr_act_right, torch.sub)
        diff_r = apply_2func_to_nested_tensors(pyr_pre1r_right, pyr_act_right, torch.sub)

        diff_center = apply_func_to_nested_tensors(diff_center, torch.abs)
        diff_l = apply_func_to_nested_tensors(diff_l, torch.abs)
        diff_r = apply_func_to_nested_tensors(diff_r, torch.abs)

        l = []
        for i, q in enumerate(reversed(list(zip(pyr_disp, diff_center, diff_l, diff_r)))):
            pyr, p, le, r = q
            cur_w = (w0**2 / (i + 1))  # force smaller images to contribute more to loss despite having less pixels
            select_left = torch.logical_and(le<p, le<r)
            pyr[select_left] -= 1
            select_right = torch.logical_and(r<p, r<le)
            pyr[select_right] += 1
            # remove rightmost pixel because it will likely contain errors,
            # which will travel to the left due to pyramidalization:
            pyr[..., -1:] = 0

            # p[r<p]
            l.insert(0, cur_w * pyr)

        l_img = depyr2d.forward(l)

        disp2 = torch.clone(disparity)
        disp2[...] = l_img

    loss = F.smooth_l1_loss(disparity, disp2)

    return loss
