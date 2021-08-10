import random

import numpy as np
from displayarray import display
import time
import torch
from torch import nn, optim
import os
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torchvision.transforms.functional import resize
import torch.nn.functional as F
from recursive_pyramidalize import RecursivePyramidalize2D, apply_2func_to_nested_tensors, nested_convs, cat_nested_tensors, RecursiveSumDepyramidalize2D, RecursiveDepthDepyramidalize2D, apply_func_to_nested_tensors, log2_channel_distance_to_float_distance
from torchvision.transforms import functional as FV
import math
from torch.nn.utils import weight_norm

def disp2depth(disp, calib):
    depth = calib[:, None, None] / disp.clamp(min=1e-8)
    return depth


def inference(imgL, imgR, calib, model):
    #model.eval()
    imgL, imgR, calib = imgL.cuda(), imgR.cuda(), calib.float().cuda()

    #with torch.no_grad():
    output, prob = model(imgL, imgR)
    # output = disp2depth(output, calib)
    return output, prob

def W1_loss(prob, target, off, mask=None, maxdisp=192, down=1, reduction="mean"):
    # B,D,H,W to B,H,W,D
    off = off.permute(0, 2, 3, 1)
    prob = prob.permute(0, 2, 3, 1)
    grid = torch.arange(0, maxdisp // down, device="cuda", requires_grad=False).float()[
        None, None, None, :
    ]
    depth = (grid + off) * down
    if mask is None:
        mask = (target > 0) * (target < maxdisp)
        mask.detach_()
    target = target.unsqueeze(3)
    out = torch.abs(depth[mask] - target[mask])
    loss = torch.sum(prob[mask] * out, 1)
    # note: comparing loss to test out, not train. Each prob needs to be modded to target
    # self training: off+prob, off const, prob goes to infinity. Fix based off pred of all
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

from edge_pyramid import image_to_edge_pyramid, edge_detector, edge

edge = torch.FloatTensor(edge_detector(2))/2.0
edge = torch.swapaxes(edge, 0, 3)
edge = torch.swapaxes(edge, 1, 2)

class PyrConv3ChanSum(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.nc2 = nested_convs(2, 24, 3, padding=1, bias=True)  # use kernel and padding size to not reduce image size at all
        self.nc3 = nested_convs(24, 1, 3, padding=1, bias=True)
        self.nc4 = nested_convs(12, 1, 3, padding=1, bias=True)

        self.bias_multiplier = torch.nn.Parameter(torch.FloatTensor([1.0]))
        self.bias_addition = torch.nn.Parameter(torch.FloatTensor([0.0]))


        #self.nc4 = nested_convs(12, 6, 3, padding=1, bias=True)
        #self.nc5 = nested_convs(6, 1, 3, padding=1, bias=True)


        self.pyr = RecursivePyramidalize2D()
        self.de = RecursiveSumDepyramidalize2D(scale_pow=1)
        self.de1 = RecursiveDepthDepyramidalize2D(12, 1, 1)
        #self.de1 = RecursiveConvDepyramidalize2D(288, 32)
        #self.de2 = RecursiveConvDepyramidalize2D(384, 1)


        #self.ygrid, self.xgrid = torch.meshgrid((torch.arange(0, 255) - 255 / 2) / (255 / 2),
        #                       (torch.arange(0, 255) - 255 / 2) / (255 / 2))

        #self.xgrid = self.xgrid.cuda()

        #self.xpyr = self.pyr(self.xgrid[None, None, ...])

        self.lateral = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.zero_()
                #m.weight.data.normal_(0, math.sqrt(2. / n))

        self.cuda()

    def blink(self, l):
        self.lateral = self.pyr.forward(
            self.pyr.forward(torch.zeros(tuple([l.shape[0]] + [6] + list(l.shape[2:]))).cuda()))

    def forward(self, l, r):
        if l.shape[1] == 3 and r.shape[1]==3:
            # assuming bgr from opencv
            lb = l[:,0:1, ...]
            lg = l[:,1:2, ...]
            lr = l[:,2:3, ...]
            rb = r[:, 0:1, ...]
            rg = r[:, 1:2, ...]
            rr = r[:, 2:3, ...]

            l = 0.299 * lr + 0.587 * lg + 0.114 * lb
            r = 0.299 * rr + 0.587 * rg + 0.114 * rb

        #lpyr = self.pyr(l)
        #rpyr = self.pyr(r)

        lepyr = image_to_edge_pyramid(l)
        lepyr = apply_func_to_nested_tensors(lepyr, torch.abs)
        lepyr = apply_2func_to_nested_tensors(lepyr, 1e-12, torch.add)
        norm = apply_func_to_nested_tensors(lepyr, torch.norm, dim=(2,3) )
        lepyr = apply_2func_to_nested_tensors(lepyr, norm, torch.div)

        repyr = image_to_edge_pyramid(r)
        repyr = apply_func_to_nested_tensors(repyr, torch.abs)
        repyr = apply_2func_to_nested_tensors(repyr, 1e-12, torch.add)
        norm = apply_func_to_nested_tensors(repyr, torch.norm, dim=(2,3))
        repyr = apply_2func_to_nested_tensors(repyr, norm, torch.div)

        edge = cat_nested_tensors([lepyr, repyr], 1)

        lc2 = self.nc2.forward(edge)

        lc2p = self.pyr.forward(lc2)

        lc3 = self.nc3.forward(lc2p)

        dist1 = self.de.forward(lc3)

        dist2 = self.de1.forward(dist1)

        #dist3 = self.nc4.forward(dist2)


        #dist = self.de.forward(dist2)

        #dist2 = dist2/(torch.max(dist2, dim=1).values+1e-7)

        dist3, prob3 = log2_channel_distance_to_float_distance(dist2)

        if self.bias_multiplier.device != dist3.device:
            self.bias_multiplier = self.bias_multiplier.to(dist3.device)
            self.bias_addition = self.bias_addition.to(dist3.device)

        dist3 = dist3/self.bias_multiplier + self.bias_addition

        return dist3, prob3
import time

def run():
    fout = open('pow_conv_end_with_bias_mul_add.csv', mode='w')
    i=0
    t0 = time.time()
    fout.write(f'i, loss\n')

    maxdisp=192
    down = 1
    model = PyrConv3ChanSum()  # down=1
    print(model)
    #model = nn.DataParallel(model).cuda()
    #torch.backends.cudnn.benchmark = True

    optimizer = optim.AdamW(model.parameters(), lr=0.01, betas=(0.9, 0.999))
    scheduler = MultiStepLR(optimizer, milestones=[100, 200], gamma=0.001)

    '''check_path = '..' + os.sep + 'kitti15_w1_disp_train' + os.sep + 'model_best.pth.tar'
    print(f"=> loading checkpoint '{check_path}'")
    checkpoint = torch.load(check_path)
    model.load_state_dict(checkpoint["state_dict"])
    start_epoch = checkpoint["epoch"]
    optimizer.load_state_dict(checkpoint["optimizer"])
    best_RMSE = checkpoint["best_EPE"]
    scheduler.load_state_dict(checkpoint["scheduler"])'''

    calib = torch.FloatTensor([.5])

    d = display("tcp://127.0.0.1:7880#stereo_in")

    model.train()
    model.training = True
    optimizer.zero_grad()
    loss = None

    for up in d:
        if up:
            iup = next(iter(up.values()))[0]
            left = torch.Tensor(np.copy(iup[0][0]))
            right = torch.Tensor(np.copy(iup[1][0]))
            left = left[np.newaxis, ...]
            left = torch.swapaxes(left, 1, 3)
            left = torch.swapaxes(left, 2, 3)
            right = right[np.newaxis, ...]
            right = torch.swapaxes(right, 1, 3)
            right = torch.swapaxes(right, 2, 3)

            left = resize(left, [255,255])
            right = resize(right, [255,255])

            H = left.shape[-2]
            W = left.shape[-1]

            # todo: multiscale loss instead of final image pix difference
            #with torch.autograd.detect_anomaly():
            # train by guessing other image using offsets
            if loss is None:
                loss = torch.zeros_like(left)
            #if random.randint(0,20)==10:
            #    model.blink(left)
            pred, prob = inference(left, right, calib, model)
            #pred1 = torch.nan_to_num(pred1)
            #pred2 = torch.nan_to_num(pred2)
            #pred3 = torch.nan_to_num(pred3)
            #off1 = torch.nan_to_num(off1)
            #off2 = torch.nan_to_num(off2)
            #off3 = torch.nan_to_num(off3)

            # pred_disp_r = inference(left, right, calib, model)

            pred_disp_l = pred

            with torch.no_grad():
                # get predicted version of right image
                lgrid = torch.meshgrid((torch.arange(0, H) - H / 2) / (H / 2),
                                       (torch.arange(0, W) - W / 2) / (W / 2))
                norm_pred_disp_l = pred_disp_l / (W / 2)
                ldiffgrid = lgrid[1][None, ...].to(pred_disp_l.device) - torch.abs(norm_pred_disp_l[0])
                lgrid = torch.stack([ldiffgrid, lgrid[0][None, ...].to(pred_disp_l.device)], dim=-1)
                lgrid = lgrid.to(left.dtype)
                right_pred = F.grid_sample(left.to(lgrid.device), lgrid)
                #right_pred = torch.swapaxes(right_pred, 2, 3)

                # get disparity approximation from predicted right image vs actual
                # requirement: r=rpred -> approx_pred=nn_pred
                # requirement: r-rpred^ -> approx_pred-nn_pred
                right = right.to(right_pred.device)
                diff_center = torch.abs(right - right_pred)

                #right_pred_lshift = torch.roll(right_pred, 1, 3)
                #right_pred_lshift[..., -1] = 0
                #diff_left = torch.abs(right - right_pred_lshift)

                #right_pred_rshift = torch.roll(right_pred, -1, 3)
                #right_pred_rshift[..., 1] = 0
                #diff_right = torch.abs(right - right_pred_rshift)

                # todo: if I restrict the output to be between 0 and 1, then guess distance based on that value and location in the pyramid, it might be a lot more accurate

                diff_approx = torch.clone(pred_disp_l)
                sumdiffc = torch.sum(diff_center, dim=1)  # sum channel diffs
                #sumdiffl = torch.sum(diff_left, dim=1)
                #sumdiffr = torch.sum(diff_right, dim=1)
                #diffsl = (sumdiffc - sumdiffl)[None, ...]
                #diffsr = (sumdiffc - sumdiffr)[None, ...]
                diff_oob = (torch.abs(ldiffgrid) - 1.0)[None, ...]
                diff_approx += (sumdiffc / (torch.nan_to_num(
                    torch.linalg.norm(sumdiffc)) + 1e-12))  # l1 normalize over both x and y
                diff_approx[diff_oob > 0] = diff_approx[diff_oob > 0] -255*3
                #diff_approx += torch.log1p(torch.abs(pred_disp_l))*-torch.sign(pred_disp_l)
                #diff_approx[diffsl > 0] += (diffsl / (torch.nan_to_num(torch.linalg.norm(diffsl)) + 1e-12))[
                #    diffsl > 0]  # l1 normalize over both x and y
                #diff_approx[diffsr > 0] -= (diffsr / (torch.nan_to_num(torch.linalg.norm(diffsr)) + 1e-12))[
                #    diffsr > 0]
                #diff_approx[pred_disp_l<0] += pred_disp_l[pred_disp_l<0]*2

                diff_approx = diff_approx.to(pred_disp_l.device)

            #loss = W1_loss(pred3, diff_approx, off1, reduction="mean")
            #loss = (
            #        0.5 * W1_loss(pred1, diff_approx, off1, reduction="mean")
            #        + 0.7 * W1_loss(pred2, diff_approx, off2, reduction="mean")
            #        + W1_loss(pred3, diff_approx, off3, reduction="mean")
            #)
            p2d = RecursivePyramidalize2D()
            pyr_pred = p2d.forward(pred_disp_l)
            pyr_approx = p2d.forward(diff_approx)
            #probpyr = p2d.forward(prob)
            #pyr_pred = apply_2func_to_nested_tensors(pyr_pred, probpyr, torch.mul)
            #pyr_approx = apply_2func_to_nested_tensors(pyr_approx, probpyr, torch.mul)

            loss_list = apply_2func_to_nested_tensors(pyr_pred, pyr_approx, F.smooth_l1_loss)
            #loss2 = F.smooth_l1_loss(prob, torch.ones_like(prob))*2
            loss = sum(loss_list)

            fout.write(f'{i}, {loss.cpu().item()}\n')
            i+=1

            print(time.time() - t0)
            #if time.time()-t0 >= 60:
            #    exit()

            loss.backward()

            #torch.nn.utils.clip_grad_value_(model.parameters(), 100)
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1000.0, error_if_nonfinite=True)
            if not torch.isnan(loss).item():
                optimizer.step()
            else:
                print('training failed')

            loss_disp = (diff_approx - pred_disp_l)
            loss_disp = (loss_disp / torch.max(loss_disp) + 1.0) / 2.0
            loss_disp = torch.swapaxes(loss_disp, 0, 2)
            loss_disp = torch.swapaxes(loss_disp, 0, 1)
            loss_disp = torch.squeeze(loss_disp)
            loss_disp = loss_disp.data.cpu().numpy()
            d.update(loss_disp, 'loss')

            # format disparity for output
            out = torch.abs(pred_disp_l)
            out = ((out-torch.min(out)) / (torch.max(out)-torch.min(out)))
            out = torch.swapaxes(out, 0, 2)
            out = torch.swapaxes(out, 0, 1)
            out = torch.squeeze(out)
            out = out.data.cpu().numpy()
            d.update(out, 'disparity')

            right_pred_out = torch.swapaxes(right_pred, 2, 3)
            right_pred_out = torch.swapaxes(right_pred_out, 1, 3)
            right_pred_out = torch.squeeze(right_pred_out)
            right_pred_out = right_pred_out / torch.max(right_pred_out)
            right_pred_out = right_pred_out.data.cpu().numpy()

            d.update(right_pred_out, 'right_pred')

if __name__ == '__main__':
    run()
