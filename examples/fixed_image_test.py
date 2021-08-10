from loss_functions.img_diff import pyramidal_loss
from train_images import box_left_2, box_right_2
from models.fixed_depth import FixedDepthEst
import torch
from visualize.right_from_left import right_from_left
from loss_functions.toy import toy_loss
from displayarray import display
from sparsepyramids.edge_pyramid import edge
import torch.nn.functional as F

def format_torch_image_for_display(t: torch.Tensor):
    # todo: move to displayarray
    assert len(list(t.shape)) == 3 or t.shape[
        0] == 1, "tensor to be displayed should be an image or have a batch size of one. If you want to display an entire batch, you can run a for loop over the batch dimension."
    t = torch.squeeze(t)
    if len(list(t.shape))==3:
        t = torch.permute(t, (1, 2, 0))
    tc = t.cpu().detach().numpy()
    return tc


def rgb_to_grayscale(t: torch.Tensor):
    assert t.shape[1] == 3

    # assuming bgr from opencv
    b = t[:, 0:1, ...]
    g = t[:, 1:2, ...]
    r = t[:, 2:3, ...]

    o = 0.299 * r + 0.587 * g + 0.114 * b
    return o


def run():
    torch_left_image = torch.FloatTensor(box_left_2)
    torch_right_image = torch.FloatTensor(box_right_2)

    torch_left_image = torch.permute(torch_left_image, (2, 0, 1))[None,]
    torch_right_image = torch.permute(torch_right_image, (2, 0, 1))[None,]

    torch_left_image = rgb_to_grayscale(torch_left_image)
    torch_right_image = rgb_to_grayscale(torch_right_image)

    torch_left_edge = F.conv2d(torch_left_image, edge, padding=1)
    torch_right_edge = F.conv2d(torch_right_image, edge, padding=1)

    model = FixedDepthEst(list(torch_left_image.shape[-2:]))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, betas=(0.9, 0.999))
    model.train()
    optimizer.zero_grad()

    torch_right_edge_display = torch_right_edge-torch.min(torch_right_edge) / (2*(torch.max(torch_right_edge)-torch.min(torch_right_edge) + 1e-7))

    d = display()
    while True:
        disp = model.forward()
        guess_right = right_from_left(torch_left_image, disp)
        loss = pyramidal_loss(torch_left_image, torch_right_image, guess_right, disp)
        # print(loss)
        # print(disp)
        loss.backward()
        optimizer.step()

        disp = disp/(torch.max(disp)+1e-7)
        #guess_right = guess_right - torch.min(guess_right) / (2*(
        #            torch.max(guess_right) - torch.min(guess_right) + 1e-7))

        d.update(format_torch_image_for_display(disp), 'disparity')
        d.update(format_torch_image_for_display(guess_right)/255, 'predicted right')
        #d.update(format_torch_image_for_display(torch_right_edge_display), 'actual right edge')
        d.update(format_torch_image_for_display(torch_right_image)/255, 'actual right image')


if __name__ == '__main__':
    run()
