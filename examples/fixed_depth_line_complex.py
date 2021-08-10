from loss_functions.img_diff import pyramidal_loss
from train_images.toy import left_img, right_img
from models.fixed_depth import FixedDepthEst
import torch
from visualize.right_from_left import right_from_left
from loss_functions.toy import toy_loss

def run():
    model = FixedDepthEst(list(left_img.shape[-2:]))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, betas=(0.9, 0.999))
    model.train()
    optimizer.zero_grad()

    torch_left_image = torch.FloatTensor(left_img)
    torch_right_image = torch.FloatTensor(right_img)

    while True:
        disp = model.forward()
        guess_right = right_from_left(torch_left_image, disp)
        loss = pyramidal_loss(torch_left_image, torch_right_image, guess_right, disp)
        print(loss)
        print(disp)
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    run()
