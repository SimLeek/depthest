from loss_functions.toy import toy_loss
from train_images.toy import left_img, right_img
from models.fixed_depth import FixedDepthEst
import torch


def run():
    model = FixedDepthEst(list(left_img.shape[-2:]))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, betas=(0.9, 0.999))
    model.train()
    optimizer.zero_grad()

    while True:
        depth = model.forward()
        loss = toy_loss(depth)
        print(loss)
        print(depth)
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    run()
