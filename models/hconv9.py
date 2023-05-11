import torch


class HConv9Est(torch.nn.Module):
    def __init__(self, height_width):
        super().__init__()
        h = height_width[0]
        w = height_width[1]
        self.pool = torch.nn.MaxPool2d(2, stride=2, return_indices=True)
        self.unpool = torch.nn.MaxUnpool2d(2,2)

        self.conv = torch.nn.Conv2d(2, 4, (3,3), padding=(1,1))
        self.conv2 = torch.nn.Conv2d(4, 8, (3,3), padding=(1,1))
        self.conv3 = torch.nn.Conv2d(8, 16, (3,3), padding=(1,1))
        self.conv4 = torch.nn.Conv2d(16, 32, (3,3), padding=(1,1))
        self.conv5 = torch.nn.Conv2d(32, 64, (3,3), padding=(1,1))
        self.conv6 = torch.nn.Conv2d(64, 128, (3,3), padding=(1,1))

        self.conv6t = torch.nn.ConvTranspose2d(128, 64, (3, 3), padding=(1, 1))
        self.conv5t = torch.nn.ConvTranspose2d(64, 32, (3, 3), padding=(1, 1))
        self.conv4t = torch.nn.ConvTranspose2d(32, 16, (3, 3), padding=(1, 1))
        self.conv3t = torch.nn.ConvTranspose2d(16, 8, (3, 3), padding=(1, 1))
        self.conv2t = torch.nn.ConvTranspose2d(8, 4, (3, 3), padding=(1, 1))
        self.convt = torch.nn.ConvTranspose2d(4, 1, (3, 3), padding=(1, 1))


    def forward(self, l, r):
        lr = torch.cat((l,r), dim=1)
        lr1 = self.conv.forward(lr)
        lr1s, p1i = self.pool.forward(lr1)
        lr2 = self.conv2.forward(lr1s)
        lr2s, p2i = self.pool.forward(lr2)
        lr3 = self.conv3.forward(lr2s)
        lr3s, p3i = self.pool.forward(lr3)
        lr4 = self.conv4.forward(lr3s)
        lr4s, p4i = self.pool.forward(lr4)
        lr5 = self.conv5.forward(lr4s)
        lr5s, p5i = self.pool.forward(lr5)
        lr6 = self.conv6.forward(lr5s)

        lr6t = self.conv6t.forward(lr6)
        lr6u = self.unpool.forward(lr6t, p5i, output_size=lr5.size())
        lr5t = self.conv5t.forward(lr6u)
        lr5u = self.unpool.forward(lr5t, p4i, output_size=lr4.size())
        lr4t = self.conv4t.forward(lr5u)
        lr4u = self.unpool.forward(lr4t, p3i, output_size=lr3.size())
        lr3t = self.conv3t.forward(lr4u)
        lr3u = self.unpool.forward(lr3t, p2i, output_size=lr2.size())
        lr2t = self.conv2t.forward(lr3u)
        lr2u = self.unpool.forward(lr2t, p1i, output_size=lr1.size())
        lr1t = self.convt.forward(lr2u)
        #lr1u = self.unpool.forward(lr1t, p1i)

        return lr1t