import torch
import torch.nn as nn

class _netD64(nn.Module):
    def __init__(self, ngpu, num_classes=1, nc=3, ndf=64):
        super(_netD64, self).__init__()
        self.ngpu = ngpu
        self.num_classes = num_classes
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            #nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Conv2d(ndf * 8, num_classes, 4, 1, 0, bias=False),
            # out size = batch x num_classes x 1 x 1
            #nn.Sigmoid()
        )

        if self.num_classes == 1:
          self.main.add_module('prob', nn.Sigmoid())
          # output = probability
        else:
          pass
          # output = scores

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(input.size(0), self.num_classes).squeeze(1)


class _netD(nn.Module):
    def __init__(self, ngpu, num_classes=1, nc=3, ndf=64):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.num_classes = num_classes
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            # conv2D(in_channels, out_channels, kernelsize, stride, padding)
            nn.Conv2d(nc, ndf , 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 2 x 2
            nn.Conv2d(ndf * 8, num_classes, 2, 1, 0, bias=False),
            # out size = batch x num_classes x 1 x 1
        )

        if self.num_classes == 1:
          self.main.add_module('prob', nn.Sigmoid())
          # output = probability
        else:
          pass
          # output = scores

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(input.size(0), self.num_classes).squeeze(1)



class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class _netD_v2(nn.Module):
    def __init__(self, ngpu, num_classes=1, nc=3, ndf=64):
        super(_netD_v2, self).__init__()
        self.ngpu = ngpu
        self.num_classes = num_classes
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            # conv2D(in_channels, out_channels, kernelsize, stride, padding)
            nn.Conv2d(nc, ndf , 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 2 x 2
            Reshape(-1, ndf*8*2*2),
            nn.Linear(ndf*8*2*2, num_classes),
            # Note: the difference from v1 is: using linear at the last layer
            # and use bias=True
        )

        if self.num_classes == 1:
          self.main.add_module('prob', nn.Sigmoid())
          # output = probability
        else:
          pass
          # output = scores

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(input.size(0), self.num_classes).squeeze(1)

class _netD_synth(nn.Module):
    def __init__(self, ngpu, dimx=100, leaky_inplace=False):
        super(_netD_synth, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(dimx, 1000, bias=True),
            nn.LeakyReLU(0.2, inplace=leaky_inplace),
            nn.Linear(1000, 1, bias=True)
            # This has two classes (one logit)
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(input.size(0), self.num_classes).squeeze(1)