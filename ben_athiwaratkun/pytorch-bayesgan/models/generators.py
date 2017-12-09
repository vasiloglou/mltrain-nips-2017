import torch
import torch.nn as nn

class _netG64(nn.Module):
    def __init__(self, ngpu, nz=100, ngf=64, nc=3):
        super(_netG64, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class _netG(nn.Module):
    def __init__(self, ngpu, nz=100, ngf=64, nc=3, batch_norm_layers=[], affine=True):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        if batch_norm_layers == []:
            print("Initializing the Batch Norm layers. Affine = {}".format(affine))
            batch_norm_layers.append(nn.BatchNorm2d(ngf * 8, affine=affine))
            batch_norm_layers.append(nn.BatchNorm2d(ngf * 4, affine=affine))
            batch_norm_layers.append(nn.BatchNorm2d(ngf * 2, affine=affine))
        else:
            assert len(batch_norm_layers) == 3
            #print("Reusing the Batch Norm Layers")
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            #nn.BatchNorm2d(ngf * 8),
            batch_norm_layers[0],
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf * 4),
            batch_norm_layers[1],
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf * 2),
            batch_norm_layers[2],
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     nc, 4, 2, 1, bias=False),
            # state size. (nc) x 32 x 32
            nn.Tanh()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class _netG_v2(nn.Module):
    def __init__(self, ngpu, nz=100, ngf=64, nc=3, batch_norm_layers=[], affine=True):
        super(_netG_v2, self).__init__()
        self.ngpu = ngpu
        if batch_norm_layers == []:
            print("Initializing the Batch Norm layers. Affine = {}".format(affine))
            batch_norm_layers.append(nn.BatchNorm2d(ngf * 8, affine=affine))
            batch_norm_layers.append(nn.BatchNorm2d(ngf * 4, affine=affine))
            batch_norm_layers.append(nn.BatchNorm2d(ngf * 2, affine=affine))
            batch_norm_layers.append(nn.BatchNorm2d(ngf * 1, affine=affine))
        else:
            assert len(batch_norm_layers) == 4
            print("Reusing the Batch Norm Layers")
        self.main = nn.Sequential(
            # input is Z, going into a a linear layer and reshape
            Reshape(-1, nz),
            nn.Linear(in_features=nz, out_features=2*2*ngf*8, bias=True),
            Reshape(-1, ngf*8, 2, 2),
            #ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
            # in: 2 x 2 with kernel 5, stride 2, padding 2
            batch_norm_layers[0],
            nn.ReLU(True),
            nn.ConvTranspose2d(  ngf*8, ngf * 4, kernel_size=4, stride=2, padding=1),
            batch_norm_layers[1],
            nn.ReLU(True),
            nn.ConvTranspose2d(  ngf*4, ngf * 2, 4, stride=2, padding=1),
            batch_norm_layers[2],
            nn.ReLU(True),
            nn.ConvTranspose2d(  ngf*2, ngf * 1, 4, stride=2, padding=1),
            batch_norm_layers[3],
            nn.ReLU(True),
            nn.ConvTranspose2d(  ngf*1,      nc, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class _netG_synth(nn.Module):
    def __init__(self, ngpu, nz=2, dimx=100, leaky_inplace=False):
        super(_netG_synth, self).__init__()
        self.ngpu = ngpu
        # map 2 dim to 100 dim
        self.main = nn.Sequential(
            nn.Linear(nz, 1000, bias=True),
            nn.LeakyReLU(0.2, inplace=leaky_inplace),
            nn.Linear(1000, dimx)
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output