from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os
import numpy as np
from dataset import HDF5Dataset
from hdf5_io import save_hdf5
import dcgan

parser = argparse.ArgumentParser()
parser.add_argument('--seedmin', type=int, default=43)
parser.add_argument('--seedmax', type=int, default=64)
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--imsize', type=int, default=1, help='the height of the z tensor')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
opt = parser.parse_args()
print(opt)

if opt.experiment is None:
    opt.experiment = 'samples'
os.system('mkdir {0}'.format(opt.experiment))

for seed in range(opt.seedmin, opt.seedmax, 1):
    random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    nc = 1
    
    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    netG = dcgan.DCGAN3D_G(opt.imageSize, nz, nc, ngf, ngpu)
    netG.apply(weights_init)
    if opt.netG != '': # load checkpoint if needed
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)
    
    fixed_noise = torch.FloatTensor(1, nz, opt.imsize, opt.imsize, opt.imsize).normal_(0, 1)

    if opt.cuda:
        netG.cuda()
        fixed_noise = fixed_noise.cuda()

    fixed_noise = Variable(fixed_noise)

    fake = netG(fixed_noise)

    save_hdf5(fake.data, '{0}/{1}_{2}.hdf5'.format(opt.experiment, opt.experiment, seed))