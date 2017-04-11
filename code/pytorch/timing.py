from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import os
import numpy as np
from dataset import HDF5Dataset
from hdf5_io import save_hdf5
import dcgan
import time

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=43)
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--imsize', type=int, default=1, help='the height of the z tensor')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
opt = parser.parse_args()
print(opt)

if opt.experiment is None:
    opt.experiment = 'samples'
os.system('mkdir {0}'.format(opt.experiment))

repeat = opt.repeat
seed = opt.seed
random.seed(seed)
torch.manual_seed(seed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
	print("WARNING: You have a CUDA device, so you should probably run with --cuda")

ngpu = int(opt.ngpu)
nz = int(opt.nz)
nc = 1
ngf = int(opt.ngf)
ndf = int(opt.ndf)

# custom weights initialization called on netG and netD
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

def weights_bias(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.bias.data.fill_(0)

netG = dcgan.DCGAN3D_G_CPU(opt.imageSize, nz, nc, ngf, ngpu)
netG.apply(weights_init)
try:
    if opt.netG != '': # load checkpoint if needed
	netG.load_state_dict(torch.load(opt.netG))
except Exception:
    pass
netG.apply(weights_bias)

if opt.cuda:
    netG = netG.cuda()
else:
    netG = netG.cpu().float()

for im_size in range(1, 29, 1):
    fixed_noise = torch.FloatTensor(1, nz, im_size, im_size, im_size).normal_(0, 1)
    
    if opt.cuda:
        fixed_noise = fixed_noise.cuda()

    fixed_noise = Variable(fixed_noise)
    times = np.zeros(repeat, dtype=np.float64)
    t = 0.0
    fake = None
    for i in range(repeat):
        
        now = time.clock()
        try:
            fake = netG(fixed_noise)
        except RuntimeError:
            print("Couldn't run")
        post = time.clock()
        dt = post-now
	t += dt
        times[i] = dt
    print(str(fake.size()[2])+', {:16f}, {:16f}, {:16f}'.format(t/float(repeat), np.mean(times), np.std(times)))