from __future__ import print_function
import argparse
import os
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
from dataset import HDF5Dataset
from hdf5_io import save_hdf5
import dcgan
import numpy as np
np.random.seed(43)

#Change workdir to where you want the files output
work_dir = os.path.expandvars('./')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='3D')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass
opt.manualSeed = 43 # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['3D']:
    dataset = HDF5Dataset(opt.dataroot,
                          input_transform=transforms.Compose([
                          transforms.ToTensor()
                          ]))
assert dataset

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

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
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = dcgan.DCGAN3D_D(opt.imageSize, nz, nc, ndf, ngpu)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()

input, noise, fixed_noise, fixed_noise_TI = None, None, None, None
input = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1, 1)
fixed_noise = torch.FloatTensor(1, nz, 7, 7, 7).normal_(0, 1)
fixed_noise_TI = torch.FloatTensor(1, nz, 1, 1, 1).normal_(0, 1)

label = torch.FloatTensor(opt.batchSize)
real_label = 0.9
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
    fixed_noise_TI = fixed_noise_TI.cuda()

input = Variable(input)
label = Variable(label)
noise = Variable(noise)

fixed_noise = Variable(fixed_noise)
fixed_noise_TI = Variable(fixed_noise_TI)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))

gen_iterations = 0
for epoch in range(opt.niter):
    
    for i, data in enumerate(dataloader, 0):
        f = open(work_dir+"training_curve.csv", "a")
        
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        
        real_cpu = data
            
        batch_size = real_cpu.size(0)
        input.data.resize_(real_cpu.size()).copy_(real_cpu)
        label.data.resize_(batch_size).fill_(real_label)
        
        output = netD(input)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        noise.data.resize_(batch_size, nz, 1, 1, 1)
        noise.data.normal_(0, 1)
        fake = netG(noise).detach()
        label.data.fill_(fake_label)
        output = netD(fake)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()
        
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        g_iter = 1
        while g_iter != 0:
            netG.zero_grad()
            label.data.fill_(1.0) # fake labels are real for generator cost
            noise.data.normal_(0, 1)
            fake = netG(noise)
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.data.mean()
            optimizerG.step()
            g_iter -= 1
        
        gen_iterations += 1

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
        f.write('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
        f.write('\n')
        f.close()
        
    if epoch % 5 == 0:
        fake = netG(fixed_noise)
        fake_TI = netG(fixed_noise_TI)
        save_hdf5(fake.data, work_dir+'fake_samples_{0}.hdf5'.format(gen_iterations))
        save_hdf5(fake_TI.data, work_dir+'fake_TI_{0}.hdf5'.format(gen_iterations))
	
    # do checkpointing
    if epoch % 5 == 0:
        torch.save(netG.state_dict(), work_dir+'netG_epoch_%d.pth' % (epoch))
        torch.save(netD.state_dict(), work_dir+'netD_epoch_%d.pth' % (epoch))
