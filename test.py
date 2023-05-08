#!/usr/bin/python3

import argparse
import sys
import os
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from PIL import Image
from models_proposed import Generator
from datasets import ImageDataset


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='datasets/kaggle/', help='root directory of the dataset')
    parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=6, help='number of cpu threads to use during batch generation')
    parser.add_argument('--generator_A2B', type=str, default='./models/saved_models/netG_A2B_kaggle.pth', help='A2B generator checkpoint file')
    parser.add_argument('--filename',type=str, default="abc")
    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


    netG_A2B = Generator(opt.input_nc, opt.output_nc)

    torch.backends.cudnn.enabled = False

    if opt.cuda:
        netG_A2B.cuda()

    # Load state dicts
    netG_A2B.load_state_dict(torch.load(opt.generator_A2B))

    # Set model's test mode
    netG_A2B.eval()

 

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor

    # Dataset loader
    transforms_ = [ transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                    ]
    dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test'), 
                            batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)


    # Create output dirs if they don't exist
    if not os.path.exists('./output/'+opt.filename+'/B'):
        os.makedirs('./output/'+opt.filename+'/B')
    if not os.path.exists('./input/'+opt.filename+'/A'):
        os.makedirs('./input/'+opt.filename+'/A')


    for i, batch, in enumerate(dataloader):
        # Set model input

        width= batch['A'].shape[2]
        height= batch['A'].shape[3]
        input_A = Tensor(opt.batchSize, opt.input_nc,width, height)
        real_A = Variable(input_A.copy_(batch['A']))
        real_A_path = batch['Path_A']


        # Generate output
        fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
        path_save= real_A_path[0].split('/')

        # Save image files
        save_image(fake_B, './output_final/'+opt.filename+'/B/'+str(path_save[-1]))
        save_image(real_A, './input_final/'+opt.filename+'/A/%04d.png' % (i+1))

        sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

    sys.stdout.write('\n')
    ###################################
