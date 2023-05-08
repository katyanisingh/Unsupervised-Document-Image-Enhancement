import argparse
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.autograd import Variable
from torch.autograd import grad
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from models_proposed import Generator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import weights_init_normal
from datasets import ImageDataset
from model_crnn_proposed import CRNN
from transform_helper import PadWhite
import json
import wandb  
import random

if __name__ == '__main__':

    random_seed = random.randint(0,100)
    print("Using random seed: ", random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    wandb.init(project='GAN', entity='')  # if wandb integration needed replace entity name
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, default='./params.json')
    parser.add_argument('--filename',type=str, default="model_1")
    parser.add_argument('--wandb_run_name') 
    opt = parser.parse_args()
   
    with open(opt.params, 'r', encoding='utf-8') as f:
        config = json.loads(f.read())
    wandb.config.update(config)
    wandb.run.name = opt.wandb_run_name

    if torch.cuda.is_available() and not config["cuda"]:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


    ###### Definition of variables ######

    netG_A2B = Generator(config["input_nc"], config["output_nc"])
    netG_B2A = Generator(config["output_nc"], config["input_nc"])


    netD_A = CRNN(config["input_nc"])
    netD_B = CRNN(config["output_nc"])

    if config["cuda"]:
        netG_A2B.cuda()
        netG_B2A.cuda()
        netD_A.cuda()
        netD_B.cuda()
       
    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)

    #LOSS FUNCTIONS
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
  
   
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                    lr=config["lr"], betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=config["lr"], betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=config["lr"], betas=(0.5, 0.999))


    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(config["n_epochs"], config["epoch"], config["decay_epoch"]).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(config["n_epochs"], config["epoch"], config["decay_epoch"]).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(config["n_epochs"], config["epoch"], config["decay_epoch"]).step)


    Tensor = torch.cuda.FloatTensor if config["cuda"] else torch.Tensor
    input_A = Tensor(config["batchSize"], config["input_nc"], config["size"], config["size"])
    input_B = Tensor(config["batchSize"], config["output_nc"], config["size"], config["size"])
    target_real = Variable(Tensor(config["batchSize"]).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(config["batchSize"]).fill_(0.0), requires_grad=False)


    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Dataset loader
    transforms_ = [ PadWhite((config["size"],config["size"])),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.5)) ]
    dataloader = DataLoader(ImageDataset(config["dataroot"], transforms_=transforms_, unaligned=True), 
                            batch_size=config["batchSize"], shuffle=True, num_workers=config["n_cpu"])


    ###### Training ######
    for epoch in range(config["epoch"], config["n_epochs"]):
        for i, batch in enumerate(dataloader):
  
        # Set model input
            real_A = Variable(input_A.copy_(batch['A']))
            real_B = Variable(input_B.copy_(batch['B']))

            ###### Generators A2B and B2A ######
        
            optimizer_G.zero_grad()

            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            same_B = netG_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B)*config["lambda_identity"]*config["lambda_cycle_ABA"]
            # G_B2A(A) should equal A if real A is fed
            same_A = netG_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A)*config["lambda_identity"]*config["lambda_cycle_BAB"]

            # GAN loss
            fake_B = netG_A2B(real_A)
            if epoch%10==0 and i%100==0:
                wandb.log({"sample_image": wandb.Image(fake_B)})
            
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake,target_real)


            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake,target_real)


            # Cycle loss
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA_1 = criterion_cycle(recovered_A, real_A)
            loss_cycle_ABA = (loss_cycle_ABA_1)*config["lambda_cycle_ABA"]


            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB_1 = criterion_cycle(recovered_B, real_B)
            loss_cycle_BAB= (loss_cycle_BAB_1)*config["lambda_cycle_BAB"]

            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB 
            loss_G.backward()
            optimizer_G.step()
            ###################################

            ###### Discriminator A ######
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)
            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)
            loss_D_A = (loss_D_real + loss_D_fake)*0.5 
            loss_D_A.backward()
            optimizer_D_A.step()

            ###### Discriminator B ######
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)
            
            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)
            loss_D_B = (loss_D_real + loss_D_fake)*0.5 
            loss_D_B.backward()
            optimizer_D_B.step()

            metrics = {"Epoch": epoch,"seed": random_seed, "loss_G": loss_G, "loss_D_A": loss_D_A,
                                "loss_D_B": loss_D_B, "loss_cycle": loss_cycle_ABA + loss_cycle_BAB}
            metrics = {f"Unsupervised/{key}": value for key, value in metrics.items()}
            wandb.log(metrics)
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()


        # Save models checkpoints
        torch.save(netG_A2B.state_dict(), "./models/saved_models/netG_A2B_"+opt.filename+".pth")
        torch.save(netG_B2A.state_dict(), "./models/saved_models/netG_B2A_"+opt.filename+".pth")
        torch.save(netD_A.state_dict(), "./models/saved_models/netD_A_"+opt.filename+".pth")
        torch.save(netD_B.state_dict(),"./models/saved_models/netD_B_"+opt.filename+".pth")
###################################
