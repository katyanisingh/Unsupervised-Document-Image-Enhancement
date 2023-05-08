import torch.nn as nn
import torch.nn.functional as F
import torch

class CRNN(nn.Module):

    def __init__(self,input_nc):
        super(CRNN, self).__init__()
        self.input_nc = input_nc
        self.convo = nn.DataParallel(Convolutional(self.input_nc))
        self.lstm = nn.LSTM(512*3, 256, 2, bidirectional=True)
        self.linear1 = nn.Linear(512*3,512)
        self.linear2 = nn.Linear(512,128)
        self.linear3 = nn.Linear(128, 1)


    def forward(self, x):

        x = self.convo(x)
        x = self.map_to_sequence(x)
        x, _ = self.lstm(x)
        x = torch.squeeze(x)
        x= torch.flatten(x)
        x = self.linear1(x) 
        x = self.linear2(x) 
        x= self.linear3(x)
        return x

    def map_to_sequence(self, map):
        batch, channel, height, width = map.size()
        map= map.permute(3,0,1,2)
        map= map.view(width,batch,-1)
        return map

    
    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

class Convolutional (nn.Module):
    def __init__(self, input_nc):
        super(Convolutional, self).__init__()
        self.conv1 = nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(num_features=512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(num_features=512)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(512,512, kernel_size=3, stride=1, padding=0)
   
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2,2))
        x = F.relu(self.batchnorm1(self.conv5(x)))
        x = F.relu(self.batchnorm2(self.conv6(x)))
        x = F.max_pool2d(x, (2,2))
        x = F.max_pool2d(F.relu(self.conv7(x)),(2,2))
        x= F.max_pool2d(F.relu(self.conv8(x)),(2,2))
        return x

    
