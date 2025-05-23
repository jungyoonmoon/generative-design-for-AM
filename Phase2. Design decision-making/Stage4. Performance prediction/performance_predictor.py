import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error
from natsort import natsorted

import math


from torchsummary import summary

class CNN3D(nn.Module):
    def __init__(self,conv_dim,fc_dim):
        super(CNN3D, self).__init__()

        #Dimensions
        self.conv_dim = conv_dim #conv_dim = 8
        self.fc_dim = fc_dim #fc_dim = 64
        self.vol_dim = 1
        self.surface_dim = 1

        #Convolution layers
        self.conv1 = nn.Conv3d(1, self.conv_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv3d(self.conv_dim, self.conv_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.lrelu2 = nn.LeakyReLU(0.1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv3d(self.conv_dim, self.conv_dim*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.lrelu3 = nn.LeakyReLU(0.1)
        self.conv4 = nn.Conv3d(self.conv_dim*2, self.conv_dim*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.lrelu4 = nn.LeakyReLU(0.1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv3d(self.conv_dim*2, self.conv_dim*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.lrelu5 = nn.LeakyReLU(0.1)
        self.conv6 = nn.Conv3d(self.conv_dim*4, self.conv_dim*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.lrelu6 = nn.LeakyReLU(0.1)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv3d(self.conv_dim*4, self.conv_dim*8, kernel_size=3, stride=1, padding=1, bias=True)
        self.lrelu7 = nn.LeakyReLU(0.1)
        self.conv8 = nn.Conv3d(self.conv_dim*8, self.conv_dim*8, kernel_size=3, stride=1, padding=1, bias=True)
        self.lrelu8 = nn.LeakyReLU(0.1)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        #FC-layers
        self.fc_vol = nn.Linear(self.vol_dim,self.vol_dim)
        self.fc_surface = nn.Linear(self.surface_dim,self.surface_dim)
        self.fc1 = nn.Linear(self.conv_dim*4096+self.vol_dim+self.surface_dim, self.fc_dim*64)
        self.lrelu5 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(self.fc_dim*64, self.fc_dim*16) #In: 4096, Out: 1024
        self.lrelu6 = nn.LeakyReLU(0.1)
        self.fc3 = nn.Linear(self.fc_dim*16,self.fc_dim*4) #In: 1024, Out: 256
        self.lrelu7 = nn.LeakyReLU(0.1)
        self.fc4 = nn.Linear(self.fc_dim*4, self.fc_dim*2) #In: 256, Out: 128
        self.lrelu8 = nn.LeakyReLU(0.1)
        self.fc5 = nn.Linear(self.fc_dim*2, self.fc_dim) #In: 128, Out: 64
        self.lrelu9 = nn.LeakyReLU(0.1)
        self.fc6 = nn.Linear(self.fc_dim, 1)

        #initializer
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0)
        nn.init.xavier_normal_(self.conv3.weight)
        nn.init.constant_(self.conv3.bias, 0)
        nn.init.xavier_normal_(self.conv4.weight)
        nn.init.constant_(self.conv4.bias, 0)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.constant_(self.fc4.bias, 0)
        nn.init.xavier_normal_(self.fc5.weight)
        nn.init.constant_(self.fc5.bias, 0)
        nn.init.xavier_normal_(self.fc6.weight)
        nn.init.constant_(self.fc6.bias, 0)


    def forward(self, x1, vol, surface):

        #print(x1.shape)
        x2 = self.lrelu1(self.conv1(x1))
        # print('conv1 = ',x2.shape)
        x3 = self.lrelu2(self.conv2(x2))
        #print('conv2 = ',x3.shape)
        x4 = self.pool1(x3)
        #print('pool1 = ',x4.shape)


        x5 = self.lrelu1(self.conv3(x4))
        #print('conv3 = ',x5.shape)
        x6 = self.lrelu2(self.conv4(x5))
        #print('conv4 = ',x6.shape)
        x7 = self.pool1(x6)
        #print('pool2 = ',x7.shape)

        x8 = self.lrelu1(self.conv5(x7))
        #print('conv5 = ',x8.shape)
        x9 = self.lrelu2(self.conv6(x8))
        #print('conv6 = ',x9.shape)
        x10 = self.pool1(x9)
        #print('pool3 = ',x10.shape)

        x11 = self.lrelu1(self.conv7(x10))
        #print('conv7 = ',x11.shape)
        x12 = self.lrelu2(self.conv8(x11))
        #print('conv8 = ',x12.shape)
        x13 = self.pool1(x12)
        #print('pool4 = ',x13.shape)


        x13 = x13.view(x13.size(0),-1)

        x_vol = self.fc_vol(vol)
        x_surface = self.fc_surface(surface)
        # print(x13)
        # print(x13.shape)
        # print(x_vol)
        # print(x_vol.shape)
        # print(x_surface)
        # print(x_surface.shape)
        x14 = torch.cat((x13, x_vol, x_surface), dim=1)  # Concatenate x1 and x2 along the channel dimension
        #print('concat = ',x14.shape)
        x14 = x14.view(x14.size(0), -1)

        x15 = self.lrelu5(self.fc1(x14))
        #print(x15.shape)
        x16 = self.lrelu6(self.fc2(x15))
        #print(x16.shape)
        x17 = self.lrelu7(self.fc3(x16))
        #print(x17.shape)
        x18 = self.lrelu8(self.fc4(x17))
        #print(x18.shape)
        x19 = self.lrelu9(self.fc5(x18))
        #print(x19.shape)
        x20 = self.fc6(x19)
        return x20


########## ?WHICH MODEL? ###########
OUTPUT_performance = 'stress'  # 'support' or 'time' or 'stress'
epochs = 300
res = 128

###################   DATA   #######################
# load X
X_PATH = './data/generated design set/npy'
list_dir = os.listdir(X_PATH)
print(list_dir)
# list_dir = natsorted(os.listdir(X_PATH))
x_list = list()

# load volume and surface
prep_PATH = './data/generated design set/vol_surf/vol_surf.xlsx'
prep_df = pd.read_excel(prep_PATH)

idx_prep = prep_df['ID'].to_numpy()

vol = prep_df['Volume'].to_numpy()
surf = prep_df['Surface'].to_numpy()
print(idx_prep)
print(vol)
print(surf)
print('-----------------------')

vol = (vol - min(vol)) / (max(vol) - min(vol))
surf = (surf - min(surf)) / (max(surf) - min(surf))

# load model
PATH = f'./weights/{OUTPUT_performance}'
w_path = os.path.join(PATH, f'{OUTPUT_performance}_model_ep{epochs}.pt')
model = torch.load(w_path)
out_list = list()

#model inference
for i in range(len(list_dir)):
    x_np = np.load(os.path.join(X_PATH, list_dir[i]))
    x = torch.tensor(x_np.reshape(1, 1, res, res, res)).cuda().float()
    input_vol = torch.tensor(vol[i]).cuda().float()
    input_surf = torch.tensor(surf[i]).cuda().float()

    input_vol = input_vol.reshape(1,1)
    input_surf = input_surf.reshape(1, 1)
    out = model(x, input_vol,input_surf)
    out_list.append(out.item())
    print(list_dir[i])
    print(idx_prep[i])
    print(out)

save_path = r'./prediction results'
df = pd.DataFrame([x for x in zip(list_dir,out_list)])
df.to_excel(save_path +'/'+ f'output_{OUTPUT_performance}_model_ep{epochs}_.xlsx')
