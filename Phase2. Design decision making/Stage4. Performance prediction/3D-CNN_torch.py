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


if __name__ == '__main__':
    ############# Hyper-parameters ############

    res = 128
    conv_dim = 8
    fc_dim = 64
    batch_size = 32
    epochs = 500
    device = 'cuda'
    loss_func = 'MSE'
    learning_rate = 0.000005
    out = 'stress'
    #########################################
    if loss_func == 'MSE':
        def network_loss (y,y_hat):
            return torch.mean((y - y_hat) ** 2)
    elif loss_func == 'MAE':
        def network_loss (y,y_hat):
            return torch.mean(torch.abs(y - y_hat))
    elif loss_func == 'MSLE':
        def network_loss (y,y_hat):
            return torch.mean((torch.log(y+1)-torch.log(y_hat+1))**2)

    cuda = torch.device(device)
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))

    ###################   DATA   #######################
    weight_PATH = f'./weights/{out}'
    #load X
    X_PATH = './data/X/npy'
    list_dir = natsorted(os.listdir(X_PATH))
    print(list_dir)

    x_list = list()
    for i in range(len(list_dir)):
        print(os.path.join(X_PATH,list_dir[i]))
        temp = np.load(os.path.join(X_PATH,list_dir[i]))
        x_list.append(temp)
        print(temp.shape)
    x = np.array(x_list)
    print(x_list)
    print(x.shape)
    x = x.reshape(len(x),1,res,res,res)

    # print()

    #load preprocessed data (volume, surface, y_time, y_support)
    prep_PATH = './data/X/vol_surf/vol_surf.xlsx'
    prep_df = pd.read_excel(prep_PATH)
    vol = prep_df['volume'].to_numpy()
    surf = prep_df['surface_area'].to_numpy()

    y_PATH =f'./data/y/{out}/y.xlsx'
    y_df = pd.read_excel(y_PATH)
    y = y_df['y'].to_numpy()



    vol = (vol-min(vol))/(max(vol)-min(vol))
    surf =(surf-min(surf))/(max(surf)-min(surf))
    # vol = np.log(vol)
    # surf = np.log(surf)
    y = np.log(y)



    vol = vol.reshape(len(vol),1)
    surf = surf.reshape(len(surf),1)
    y = y.reshape(len(y),1)



    print(vol.shape)
    print(surf.shape)
    print(y_time.shape)
    print(y_support.shape)

    #train-test split
    x1_train, x1_test, vol_train, vol_test,  surf_train, surf_test ,\
    y_train,y_test \
        = train_test_split(x, vol,surf, y, test_size=0.2)
    x1_train = torch.tensor(x1_train)
    x1_test = torch.tensor(x1_test)
    vol_train = torch.tensor(vol_train)
    vol_test = torch.tensor(vol_test)
    surf_train = torch.tensor(surf_train)
    surf_test = torch.tensor(surf_test)
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)


    # print(x1_train.shape)
    # print(x1_test.shape)
    # print(vol_train.shape)
    # print(vol_test.shape)
    # print(surf_train.shape)
    # print(surf_test.shape)
    # print(y_support_train.shape)
    # print(y_support_test.shape)
    ############################Training#############################################
    train_dataset = TensorDataset(x1_train, vol_train, surf_train, y_train)
    test_dataset = TensorDataset(x1_test, vol_test, surf_test, y_test)
    # batch_size = batch_size
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    ####################################################
    model = CNN3D(conv_dim=conv_dim, fc_dim=fc_dim)
    model = model.cuda().float()
    writer = SummaryWriter()

    train_loss_list = list()
    val_loss_list = list()
    val_MAPE_list = list()
    val_RMSE_list = list()
    eposh_list = list()
    #print(model)
    for epoch in range(epochs):
        for x1_batch, vol_batch, surf_batch, label_batch in dataloader:
            x1_batch = x1_batch.cuda().float()
            vol_batch = vol_batch.cuda().float()
            surf_batch = surf_batch.cuda().float()
            label_batch = label_batch.cuda().float()
            # print(111)
            # print(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            output = model(x1_batch,vol_batch,surf_batch).cuda()
            err_=network_loss(label_batch,output)
            err_.backward()
            optimizer.step()
            #print(model)
        train_loss = err_ / len(dataloader)

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        RMSE_total = 0
        MAPE_total = 0
        val_err = 0.0

        with torch.no_grad():
            for x1_test, vol_test, surf_test, label_test in test_dataloader:
                x1_test = x1_test.cuda().float()
                vol_test = vol_test.cuda().float()
                surf_test = surf_test.cuda().float()
                label_test = label_test.cuda().float()

                outputs = model(x1_test, vol_test, surf_test)
                val_err += network_loss(outputs, label_test).item()
                outputs_value = outputs.data.cpu().numpy()
                label_test_value = label_test.cpu().numpy()

                MSE = mean_squared_error(label_test_value, outputs_value)
                RMSE = np.sqrt(MSE)
                MAPE= np.mean(np.abs((label_test_value - outputs_value) / label_test_value)) * 100

                RMSE_total += RMSE
                MAPE_total += MAPE

        # Calculate validation loss and accuracy for the epoch
        val_RMSE = RMSE_total/len(test_dataloader)
        val_MAPE = MAPE_total / len(test_dataloader)
        val_loss = val_err/len(test_dataloader)

        eposh_list.append(epoch)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_MAPE_list.append(val_MAPE)
        val_RMSE_list.append(val_RMSE)

        # Log the training and validation metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('RMSE/validation', val_RMSE, epoch)
        writer.add_scalar('MAPE/validation', val_MAPE, epoch)



        print(
            f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, RMSE: {val_RMSE:.4f}, MAPE: {val_MAPE:.4f}")
        if epoch % 100 == 0:
            torch.save(model, weight_PATH + '/' + f'{out}_model_ep{epoch}.pt')
            print(f'[Save {out}_model_ep{epoch}.pt]')

    # Close TensorBoard writer
    writer.close()


    ############ weight save ###############

    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, weight_PATH +'/'+ f'{out}_all_ep{epochs}.tar')

    df = pd.DataFrame([x for x in zip(eposh_list,train_loss_list, val_loss_list,val_MAPE_list,val_RMSE_list)])
    df.to_excel(weight_PATH +'/'+ f'loss_{out}_model_ep{epochs}.xlsx')



    # print(model)
    # model.to('cuda')
    # model.train()

    # Print the model architecture
    # print(model)
    # print('-------------------------SUMMARY-----------------------')
    # summary(model(x1,vol_dim,surface_dim),(1,256,256,256))
