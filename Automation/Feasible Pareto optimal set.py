import cc3d
import numpy as np
import os
import shutil
import pandas as pd
import torch
import torch.nn as nn
from stl import mesh
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

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
def TOPSIS(design_id, support, time, stress, w_support, w_time, w_stress):

    # Step 1: Squared
    sqr_support = np.square(support)
    sqr_time = np.square(time)
    sqr_stress = np.square(stress)

    # Step 2: Normalization
    norm_support = (support / np.sqrt(np.sum(sqr_support)))
    norm_time = (time / np.sqrt(np.sum(sqr_time)))
    norm_stress = (stress / np.sqrt(np.sum(sqr_stress)))

    # Step 3: weight multiply
    weighted_support = w_support * norm_support
    weighted_time = w_time * norm_time
    weighted_stress = w_stress * norm_stress

    # Step 4: Ideal solution
    PIS_support = np.min(weighted_support)
    NIS_support = np.max(weighted_support)
    PIS_time = np.min(weighted_time)
    NIS_time = np.max(weighted_time)
    PIS_stress = np.min(weighted_stress)
    NIS_stress = np.max(weighted_stress)

    PIS_support_dist = np.square(PIS_support - weighted_support)
    NIS_support_dist = np.square(NIS_support - weighted_support)
    PIS_time_dist = np.square(PIS_time - weighted_time)
    NIS_time_dist = np.square(NIS_time - weighted_time)
    PIS_stress_dist = np.square(PIS_stress - weighted_stress)
    NIS_stress_dist = np.square(NIS_stress - weighted_stress)

    # Step 5: Closeness
    PIS_dist = np.sqrt(np.sum([PIS_support_dist, PIS_time_dist, PIS_stress_dist], axis=0))
    NIS_dist = np.sqrt(np.sum([NIS_support_dist, NIS_time_dist, NIS_stress_dist], axis=0))
    closeness = NIS_dist / (PIS_dist + NIS_dist)
    closeness_list = closeness.tolist()
    # print(PIS_dist)
    # print(NIS_dist)
    # print(closeness)
    # print(closeness_list)

    # Step 6: Rank
    rank = zip(design_id, closeness_list)
    rank1 = max(rank, key=lambda x: x[1])
    # print(rank)
    # print(rank1)
    rank1_list =  list(rank1)
    return rank1_list[0],rank1_list[1], rank1_list

if __name__ == '__main__':

#### 1. Defective design filtering--------------------------------------------------------------------------------------

    npy_design_path = './1. Generated design dataset/npy'
    stl_path = './1. Generated design dataset/stl'
    normal_npy_PATH ='./2. Feasible design generation set/npy'
    normal_stl_PATH ='./2. Feasible design generation set/stl'
    broken_PATH = './infeasible design dataset'

    iter_npy_names = os.listdir(npy_design_path)
    print(iter_npy_names)
    valid_num=int()
    unvalid_num=int()
    unvalid_list=list()

    connectivity = 26

    for idx in range(len(iter_npy_names)):
        NPY_PATH = os.path.join(npy_design_path, iter_npy_names[idx])
        STL_PATH = os.path.join(stl_path, iter_npy_names[idx].replace('npy','stl'))
        voxel = np.load(NPY_PATH)
        label_out, num_comp = cc3d.connected_components(voxel, return_N=True, connectivity=connectivity)  # free
        # print(label_out)
        if num_comp == 1:
            valid_num +=1
            shutil.copy(NPY_PATH,os.path.join(normal_npy_PATH, iter_npy_names[idx]))
            shutil.copy(STL_PATH, os.path.join(normal_stl_PATH, iter_npy_names[idx].replace('npy','stl')))
        else:
            unvalid_num+=1
            unvalid_list.append(iter_npy_names[idx][0:-4])
            shutil.copy(NPY_PATH, os.path.join(broken_PATH, iter_npy_names[idx]))
            shutil.copy(STL_PATH, os.path.join(broken_PATH, iter_npy_names[idx].replace('npy','stl')))

            print(iter_npy_names[idx][0:-4])


    print("normal numbers: ",valid_num)
    print("abnormal numbers: ",unvalid_num)
    print("abnormal list: ", unvalid_list)
####--------------------------------------------------------------------------------------------------------------------


#### 2. Performance prediction for generated feasible design dataset----------------------------------------------------

    PATH = './2. Feasible design generation set/stl'
    SAVE_PATH = './2. Feasible design generation set/vol_surf'
    dir_list = os.listdir(PATH)

    print(dir_list)

    vol_list = list()
    surf_list = list()

    for i in range(len(dir_list)):
        your_mesh = mesh.Mesh.from_file(os.path.join(PATH,f'{dir_list[i]}'))
        normals = your_mesh.normals

        # Calculate the triangle area
        a = your_mesh.vectors[:, 0, :]
        b = your_mesh.vectors[:, 1, :]
        c = your_mesh.vectors[:, 2, :]
        cross_product = np.cross(b-a, c-a)
        #surface
        areas = 0.5 * np.sqrt(np.sum(cross_product**2, axis=1))
        total_areas = np.sum(areas)
        #volume
        volumes = np.abs(np.sum(a * cross_product, axis=0)) / 6.0
        total_volume = np.sum(volumes)

        surf_list.append(total_areas)
        vol_list.append(total_volume)
        print('Surface area:', total_areas)
        print('Volume area:', total_volume)

    dir_list = [file.replace('.stl','') for file in dir_list]

    df = pd.DataFrame({'ID': dir_list, 'Surface': surf_list, 'Volume': vol_list})
    df.to_excel(os.path.join(SAVE_PATH,'vol_surf.xlsx'))



    ########## ?WHICH MODEL? ###########
    output = ['support', 'time', 'stress']
    output_support = list()
    output_time = list()
    output_stress = list()

    for performance in output:

        OUTPUT_performance = performance  # 'support' or 'time' or 'stress'
        epochs = 300
        res = 128

        ###################   DATA   #######################
        # load X
        X_PATH = './2. Feasible design generation set/npy'
        list_dir = os.listdir(X_PATH)
        print(list_dir)
        # list_dir = natsorted(os.listdir(X_PATH))
        x_list = list()

        # load volume and surface
        prep_PATH = './2. Feasible design generation set/vol_surf/vol_surf.xlsx'
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
        PATH = f'./3. Weights/{OUTPUT_performance}'
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
            if performance == 'support':
                output_support.append(np.exp(out.item()))
            elif performance == 'time':
                output_time.append(np.exp(out.item()))
            elif performance == 'stress':
                output_stress.append(np.exp(out.item()))

        new_list_dir = list()

        for j in range(len(list_dir)):
            temp = list_dir[j].replace('z_out', '')
            temp1 = temp.replace('.npy', '')
            new_list_dir.append(temp1)

        print(new_list_dir)

    pred_save_path = './4. Prediction results'
    df = pd.DataFrame([x for x in zip(new_list_dir, output_support, output_time, output_stress)],
    columns=["Design", "Support", "Time", "Stress"])
    df.to_excel(pred_save_path +'/'+ 'prediction_results.xlsx')
####--------------------------------------------------------------------------------------------------------------------


#### 3. Construct the Pareto optimal set-------------------------------------------------------------------------------

    PATH = os.path.join('./4. prediction results/prediction_results.xlsx')
    # print(PATH)

    df = pd.read_excel(PATH)

    design_id = df['Design'].tolist()
    support = df['Support'].to_numpy()
    time = df['Time'].to_numpy()
    stress = df['Stress'].to_numpy()

    design_id_list = list()

    w_support_list=list()
    w_time_list=list()
    w_stress_list=list()

    closeness_list = list()
    rank1_list = list()

    # Construct the efficient frontier
    for _ in range(100000):
        weights = np.random.dirichlet([1, 1, 1])
        w_support, w_time, w_stress = weights


        # w_support_list.append(w_support)
        # w_time_list.append(w_time)
        # w_stress_list.append(w_stress)

        print('w_support:',w_support)
        print('w_time:',w_time)
        print('w_stress:',w_stress)

        _, _, rank1 = TOPSIS(design_id, support, time, stress, w_support, w_time, w_stress)
        rank1.append(w_support)
        rank1.append(w_time)
        rank1.append(w_stress)
        rank1_list.append(rank1)
        # rank1_list.append(w_support)
        # rank1_list.append(w_time)
        # rank1_list.append(w_stress)
    print(rank1_list)
    rank1_df = pd.DataFrame(rank1_list, columns=['Design ID','Closeness','w_support','w_time','w_stress'])
    rank1_df.to_excel('5. Pareto optimal set/efficient_frontier.xlsx')
####--------------------------------------------------------------------------------------------------------------------


#### 4. Visualize the Pareto optimal set ------------------------------------------------------------------------------

    matplotlib.use('Qt5Agg')
    #### Input the results of design ID of Pareto solutions selected from TOPSIS
    num_classes = [ 37,  95, 131, 139, 147, 152, 195, 207, 236, 275, 276, 330, 336, 384, 386, 411, 425, 487, 500, 556, 577, 636, 693, 757, 761, 800, 831, 857, 858, 859, 882, 896, 954, 959]

    file = './5. Pareto optimal set/efficient_frontier.xlsx'

    df = pd.read_excel(file)

    ID = df['Design ID'].to_numpy()
    w_support = df['w_support']
    w_time = df['w_time']
    w_stress = df['w_stress']
    w_data = df[['w_support', 'w_time', 'w_stress']].to_numpy()
    # uni=ID.unique()
    # print(len(uni))
    labels = ID
    data = w_data

    print(data)
    print(labels)
    ##COLOR### Note: Now color is set as 34 colors corresponding to number of pareto solutions in this study,
    #                please change the number depending on the number of your pareto solutions sets
    colormap1 = plt.cm.tab20(np.linspace(0, 1, 20))
    colormap2 = plt.cm.Set3(np.linspace(0, 1, 14))

    colors = np.concatenate((colormap1, colormap2), axis=0)

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for class_label in num_classes:
        class_data = data[labels == class_label]
        print(class_data)
        ax.scatter(class_data[:, 0], class_data[:, 1], class_data[:, 2], label=f'{class_label}',
                   color=colors[num_classes.index(class_label)])

    # Add legend
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=10, title='Generated Design ID')
    # legend = ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1), title='Generated Design ID')

    # Set labels for each axis
    ax.set_xlabel('Weight of material consumption for support structure')
    ax.set_ylabel('Weight of fabrication time')
    ax.set_zlabel('Weight of static mechanical stress')
    # Set plot title
    ax.set_title('Efficient frontier')

    # Show the plot
    plt.show()
####--------------------------------------------------------------------------------------------------------------------

