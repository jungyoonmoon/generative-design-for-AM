import cc3d
import numpy as np
import os
import shutil

# matplotlib.use('Qt5Agg')
npy_path = r'./voxel_npy'
normal_PATH =r'./normal'
broken_PATH = r'./fractured'



iter_npy_names = os.listdir(npy_path)
print(iter_npy_names)
valid_num=int()
unvalid_num=int()
unvalid_list=list()

connectivity = 26

for idx in range(len(iter_npy_names)):
    NPY_PATH = os.path.join(npy_path, iter_npy_names[idx])
    voxel = np.load(NPY_PATH)
    label_out, num_comp = cc3d.connected_components(voxel, return_N=True, connectivity=connectivity)  # free
    # print(label_out)
    if num_comp == 1:
        valid_num +=1
        shutil.copy(NPY_PATH,os.path.join(normal_PATH, iter_npy_names[idx]))
    else:
        unvalid_num+=1
        unvalid_list.append(iter_npy_names[idx][0:-4])
        shutil.copy(NPY_PATH, os.path.join(broken_PATH, iter_npy_names[idx]))
        print(iter_npy_names[idx][0:-4])


print("normal numbers: ",valid_num)
print("abnormal numbers: ",unvalid_num)
print("abnormal list: ", unvalid_list)

