from stl import mesh
import stltovoxel
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import copy
import os
import sys
import math

def stl_voxelizer(dir_names, resolution):
    input = dir_names
    meshes = list()
    resolution = resolution
    mesh_obj = mesh.Mesh.from_file(input)
    mesh_obj.rotate([1,0,0], math.radians(90))
    mesh_obj.rotate([0,1,0], math.radians(90))
    mesh_obj.rotate([1,0,0], math.radians(180))

    org_mesh = np.hstack((mesh_obj.v2[:, np.newaxis], mesh_obj.v1[:, np.newaxis], mesh_obj.v0[:, np.newaxis]))
    meshes.append(org_mesh)
    vol, scale, shift = stltovoxel.convert_meshes(meshes,resolution-1)
    print(scale)
    vol = np.rot90(vol,k=3, axes=(1,2))
    print(np.shape(vol))
    if np.shape(vol)[1] % 2 == int(0):
        vol = np.pad(vol,((0,0),(int((resolution-int(np.shape(vol)[1]))/2),int((resolution-int(np.shape(vol)[1]))/2)),(0,resolution-int(np.shape(vol)[2]))),mode='constant')
    else:
        vol = np.pad(vol,((0,0),(int((resolution-int(np.shape(vol)[1]))/2),int((resolution-int(np.shape(vol)[1]))/2)+1),(0,resolution-int(np.shape(vol)[2]))),mode='constant')
    if np.shape(vol)[0] == int(255):
        if np.shape(vol)[1] % 2 == int(0):
            vol = np.pad(vol, (
            (0, 1), (int((resolution - int(np.shape(vol)[1])) / 2), int((resolution - int(np.shape(vol)[1])) / 2)),
            (0, resolution - int(np.shape(vol)[2]))), mode='constant')
        else:
            vol = np.pad(vol, (
            (0, 1), (int((resolution - int(np.shape(vol)[1])) / 2), int((resolution - int(np.shape(vol)[1])) / 2) + 1),
            (0, resolution - int(np.shape(vol)[2]))), mode='constant')
    return vol

if __name__ == '__main__':

    PATH = r'./stl_data'
    SAVE_PATH = r'./voxel_data'
    resolution = 128

    stl_names = os.listdir(PATH)
    print(stl_names)

    for idx in range(len(stl_names)):
        STL_PATH = os.path.join(PATH,stl_names[idx])
        STL_NAME = stl_names[idx].replace('.stl','')
        vox= stl_voxelizer(STL_PATH,resolution=resolution)
        np.save(SAVE_PATH+fr'\{STL_NAME}',vox)
        print(np.shape(vox))
        if np.shape(vox) != (resolution,resolution,resolution):
            print(stl_names[idx])   # If the voxelized data does not correspond with the resolution, the filename will be displayed.
        else:
            print('good')