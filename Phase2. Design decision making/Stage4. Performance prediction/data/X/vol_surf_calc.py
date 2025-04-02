import numpy as np
from stl import mesh
import os
import pandas as pd

# STL 파일 로드
PATH = r'./stl'
SAVE_PATH = r'./vol_surf'
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

df = pd.DataFrame({'ID': dir_list, 'Surface': surf_list, 'Volume': vol_list})
df.to_excel(os.path.join(SAVE_PATH,'vol_surf.xlsx'))

print('Surface area:', total_areas)