# Performance Prediction Using CNN

This folder contains scripts for **performance prediction** using a **3D Convolutional Neural Network (CNN)**.

## Steps to Perform Prediction

### 1. Prepare the **X Dataset**
1. Place design data in the following directories:
   - **Voxelized data (`.npy`)** → `./data/X/npy/`
   - **3D mesh data (`.stl`)** → `./data/X/stl/`
   
2. Compute the **volume** and **surface area**:
   - Run `vol_surf_calc.py` to calculate design properties.
   - The calculated values will be stored in:
     ```
     ./data/X/vol_surf/vol_surf.xlsx
     ```

### 2. Prepare the **y Dataset**
- Create the **performance metrics dataset** in: './data/y/stress/', './data/y/support/' ,'./data/y/time/'
- Follow the format in the example files (`y(ex).xlsx`) provided in each directory.

### 3. Run the CNN Model
- Open and execute `3D-CNN_torch.py` in your **IDE**.

### 4. Predict the performance of generated designs by trained 3D-CNN
1. Place generated design data in the following directories:
   - **Voxelized data (`.npy`)** → `./data/generated design set/npy/`
   - **3D mesh data (`.stl`)** → `./data/generated design set/stl/`

2. Compute the **volume** and **surface area** of generated designs:
   - Run `vol_surf_calc.py`.
   - The calculated values will be stored in:
     ```
     ./data/generated design set/vol_surf/vol_surf.xlsx
     ```

3. Predict the performnaces using 'performance_predictor.py' code
   - Set the "?WHICH MODEL?" section in 'performance_predictor.py' 
   - Run the code in your IDE
   - Check the predicted performance results in './prediction results/'
   
