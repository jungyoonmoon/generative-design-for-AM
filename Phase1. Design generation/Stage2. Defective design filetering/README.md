# 3D Connectec-component labeling (3D CCL) for Defective Design Filtering

This process filters defective designs using **connected-component Labeling (CCL)** on voxelized data.

Version requirements:

cc3d (connected-components-3d) >= 3.10.5 ver.


## Steps to Perform Defective Design Filtering

1. **Prepare the Voxelized Design Data**  
   - Place your voxelized data in:
     ```
     ./voxel_npy/
     ```

2. **Run the CCL Tool**  
   - Open and execute `CCL_tool.py` in your **IDE**.

3. **Review the Results**  
   - Fractured (defective) parts will be stored in:
     ```
     ./fractured/
     ```
   - Normal (valid) parts will be stored in:
     ```
     ./normal/
     ```
