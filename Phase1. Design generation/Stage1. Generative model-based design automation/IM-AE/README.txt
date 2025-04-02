# IM-AE Training

This file is used to scripts required for training the **IM-AE** model.

## Training Workflow

Follow these steps to prepare and train the IM-AE model:

1. **Convert STL to Voxel Data**  
   - Use `voxelizer.py` located in the `voxelization/` directory to transform your custom **STL files** into **voxel data**.

2. **Perform Point Sampling**  
   - Run `point_sampling.py` in the `point_sampling/` directory to extract **surface point data** for training.

3. **Train the IM-AE Model**  
   - Use the commands provided in `IM-AE training.txt`.  
   - Execute the commands in your console to start training.

4. **Check Reconstruction Results**  
   - After training, view the reconstructed outputs in the `sample/` directory.

5. **Reconstruct the generated latent vector to 3D mesh"
   - Use the commands provided in "IM-AE reconstruction.txt"