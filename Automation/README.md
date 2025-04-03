## Usage Instructions for **Automation**

1. **Place the generated design dataset from Stage 1**  
   - For voxel data: `./1. Generated design set/npy`  
   - For STL data: `./1. Generated design set/stl`

2. **Calculate the volume and surface area**  
   - Run the script:  
     ```bash
     python vol_surf_calc.py
     ```

3. **Add trained weights for the 3D-CNN models**  
   - Stress prediction weights: `./3. Weights/stress`  
   - Support prediction weights: `./3. Weights/support`  
   - Time prediction weights: `./3. Weights/time`

4. **Run the final script to obtain the feasible Pareto optimal set**  
   - Execute:  
     ```bash
     python Feasible Pareto optimal set.py
     ```
