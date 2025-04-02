# Extracting Latent Vectors from Dataset

This guide explains how to extract **latent vectors** from a dataset using the trained **IM-AE** encoder.

## Steps to Extract Latent Vectors

1. **Ensure Trained Encoder Weights are Available**  
   - The trained encoder weights should be located in:
     ```
     IM-AE/checkpoint/
     ```

2. **Prepare the Dataset**  
   - Place the dataset in:
     ```
     IM-AE/dataset/
     ```

3. **Run the Latent Vector Extraction Command**  
   Execute the following command in your console:
   
   ```bash
   python main.py --ae --getz --dataset dataset --data_dir ./data/dataset
