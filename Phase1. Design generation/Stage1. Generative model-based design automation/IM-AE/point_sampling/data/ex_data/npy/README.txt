# Voxel to Surface Point Data Transformation for IM-AE Training

This repository provides tools to transform voxel data into surface point data for training the **IM-AE** model using progressive learning.

## Dataset Requirements

- Your dataset should contain voxel data with a shape of **128 × 128 × 128**.
- Voxel data should be stored as **NumPy files** (`.npy`).
- File naming convention: `1.npy`, `2.npy`, ..., `N.npy`. (It is just convention, if you recognize others more better, you can)

## Usage

1. Place your voxel dataset in this directory.
2. rewrite the "voxel_input" variable as this folder
2. Run `point_sampling.py`, which will automatically detect and process all `.npy` files in here.

```bash
python point_sampling.py

**Note: You can also split your dataset training and test using train_test_split.py