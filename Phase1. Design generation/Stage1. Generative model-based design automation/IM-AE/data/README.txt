# The Dataset for IM-AE Training

This directory is used to store the dataset required for training the **IM-AE** model.

## Dataset Requirements

- The dataset should have already undergone **point sampling** in the previous directory.
- Place your **custom dataset** in the `dataset/` directory within this folder.
- Ensure that both **training** and **validation** datasets are included. (If you did not split, doesn't matter, do not care of it)
- The dataset should be stored in **`.hdf5` format**, as it is required for training IM-AE.

## Usage

1. Ensure your dataset has been processed with **point sampling** before moving it here.
2. Copy your dataset files into the `dataset/` directory.
3. Verify that all required `.hdf5` files are present before starting training.

