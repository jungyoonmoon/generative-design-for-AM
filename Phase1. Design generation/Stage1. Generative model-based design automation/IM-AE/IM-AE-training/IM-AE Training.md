# Progressive learning for IM-AE

This guide explains how to perform **progressive learning** for the **IM-AE** model.

## Training Process

After each training phase, execute the following commands in your console under: "Phase1. Design generation/Stage1. Generative model-based design automation/IM-AE/" 

###
**Step 1: 16^3 Resolution Training**
Train the model at **16×16×16** resolution:
```bash
python main.py --ae --train --epoch 200 --sample_dir samples/dataset_img0_16 --sample_vox_size 16
```
**Step 2: 32^3 Resolution Training**
Train the model at **32×32×32** resolution:
```bash
python main.py --ae --train --epoch 200 --sample_dir samples/dataset_img0_32 --sample_vox_size 32
```
**Step 3: 64^3 Resolution Training**
Train the model at **64×64×64** resolution:
```bash
python main.py --ae --train --epoch 1500 --sample_dir samples/dataset_img0_64 --sample_vox_size 64
```
