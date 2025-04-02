# Training the Latent-GAN

This guide explains how to train the **Latent-GAN** using extracted latent vectors.

## Preparing the Latent Vectors

1. **Extract Latent Vectors**  
   - Follow the instructions in `latent vector extraction.txt` in the **IM-AE** directory.
   - Ensure the extracted latent vectors are placed in:
     ```
     ./data/dataset_z/
     ```

## Training the Latent-GAN

Run the following command to train the Latent-GAN:

```bash
python main.py --train --epoch 10000 --dataset dataset_z
```

##Generating Latent Vectors

Once training is complete, generate new latent vectors using:

```bash
python main.py --dataset dataset_z
```
