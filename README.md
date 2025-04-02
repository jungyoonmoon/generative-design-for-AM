# A Generative Design Framework for Automated Feasible Design Generation and Design Decision-making in the Early Design Process for Additive Manufacturing
This repository contains the codes for "A Generative Design Framework for Automated Feasible Design Generation and Design Decision-making in the Early Design Process for Additive Manufacturing" paper. 

# Decriptions for each folder
<h3> Phase1. Design generation </h3>
The design generation phase have three interrelated stages.
<br> </br> 

**Stage 1. Generative model-based design automation**
- Train the IM-GAN (composed by IM-AE and Latent-GAN) to automatically generate the 3D designs.

**Stage 2. Defective design filtering**
- Filter the defective design in generated design dataset.

**Stage 3. Selective data augmentation and model update**
- Randomly select the designs in design dataset from Stage 2, and augment the original design dataset.

<h3> Phase2. Design decision making </h3>
The design decision making phase composed of two stages.

**Stage 4. Performance prediction**
- Train the 3D-CNN models to predict the performance criteria of generated designs.

**Stage 5. Design evaluation and selection**
- Select the designs by predicted performance criteria using TOPSIS method.

 
