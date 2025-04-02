# TOPSIS Process for Design Evaluation and Selection

This process evaluates and selects optimal generative designs using the **TOPSIS** method.

## Steps for Running the TOPSIS Process

### 1. Prepare the Input Data
- Copy the **predicted results** of generated design data from: './Stage4. Performance prediction/prediction results/'
and paste them into: './Stage5. Design evaluation and selection/prediction results/'.


### 2. Run the TOPSIS Algorithm
- Execute the following script to perform the **TOPSIS evaluation**:
```bash
python TOPSIS.py

### 3. Construct the Pareto Optimal Set
- Run the script to generate a 3D scatter plot representing the Pareto optimal set for generative design in Additive Manufacturing (AM):
```bash
python 3D_scatter.py
