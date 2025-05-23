import matplotlib.pyplot as plt
import os
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')


file = './efficient_frontier.xlsx'


df = pd.read_excel(file)

ID = df['Design ID']
w_support=df['w_support']
w_time=df['w_time']
w_stress=df['w_stress']
uni=ID.unique()
print(uni)
# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmap = ListedColormap(np.random.rand(34, 3))
# cmap = plt.get_cmap('viridis', 40)

sc = ax.scatter(w_support, w_time, w_stress, c=ID, cmap=cmap, marker='o')



# Customize the plot
ax.set_xlabel('Weight of Support usage')
ax.set_ylabel('Weight of fabrication time')
ax.set_zlabel('Weight of Static stress')
ax.set_title('Efficient frontier')
fig.colorbar(sc, ax=ax, label='ID')

# Show the plot
plt.show()
