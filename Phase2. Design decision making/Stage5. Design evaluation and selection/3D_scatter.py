import matplotlib.pyplot as plt
import os
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib
import seaborn as sns
from random import shuffle


matplotlib.use('Qt5Agg')
#Input the results of design ID of Pareto solutions selected from TOPSIS
# num_classes = [ 37,  95, 131, 139, 147, 152, 195, 207, 236, 275, 276, 330, 336, 384, 386, 411, 425, 487, 500, 556, 577, 636, 693, 757, 761, 800, 831, 857, 858, 859, 882, 896, 954, 959]

file = './efficient_frontier(ex).xlsx'

df = pd.read_excel(file)

ID = df['Design ID'].to_numpy()
w_support=df['w_support']
w_time=df['w_time']
w_stress=df['w_stress']
w_data = df[['w_support','w_time','w_stress']].to_numpy()
# uni=ID.unique()
# print(len(uni))
labels=ID
data=w_data

print(data)
print(labels)
##COLOR### Note: Now color is set as 34 colors corresponding to number of pareto solutions in this study,
#                please change the number depending on the number of your pareto solutions sets
colormap1 = plt.cm.tab20(np.linspace(0, 1, 20))
colormap2 =  plt.cm.Set3(np.linspace(0, 1, 14))




colors = np.concatenate((colormap1, colormap2), axis=0)

# colors = sns.color_palette("husl", n_colors=len(num_classes))
# colors = plt.cm.Set3(np.linspace(0, 1, len(num_classes)))
# colors = sns.hls_palette(n_colors=len(num_classes), h=0.5, l=0.6, s=0.6)
# colors = sns.color_palette("Paired", n_colors=len(num_classes))
# colors = sns.color_palette('Paired',n_colors=num_classes, desat=0.8, alpha=0.7)

####################

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for class_label in num_classes:
    class_data = data[labels == class_label]
    print(class_data)
    ax.scatter(class_data[:, 0], class_data[:, 1], class_data[:, 2], label=f'{class_label}', color=colors[num_classes.index(class_label)])

# Add legend
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=10,title='Generated Design ID')
# legend = ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1), title='Generated Design ID')


# Set labels for each axis
ax.set_xlabel('Weight of material consumption for support structure')
ax.set_ylabel('Weight of fabrication time')
ax.set_zlabel('Weight of static mechanical stress')
# Set plot title
ax.set_title('Efficient frontier')

# Show the plot
plt.show()

