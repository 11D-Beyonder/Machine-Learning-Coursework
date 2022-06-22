import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

import data_loader

X, y = data_loader.data_fruit()
colors = ['navy', 'turquoise', 'darkorange', 'lightcoral', 'olive', 'dodgerblue', 'crimson']
target_names = ['BERHI', 'DEGLET', 'DOKOL', 'IRAQI', 'ROTANA', 'SAFAVI', 'SOGAY']

X_r = PCA(n_components=3).fit_transform(X)
fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
ax.set_title('p')
for color, target_name in zip(colors, target_names):
    ax.scatter(X_r[y == target_name, 0], X_r[y == target_name, 1], X_r[y == target_name, 2], c=color, alpha=.6, lw=1,
               label=target_name)
ax.legend(loc='best', shadow=True, scatterpoints=1)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

fig = plt.figure()
X_r = PCA(n_components=2).fit_transform(X)
for color, target_name in zip(colors, target_names):
    plt.scatter(X_r[y == target_name, 0], X_r[y == target_name, 1], c=color, alpha=.6, lw=1,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.xticks([])
plt.yticks([])

plt.show()
