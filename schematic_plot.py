import matplotlib.pyplot as plt
import numpy as np
import os
import vtk

from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree


def read_stl(file_name):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(file_name)
    reader.Update()

    poly_data = reader.GetOutput()
    return poly_data

def serialize_centers(centers):
    k = len(centers)
    kdtree = KDTree(centers, leaf_size=30, metric='euclidean')
    distances, mapping = kdtree.query(centers, k=k, return_distance=True)
    
    start = np.where(distances[:, k - 1] == max(distances[:, k - 1]))
    start = start[0][0]
    end = mapping[start][k - 1]

    serialized_indices = [start]
    while start != end:
        i = 1
        while mapping[start][i] in serialized_indices:
            i += 1
        start = mapping[start][i]
        serialized_indices.append(start)

    return np.array([centers[x] for x in serialized_indices])


def plot(ax, points, color, cluster=6):
    kmeans = KMeans(n_clusters=cluster)
    kmeans.fit(points)
    centers = kmeans.cluster_centers_

    pca = PCA(n_components=3)
    pca.fit(np.array(centers))

    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker='o', s=50, c=color)

    center = np.mean(centers, 0)

    x, y, z = np.meshgrid(np.array([center[0] for i in range(4)]), \
         np.array([center[1] for i in range(4)]), np.array([center[2] for i in range(4)]))
    u = np.array([pca.components_[0][0], pca.components_[1][0], -pca.components_[0][0], -pca.components_[1][0]])
    v = np.array([pca.components_[0][1], pca.components_[1][1], -pca.components_[0][1], -pca.components_[1][1]])
    w = np.array([pca.components_[0][2], pca.components_[1][2], -pca.components_[0][2], -pca.components_[1][2]])
    ax.quiver(x, y, z, u, v, w, length=20)

fixed_poly_data = read_stl('./fractures/3-2-a.stl')
float_poly_data = read_stl('./fractures/3-3-a.stl')

fixed_poly_points = fixed_poly_data.GetPoints()
float_poly_points = float_poly_data.GetPoints()

m, n = fixed_poly_data.GetNumberOfPoints(), float_poly_data.GetNumberOfPoints()

fixed_points = np.zeros((m, 3), dtype=np.float64)
float_points = np.zeros((n, 3), dtype=np.float64)

for i in range(m): fixed_points[i] = fixed_poly_points.GetPoint(i)
for i in range(n): float_points[i] = float_poly_points.GetPoint(i)

fig = plt.figure()
ax = Axes3D(fig)

plot(ax, fixed_points, 'crimson',cluster=16)
plot(ax, float_points, 'darkcyan', cluster=16)

plt.show()