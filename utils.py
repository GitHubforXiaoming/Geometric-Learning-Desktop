# -- coding: utf-8 --

import os
import numpy as np
import random
import time

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from transform import *
from visualization import *


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

print('test time begin at ' + time.asctime(time.localtime(time.time())) + '\n')
all_centers = []
all_points = []

dir_of_fractures = './fractures/'
file_names = os.listdir(dir_of_fractures)

for file_name in file_names:
    poly_data = read_stl(dir_of_fractures + file_name)
    n = poly_data.GetNumberOfPoints()
    X = np.zeros((n, 3), dtype=np.float64)
    # visualization(data)
    points = poly_data.GetPoints()
    for i in range(n):
        X[i] = points.GetPoint(i)

    # random sample
    all_points.append(random.sample(X, 100))
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_

    # serialize the center points
    centers = serialize_centers(centers)
    all_centers.append(centers)

    # compute the pca of center points
    pca = PCA(n_components=3)
    pca.fit(np.array(centers))


    # connect points in proper sequence 

    # fv = FlatVisualization()
    # fv.paint_line(centers)

    # visualize fracture with the contral points (center points)

    # tv = TridimensionalVisualization(poly_data)
    # center_points_data = tv.visualize_points(centers)
    # tv.visualize_models_man(poly_data, center_points_data)
    # datas = []
    # for center in centers:
    #    sphere = tv.draw_sphere(center, 2)
    #    datas.append(sphere)
    # datas.append(poly_data)
    # tv.visualize_models_auto(datas)

all_centers = np.array(all_centers)
all_points = np.array(all_points)
centers = np.array([np.mean(center, 0) for center in all_centers])

tv = TridimensionalVisualization()
fv = FlatVisualization()
icp = ICP()
transform = Transform()

points = all_points
for i in range(len(points)):
    min_bais = []
    for j in range(len(points)):
        if i != j and abs(i - j) != 1:
            # T, distances, _ = icp.icp(all_centers[i], all_centers[j], max_iterations=80)
            # print(file_names[i], ' - ', file_names[j], '\ndistance: ', distances)
            # print('sum: ', sum(distances), '\n')
            # min_bais.append(sum(distances))

            # visualize transformed data by ICP
            # matrix = transform.get_matrix(T)
            # points_data_fixed = tv.convert_points_to_data(all_centers[i])
            # points_data_float = tv.convert_points_to_data(all_centers[j])
            # points_data_float = transform.transform_data(matrix, points_data_float)
            # tv.visualize_models_man(points_data_fixed, points_data_float)

            # visualize transformed data by pre-registeration
            # fixed_a, float_b = all_centers[i][0], all_centers[i][-1]
            # float_a, float_b = all_centers[j][0], all_centers[j][-1]
            fixed_points, float_points = points[i], points[j]
            title = file_names[i] + ' compare to ' + file_names[j]
            print(title)
            float_points_, bias, main_axis_matrix, secondary_axis_matrix, translate_axis_matrix = \
                transform.collimate_axis(fixed_points, float_points)

            min_bais.append(bias)

            # fv.paint_two_points(fixed_points, float_points_, arrow=True)
            

        else:
            min_bais.append(np.inf)
    print('---', file_names[i], ' - ', file_names[min_bais.index(min(min_bais))], ': ', min(min_bais), '\n')

print('test time end at ' + time.asctime(time.localtime(time.time())) + '\n')