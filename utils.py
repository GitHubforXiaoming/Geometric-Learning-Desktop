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


def random_sample(X, l=100):
    res = []
    indices = np.random.randint(len(X), size=l)
    for index in indices:
        res.append(X[index])
    return np.array(res)


def make_pair(fixed_points, float_points, indices):
    assert len(float_points) == len (indices) and len(fixed_points) == len(indices)
    X = []
    Y = []
    for i in range(len(indices)):
        X.append(fixed_points[i])
        Y.append(float_points[indices[i]])
    return np.array(X), np.array(Y)


print('test time begin at ' + time.asctime(time.localtime(time.time())) + '\n')
all_centers = []
all_points = []
original_points = []

dir_of_fractures = './fractures/'
prefix = '3'
file_names = os.listdir(dir_of_fractures)

for file_name in file_names:
    if file_name.startswith(prefix):
        poly_data = read_stl(dir_of_fractures + file_name)
        n = poly_data.GetNumberOfPoints()
        X = np.zeros((n, 3), dtype=np.float64)
        # visualization(data)
        points = poly_data.GetPoints()
        for i in range(n):
            X[i] = points.GetPoint(i)

        original_points.append(X)

        kmeans = KMeans(n_clusters=6)
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

# random sample
l = min([len(x) for x in original_points])
for points in original_points:
    all_points.append(random_sample(points, l))


all_centers = np.array(all_centers)
all_points = np.array(all_points)
original_points = np.array(original_points)
centers = np.array([np.mean(center, 0) for center in all_centers])


tv = TridimensionalVisualization()
icp = ICP()
transform = Transform()

points = all_centers
for i in range(len(points)):
    min_bais = []
    alternative_pair = {}
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
            # make dir
            path = './axis_transformation_pics/' + file_names[i][:-4] + '&' + file_names[j][:-4] + './'
            if not os.path.exists(path):
                print(path)
                os.makedirs(path)

            float_points_, bias, identification = transform.collimate_axis(fixed_points, float_points, path)
            min_bais.append(bias)
            transform.save_fig('./fracture_comparsion_pics/', file_names[i] + ' compares to ' + file_names[j] + '.png', fixed_points, float_points_)

        else:
            alternative_pair[file_names[j]] = np.inf
            min_bais.append(np.inf)
    alternative_pair = sorted(alternative_pair.items(), key = lambda x:x[1])
    for pair in alternative_pair:
        if pair[1] is not np.inf:
            print(file_names[i], ' compare to ', pair[0], ' with bias ', pair[1])
    print('---', file_names[i], ' - ', file_names[min_bais.index(min(min_bais))], ': ', min(min_bais), '\n')

print('test time end at ' + time.asctime(time.localtime(time.time())) + '\n')