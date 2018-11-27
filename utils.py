# -- coding: utf-8 --

import os
import numpy as np
import random

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from transform import *
from visualization import *

class Utils:

    def __init__(self, prefix, k=6):
        self.prefix = prefix
        self.k = k

    def read_stl(self, file_name):
        reader = vtk.vtkSTLReader()
        reader.SetFileName(file_name)
        reader.Update()

        poly_data = reader.GetOutput()
        return poly_data


    def serialize_centers(self, centers):
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


    def length_of_fracture(self, centers):
        return np.linalg.norm(centers[0] - centers[-1])


    def random_sample(self, X, l=100):
        res = []
        indices = np.random.randint(len(X), size=l)
        for index in indices:
            res.append(X[index])
        return np.array(res)


    def make_pair(self, fixed_points, float_points, indices):
        assert len(float_points) == len (indices) and len(fixed_points) == len(indices)
        X = []
        Y = []
        for i in range(len(indices)):
            X.append(fixed_points[i])
            Y.append(float_points[indices[i]])
        return np.array(X), np.array(Y)


    def generate_datas(self):
        all_points = []
        all_random_points = []
        all_centers = []
        all_length = []

        file_names = os.listdir(const_values.FLAGS.dir_of_fractures)

        for file_name in file_names:
            if file_name.startswith(self.prefix):
                poly_data = self.read_stl(const_values.FLAGS.dir_of_fractures + file_name)
                n = poly_data.GetNumberOfPoints()
                X = np.zeros((n, 3), dtype=np.float64)
                # visualization(data)
                points = poly_data.GetPoints()
                for i in range(n):
                    X[i] = points.GetPoint(i)

                all_points.append(X)

                # random sample
                l = min([len(x) for x in all_points])
                for points in all_points:
                    all_random_points.append(self.random_sample(points, l))

        avg_num = np.mean(np.array([len(x) for x in all_points]))
        for X in all_points:
            k = self.k
            if len(X) < avg_num:
                k = self.k // 2 + 1
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(X)
            centers = kmeans.cluster_centers_

            # serialize the center points
            centers = self.serialize_centers(centers)

            # compute the length of each fracture
            all_length.append(self.length_of_fracture(centers))
            all_centers.append(centers)

        return np.array(all_points), np.array(all_random_points), np.array(all_centers), np.array(all_length),\
            [name for name in file_names if name.startswith(self.prefix)]


    def comparsion(self):
        all_points, all_random_points, all_centers, all_length, file_names = self.generate_datas()
        centers = np.array([np.mean(center, 0) for center in all_centers])

        icp = ICP()
        transform = Transform()

        points = all_centers
        for i in range(len(points)):
            min_bais = []
            alternative_pair = {}
            for j in range(len(points)):
                if i != j and abs(i - j) != 1:
                    fixed_points, float_points = points[i], points[j]
                    # make dir
                    path = const_values.FLAGS.dir_of_axis_transformation_pics + file_names[i][:-4] + '&' + file_names[j][:-4] + '/'
                    if not os.path.exists(path):
                        print(path)
                        os.makedirs(path)

                    float_points_, bias, identification = transform.collimate_axis_general(fixed_points, float_points, path)
                    min_bais.append(bias)
                    transform.save_fig(const_values.FLAGS.dir_of_fracture_comparsion_pics, file_names[i] + ' compares to ' + file_names[j] + '.png', fixed_points, float_points_)

                else:
                    alternative_pair[file_names[j]] = np.inf
                    min_bais.append(np.inf)
            alternative_pair = sorted(alternative_pair.items(), key = lambda x:x[1])
            for pair in alternative_pair:
                if pair[1] is not np.inf:
                    print(file_names[i], ' compare to ', pair[0], ' with bias ', pair[1])
            print('---', file_names[i], ' - ', file_names[min_bais.index(min(min_bais))], ': ', min(min_bais), '\n')

