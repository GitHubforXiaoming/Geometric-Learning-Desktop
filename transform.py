import math
import vtk

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from visualization import FlatVisualization

class Transform:

    def get_matrix(self, T):
        '''
        convert the transform matrix to the form of vtkMatrix4x4

        Args:
            T: the transform matrix
        Return:
            matrix: the form of vtkMatrix4x4
        '''
        matrix = vtk.vtkMatrix4x4()

        for i in range(4):
            for j in range(4):
                matrix.SetElement(i, j, T[i][j])

        return matrix
    
    def transform_points(self, matrix, points):
        '''
        transform the two point sets to overlap approximately

        Args:
            matrix: transformation matrix
            points_fiexed: the fixed points data
            point_float: the points data to be transformed
        Return:
            the points after transformed 
        '''
        res = []
        transform = vtk.vtkTransform()
        transform.SetMatrix(matrix)
        for point in points:
            point_float = [0, 0, 0]
            transform.TransformPoint(point, point_float)
            res.append(point_float)
        return np.array(res)

    def transform_point(self, matrix, point):
        transform = vtk.vtkTransform()
        transform.SetMatrix(matrix)
        transformed_point = [0, 0, 0]
        transform.TransformPoint(point, transformed_point)
        return transformed_point

    def angle_of_normal(self, a, b):
        return math.acos(a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def transform_data(self, matrix, data):
        transform = vtk.vtkTransform()
        transform.SetMatrix(matrix)
        filter = vtk.vtkTransformPolyDataFilter()
        filter.SetInputData(data)
        filter.SetTransform(transform)
        filter.Update()

        return filter.GetOutput()

    def rotate_by_any_axis(self, axis, theta):
        '''
        rotate the float normal to be parallel to the fixed normal

        Args:
            axis: the axis to te rotated
            theta: the angle of rotation

        Return:
            matrix: transformation matrix
        '''

        matrix = vtk.vtkMatrix4x4()

        a = axis[0]
        b = axis[1]
        c = axis[2]

        
        matrix.SetElement(0, 0, a ** 2+ (1 - a ** 2) * math.cos(theta))
        matrix.SetElement(0, 1, a * b * (1 - math.cos(theta)) + c * math.sin(theta))
        matrix.SetElement(0, 2, a * c * (1 - math.cos(theta)) - b * math.sin(theta))
        matrix.SetElement(0, 3, 0)
        
        matrix.SetElement(1, 0, a * b * (1 - math.cos(theta)) - c * math.sin(theta))
        matrix.SetElement(1, 1, b ** 2 + (1 - b ** 2) * math.cos(theta))
        matrix.SetElement(1, 2, b * c * (1 - math.cos(theta)) + a * math.sin(theta))
        matrix.SetElement(1, 3, 0)
        
        matrix.SetElement(2, 0, a * c * (1 - math.cos(theta)) + b * math.sin(theta))
        matrix.SetElement(2, 1, b * c * (1 - math.cos(theta)) - a * math.sin(theta))
        matrix.SetElement(2, 2, c ** 2 + (1 - c ** 2) * math.cos(theta))
        matrix.SetElement(2, 3, 0)
        
        matrix.SetElement(3, 0, 0)
        matrix.SetElement(3, 1, 0)
        matrix.SetElement(3, 2, 0)
        matrix.SetElement(3, 3, 1)

        return matrix

    def translate(self, start, end):
        '''
        translate the start point to overlap with end point

        Args:
            start: start point (float)
            end: end point (fixed)

        Return:
            matrix: transformation matrix
        '''

        matrix = vtk.vtkMatrix4x4()

        for i in range(4):
            for j in range(4):
                if i is j:
                    matrix.SetElement(i, j, 1)
                elif j is 3:
                    matrix.SetElement(i, j, start[i] - end[i])
                else:
                    matrix.SetElement(i, j, 0)
        
        return matrix

    def compute_axis(self, points):
        pca = PCA(n_components=3)
        pca.fit(points)

        return pca.components_[0], pca.components_[1]

    def collimate_axis(self, fixed_points, float_points):
        '''
        collimate the fixed main axis and fixed secondary axis with the float ones
        there are three times transformation totaly
        Args:
            fixed_points: the baseline points ont the fracture
            float_points: the transforming points ont the fracture

        Return:
            points: the transformed points of float fracture
            matrices: kinds of transformation matrices
        '''
        # record the transformation matrix of each step
        index = 0
        main_axis_matrices = []
        secondary_axis_matrices = []
        translate_axis_matrices = []

        points = float_points
        bias = np.inf
        icp = ICP()
        # compute the main axis of fixed points
        fixed_components = self.compute_axis(fixed_points)
        float_components = self.compute_axis(float_points)
        fixed_main_axis = np.array([fixed_components[0], -fixed_components[0]])
        float_main_axis = np.array([float_components[0], -float_components[0]])
        fixed_secondary_axis = np.array([fixed_components[1], -fixed_components[1]])

        tip = ['main', 'reversed main', 'secondary', 'reversed secondary']

        for i in range(len(fixed_main_axis)):
            for j in range(len(float_main_axis)):
                # firstly, align the main axis of two points
                theta = self.angle_of_normal(fixed_main_axis[i], float_main_axis[j])
                axis = np.cross(fixed_main_axis[i], float_main_axis[j])
                rotate_matrix = self.rotate_by_any_axis(axis, theta)
                
                # apply the first transformation
                points_main_axis_transformed = self.transform_points(rotate_matrix, float_points)

                # compute the axis after main axis transformation
                float_components = self.compute_axis(points_main_axis_transformed)
                float_secondary_axis = np.array([float_components[1], -float_components[1]])
                # 
                for k in range(len(fixed_secondary_axis)):
                    for l in range(len(float_secondary_axis)):

                        # push back the previous rotation matrix
                        main_axis_matrices.append(rotate_matrix)

                        # secondly, align the secondary axis of two points
                        theta = self.angle_of_normal(fixed_secondary_axis[i], float_secondary_axis[j])
                        axis = np.cross(fixed_secondary_axis[i], float_secondary_axis[j])
                        rotate_matrix = self.rotate_by_any_axis(axis, theta)
                        secondary_axis_matrices.append(rotate_matrix)
                        
                        # apply the first transformation
                        points_secondary_axis_transformed = self.transform_points(rotate_matrix, points_main_axis_transformed)

                        # finally, translate the center points of two points
                        fixed_center = np.mean(fixed_points, 0)
                        float_center = np.mean(points_secondary_axis_transformed, 0)
                        translate_matrix = self.translate(fixed_center, float_center)
                        translate_axis_matrices.append(translate_axis_matrices)

                        points_translated = self.transform_points(translate_matrix, points_secondary_axis_transformed)
                        
                        if np.mean(icp.nearest_neighbor(fixed_points, points_translated)) < bias:
                            bias = np.mean(icp.nearest_neighbor(fixed_points, points_translated))
                            points = points_translated
                            index = int(str(i) + str(j) + str(k) + str(l), 2)

        return points, bias, main_axis_matrices[index], secondary_axis_matrices[index], translate_axis_matrices[index]

    def turn_over_by_axis(self, axis):
        return self.rotate_by_any_axis(axis, np.pi / 2)




class ICP:
    def best_fit_transform(self, A, B):
        '''
        Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
        Input:
        A: Nxm numpy array of corresponding points
        B: Nxm numpy array of corresponding points
        Returns:
        T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
        R: mxm rotation matrix
        t: mx1 translation vector
        '''

        assert A.shape == B.shape

        # get number of dimensions
        m = A.shape[1]

        # translate points to their centroids
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B

        # rotation matrix
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
            Vt[m-1,:] *= -1
            R = np.dot(Vt.T, U.T)

        # translation
        t = centroid_B.T - np.dot(R,centroid_A.T)

        # homogeneous transformation
        T = np.identity(m + 1)
        T[:m, :m] = R
        T[:m, m] = t

        return T, R, t


    def nearest_neighbor(self, src, dst):
        '''
        Find the nearest (Euclidean) neighbor in dst for each point in src
        Input:
            src: Nxm array of points
            dst: Nxm array of points
        Output:
            distances: Euclidean distances of the nearest neighbor
            indices: dst indices of the nearest neighbor
        '''

        assert src.shape == dst.shape

        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(dst)
        distances, indices = neigh.kneighbors(src, return_distance=True)
        return distances.ravel(), indices.ravel()


    def icp(self, A, B, init_pose=None, max_iterations=20, tolerance=0.001):
        '''
        The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
        Input:
            A: Nxm numpy array of source mD points
            B: Nxm numpy array of destination mD point
            init_pose: (m+1)x(m+1) homogeneous transformation
            max_iterations: exit algorithm after max_iterations
            tolerance: convergence criteria
        Output:
            T: final homogeneous transformation that maps A on to B
            distances: Euclidean distances (errors) of the nearest neighbor
            i: number of iterations to converge
        '''

        assert A.shape == B.shape

        # get number of dimensions
        m = A.shape[1]

        # make points homogeneous, copy them to maintain the originals
        src = np.ones((m + 1,A.shape[0]))
        dst = np.ones((m + 1,B.shape[0]))
        src[:m,:] = np.copy(A.T)
        dst[:m,:] = np.copy(B.T)

        # apply the initial pose estimation
        if init_pose is not None:
            src = np.dot(init_pose, src)

        prev_error = 0

        for i in range(max_iterations):
            # find the nearest neighbors between the current source and destination points
            distances, indices = self.nearest_neighbor(src[:m,:].T, dst[:m,:].T)

            # compute the transformation between the current source and nearest destination points
            T, _, _ = self.best_fit_transform(src[:m,:].T, dst[:m,indices].T)

            # update the current source
            src = np.dot(T, src)

            # check error
            mean_error = np.mean(distances)
            if np.abs(prev_error - mean_error) < tolerance:
                break
            prev_error = mean_error

        # calculate final transformation
        T, _ , _ = self.best_fit_transform(A, src[:m,:].T)

        return T, distances, i