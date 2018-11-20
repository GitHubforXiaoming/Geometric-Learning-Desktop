import const
import vtk
import math
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class TridimensionalVisualization:
    def __init__(self):
        pass

    def visualize_model(self, data):
        """
        visualize the 3d data

        """
        mapper = vtk.vtkPolyDataMapper()  
        mapper.SetInputData(data)  
        
        actor = vtk.vtkActor()  
        actor.SetMapper(mapper) 
        actor.GetProperty().SetPointSize(4) 
        
        
        ren= vtk.vtkRenderer()  
        ren.AddActor( actor )  
        # ren.SetBackground( 0.1, 0.2, 0.4 )  
        
        renWin = vtk.vtkRenderWindow()  
        renWin.AddRenderer( ren )  
        renWin.SetSize( 300, 300 )  
        renWin.Render()  
        
        iren=vtk.vtkRenderWindowInteractor()  
        iren.SetRenderWindow(renWin)  
        
        iren.Initialize()  
        iren.Start()
    

    def visualize_models_man(self, *datas):
        ren= vtk.vtkRenderer()  
        i = 0
        l = len(const.COLOR)
        for data in datas:
            mapper = vtk.vtkPolyDataMapper()  
            mapper.SetInputData(data)  
            
            actor = vtk.vtkActor()  
            actor.SetMapper(mapper) 
            actor.GetProperty().SetColor(const.COLOR[i % l][0] / 255.0, const.COLOR[i % l][1] / 255.0, const.COLOR[i % l][2] / 255.0,)
            actor.GetProperty().SetPointSize(10) 
            
            ren.AddActor( actor )  
            ren.SetBackground( 0 / 255.0, 166 / 255.0, 222 / 255.0 ) 
            i += 1
        renWin = vtk.vtkRenderWindow()  
        renWin.AddRenderer( ren )  
        # renWin.SetSize( 300, 300 )  
        renWin.Render()  
        
        iren=vtk.vtkRenderWindowInteractor()  
        iren.SetRenderWindow(renWin)  
        
        iren.Initialize()  
        iren.Start() 

    def visualize_models_auto(self, datas):
        ren= vtk.vtkRenderer()  
        for data in datas:
            mapper = vtk.vtkPolyDataMapper()  
            mapper.SetInputData(data)  
            
            actor = vtk.vtkActor()  
            actor.SetMapper(mapper) 
            #actor.GetProperty().SetPointSize(10) 
            
            ren.AddActor( actor )  
            ren.SetBackground( 0 / 255.0, 166 / 255.0, 222 / 255.0 ) 
        renWin = vtk.vtkRenderWindow()  
        renWin.AddRenderer( ren )  
        renWin.SetSize( 300, 300 )  
        renWin.Render()  
        
        iren=vtk.vtkRenderWindowInteractor()  
        iren.SetRenderWindow(renWin)  
        
        iren.Initialize()  
        iren.Start()

    def convert_points_to_data(self, centers):
        points = vtk.vtkPoints()
        vertices = vtk.vtkCellArray()
        pointData = vtk.vtkPolyData()
        id = 1
        for point in centers:
            pid = [0]
            pid[0] = points.InsertNextPoint(point)
            vertices.InsertNextCell(id, pid)
            # id += 1
        pointData.SetPoints(points)
        pointData.SetVerts(vertices)
        return pointData


    def draw_sphere(self, center, radius):
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(center)
        sphere.SetRadius(radius)
        sphere.Update()
        return sphere.GetOutput()

class FlatVisualization:

    def quiver3d(self, X, ax):
        pca = PCA(n_components=3)
        center = np.mean(X, 0)
        pca.fit(X)
        x, y, z = np.meshgrid(np.array([center[0] for i in range(4)]),
         np.array([center[1] for i in range(4)]), np.array([center[2] for i in range(4)]))

        u = np.array([pca.components_[0][0], pca.components_[1][0], -pca.components_[0][0], -pca.components_[1][0]])
        v = np.array([pca.components_[0][1], pca.components_[1][1], -pca.components_[0][0], -pca.components_[1][0]])
        w = np.array([pca.components_[0][2], pca.components_[1][2], -pca.components_[0][0], -pca.components_[1][0]])
        c = np.random.randn(4)
        ax.quiver(x, y, z, u, v, w, length=30, pivot='tip')

    def paint_two_points(self, X, Y, title='', arrow=False):
        fig = plt.figure() 
        ax = Axes3D(fig)    
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker='o', s=50, c='crimson')
        if arrow:
            self.quiver3d(X, ax)
        ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], marker='o', s=50, c='darkcyan')
        if arrow:
            self.quiver3d(Y, ax)
        ax.set_xlabel('x', color='r')
        ax.set_ylabel('y', color='g')
        ax.set_zlabel('z', color='b') 
        ax.set_title(title)
        plt.show()
    
    def paint_points(self, X):
        fig = plt.figure() 
        ax = Axes3D(fig)    
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker='o', s=50, c='blue')
        ax.set_xlabel('x', color='r')
        ax.set_ylabel('y', color='g')
        ax.set_zlabel('z', color='b') 
        plt.show()


    def paint_line(self, X):
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='crimson')
        ax.plot(X[:, 0], X[:, 1], X[:, 2], c='forestgreen')
        plt.show()  