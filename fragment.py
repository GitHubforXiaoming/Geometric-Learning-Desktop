from collections import namedtuple
from transform import *


class Fragment(object):

    def __init__(self, fragment, fractures, prefix):
        self.fragment = fragment
        self.fractures = fractures
        self.trans = Transform()
        self.prefix = prefix


    def transform_fragment(self, matrix):
        self.fragment = self.trans.transform_data(matrix, self.fragment)
        for fracture in self.fractures:
            fracture = self.trans.transform_data(matrix, fracture)


    def save_fragment(self, path):
        writer = vtk.vtkWriter()
        writer.SetInputData(self.fragment) 
        writer.SetFileName(path + 'fragment-' + self.prefix + '.stl')
        writer.Write()   