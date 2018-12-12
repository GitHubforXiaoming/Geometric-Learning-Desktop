import const_values
import getopt
import numpy as np
import os
import sys
import time
import vtk

from disc_descriptor import DiscDescriptor


def read_stl(file_name):
        reader = vtk.vtkSTLReader()
        reader.SetFileName(file_name)
        reader.Update()

        poly_data = reader.GetOutput()
        return poly_data

def usage():
    print('usage of fractures match')
    print('-d, --dir: specify the directory of the re-assembly plates.')
    print('-r, --random: draw descriptor randomly, require the random number. e.g. -r 10')
    print('-c, --config: config the options of descriptor, [num of points on disc(32)] [num of circles(32)] [init radius(0.1)] [radius delta(0.1)] [distance from disc(1)]')
    print('-v, --visualization: options([points, lines, circles, datas])')
    print('-h, --help: print help message.')


def valid_input(message, value):
    val = input(message)
    if val == '\n':
        pass
        
    return val


def parse_args(argv):
    args = argv[1:]
    try:
        opts, args = getopt.getopt(args, 'd:r:', ['dir=', 'random='])
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)
    
    random_num = 0
    distance_from_mesh = 1
    num_of_points_on_disc = 32
    num_of_circle = 32
    init_radius = 0.1
    radius_delta = 0.1
    file_names = []
    for o, a in opts:
        if o in ('-h', '--help'):
            usage()
            sys.exit(1)
        elif o in ('-d', '--dir'):
            for file_name in os.listdir(a):
                if file_name.endswith('.stl'):
                    file_names.append(a + file_name)
                    print(a + file_name)
                    
        elif o in ('-r', '--random'):
            random_num = int(a)
        elif o in ('-c', '--config'):
            distance_from_mesh = valid_input('distance from disc(1): ')
            num_of_points_on_disc = valid_input('num of points on disc(32): ')
            num_of_circle = valid_input('num of circles(32): ')
            init_radius = valid_input('init radius(0.1): ')
            radius_delta = valid_input('radius delta(0.1): ')

        else:
            print('unhandled option')
            sys.exit(3)
    
    for file_name in file_names:
        poly_data = read_stl(file_name)
        descriptor = DiscDescriptor(poly_data)
        descriptor.mesh_descriptors(random_num=random_num)

        # config desriptor
        descriptor.distance_from_mesh = distance_from_mesh
        descriptor.num_of_points_on_disc = num_of_points_on_disc
        descriptor.num_of_circle = num_of_circle
        descriptor.init_radius = init_radius
        descriptor.radius_delta = radius_delta

        # visualization
        visualized_datas = []
        circles = descriptor.draw_circles()
        lines_datas, points_datas = descriptor.draw_lines()
        visualized_datas.append(poly_data)
        visualized_datas.extend(circles)
        visualized_datas.extend(points_datas)
        visualized_datas.extend(lines_datas)
        descriptor.visualize_models(visualized_datas)


def main(argv):
    parse_args(argv)


if __name__ == "__main__":
    print('test time begin at ' + time.asctime(time.localtime(time.time())) + '\n')
    main(sys.argv)
    print('test time end at ' + time.asctime(time.localtime(time.time())) + '\n')