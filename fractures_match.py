import const_values
import getopt
import numpy as np
import sys
import time
from utils import Utils
from visualization import *


def usage():
    print('usage of fractures match')
    print('-p, --prefix(demand): the title prefix of fractures.')
    print('-m, --match: load the match pairs from file. e.g. -m ./match-pair/plate-1.txt')
    print('-v, --visualizaiton: visualize the model in vtk.')
    print('-d, --decrease: the flag of decrease the number of cluster.')
    print('-k, --cluster(default 6): the number of clusters.')
    print('-c, --comparsion: begin the comparsion among the fractures.')
    print('-t, --test: process test code without core opration.')
    print('-s, --save: save the comparsion result or not.')
    print('-h, --help: print help message.')

def parse_args(argv):
    args = argv[1:]
    try:
        opts, args = getopt.getopt(args, 'p:m:vdk:cth', ['prefix=', 'cluster=', 'match='])
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)
    prefix = ''
    pairs = {}
    n_cluster = const_values.FLAGS.default_n_cluster
    flag = False
    for o, a in opts:
        if o in ('-h', '--help'):
            usage()
            sys.exit(1)
        elif o in ('-v', '--visualization'):
            utils = Utils(prefix, k=n_cluster)
            tv = TridimensionalVisualization()
            # visualize the control points on the fracture
            all_centers = utils.generate_datas(is_decrease=flag)[2]
            for data, centers in zip(utils.datas, all_centers):
                datas = []
                datas.append(data)
                points_data = tv.convert_points_to_data(centers)
                datas.append(points_data)
                for center in centers:
                    datas.append(tv.draw_sphere(center, 1))
                tv.visualize_models_auto(datas)
        elif o in ('-p', '--prefix'):
            prefix = a
        elif o in ('-m', '--match'):
            # load fragments
            utils = Utils(prefix, k=n_cluster)
            utils.load_all_fragment()

            # read match pairs
            with open(a) as file:
                for line in file:
                    l = line.strip('\n').split('&')
                    pairs[l[0]] = l[1]
            # print(pairs)

        elif o in ('-t', '--test'):
            utils = Utils(prefix, k=n_cluster)
            utils.load_all_fragment()
            for key in pairs:
                utils.transform_pair(key, pairs[key])

            # length, names = utils.generate_datas(is_decrease=flag)[-2::]
            # print(length, names)
            # print(np.mean(length))
        elif o in ('-d', '--decrease'):
            flag = True
        elif o in ('-k', '--cluster'):
            n_cluster = int(a)
        elif o in ('-c', '--comparsion'):
            utils = Utils(prefix, k=n_cluster)
            utils.comparsion(flag)
        else:
            print('unhandled option')
            sys.exit(3)


def main(argv):
    parse_args(argv)


if __name__ == "__main__":
    print('test time begin at ' + time.asctime(time.localtime(time.time())) + '\n')
    main(sys.argv)
    print('test time end at ' + time.asctime(time.localtime(time.time())) + '\n')