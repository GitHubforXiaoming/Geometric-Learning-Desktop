import getopt
import numpy as np
import sys
import time
from utils import Utils


def usage():
    print('usage of fractures match')
    print('-p, --prefix(demand): the title prefix of fractures')
    print('-k, --cluster(default 6): the number of clusters')
    print('-c, --comparsion: begin the comparsion among the fractures')
    print('-t, --test: process test code without core opration')
    print('-h, --help: print help message')

def parse_args(argv):
    args = argv[1:]
    try:
        opts, args = getopt.getopt(args, 'p:k:cth', ['prefix=', 'cluster='])
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)
    prefix = ''
    n_cluster = 6
    for o, a in opts:
        if o in ('-h', '--help'):
            usage()
            sys.exit(1)
        elif o in ('-p', '--prefix'):
            prefix = a
        elif o in ('-t', '--test'):
            utils = Utils(prefix, k=n_cluster)
            length, names = utils.generate_datas()[-2::]
            print(length, names)
            # print(np.mean(length))
        elif o in ('-k', '--cluster'):
            n_cluster = int(a)
        elif o in ('-c', '--comparsion'):
            utils = Utils(prefix, k=n_cluster)
            utils.comparsion()
        else:
            print('unhandled option')
            sys.exit(3)


def main(argv):
    parse_args(argv)


if __name__ == "__main__":
    print('test time begin at ' + time.asctime(time.localtime(time.time())) + '\n')
    main(sys.argv)
    print('test time end at ' + time.asctime(time.localtime(time.time())) + '\n')