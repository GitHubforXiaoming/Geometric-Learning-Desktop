import sys
import time
from utils import Utils

def main(argv):
    utils = Utils(argv[1])
    utils.comparsion()


if __name__ == "__main__":
    print('test time begin at ' + time.asctime(time.localtime(time.time())) + '\n')
    main(sys.argv)
    print('test time end at ' + time.asctime(time.localtime(time.time())) + '\n')