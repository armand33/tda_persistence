import sys
import time

from pprint import pprint
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math

from filtration_sparsebool import Filtration
# from filtration2 import Filtration



def usage():
    print("{} filtration_source_file log [save]".format(sys.argv[0]))
    print("\tfiltration_source_file: file from which the filtration is extracted")
    print("\tlog: int assessing that the x-scale be logarithmic")
    print("\tsave: file in which to export the barcode (default=None)")
    sys.exit(1)


def abscisse(x,b,xm):
    return x if not b else xm if x == 0 else math.log(x)

THRESHOLD = 0.5



if __name__ == '__main__':

    if len(sys.argv) == 4:
        log = (int(sys.argv[2]) == 1)
        save = sys.argv[3]
    elif len(sys.argv) == 3:
        log = (int(sys.argv[2]) == 1)
        save = None
    else:
        usage()

    s0 = time.time()

    # print("Initiate filtration")
    filtration = Filtration(sys.argv[1])

    # print("Compute boundary matrix")
    s = time.time()
    filtration.boundary_matrix()
    bm = time.time() - s

    # print("Reduce boundary matrix")
    s = time.time()
    filtration.reduce()
    rd = time.time() - s

    # print("Compute barcode")
    s = time.time()
    filtration.barcode()

    bc = time.time() - s
    tt = time.time() - s0
    print("%fs | %fs | %fs | %fs |" % (tt, bm, rd, bc))


    print("Plot...")
    sns.set_style("white")

    xm = min([abscisse(src,log,0) for _, src, _ in filtration.bc_clean if (src != 0 or not log)])
    xM = max([abscisse(tgt,log,xm) for _, _, tgt in filtration.bc_clean if tgt != np.inf])
    dM = max([dim for dim, _, _ in filtration.bc_clean])

    max_x = xM * (1 + 1 / dM)

    n = 0

    plt.figure(figsize=(20, 10))
    filtration.bc_clean.sort(key = lambda tup : tup[0])
    for i, (dim, src, tgt) in tqdm(enumerate(filtration.bc_clean),total=len(filtration.bc_clean)):
        #filtration A
        #if abs(abscisse(tgt,log,xm)-abscisse(src,log,xm)) > THRESHOLD:

        #filtration B
        if abs(tgt-src) > THRESHOLD:

            plt.plot([abscisse(src,log,xm), min(abscisse(tgt,log,xm),max_x)], [10 * dim + n, 10 * dim + n],c=str(dim / (dM+1)))
            n += 1

    plt.xlim((xm, max_x))
    plt.ylim((-5, n + 10 * dM + 10))
    plt.axis("off")

    if save:
        plt.savefig(save)
    else:
        plt.show()
