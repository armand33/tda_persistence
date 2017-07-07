
from itertools import combinations

import numpy as np
from tqdm import tqdm


def find_pivot(matrix, column):
    try:
        return max(matrix.d[column])
    except (KeyError, ValueError):
        return False


class SparseBool:
    """ Tailor made sparse boolean matrix, wrapping a dictionary with columns
        indices for keys, and rows of True values
        Access is O(1)
        Main operation is at worst O(size of filtration)
    """

    def __init__(self, tots):
        self.d = dict()
        self.length = tots

    def set(self, row, col):
        try:
            self.d[col].add(row)
        except KeyError:
            self.d[col] = set([row])

    def add(self, colA, colB):
        try:
            self.d[colA].symmetric_difference_update(self.d[colB])
        except (KeyError, TypeError):
            pass

    def __str__(self):
        assert self.length < 30
        m = np.zeros((self.length, self.length), dtype="int8")
        for col, rows in self.d.items():
            for row in rows:
                m[row, col] = 1
        return m.__str__()


class Simplex:
    """ Basic simplex class with val as its time of appearance, its dimension,
        and vert as the indices of its members
    """

    def __init__(self, val, dim, vert):
        # Time of appearance
        self.val = val
        # Dimension of the simplex
        self.dim = dim
        # Vertices of the simplex
        self.vert = vert
        # Its boundary
        self.boundary = [
            "-".join(map(str, sorted(b)))
            for b in combinations(vert.split("-"), len(self.vert.split("-")) - 1)
        ]

    def __str__(self):
        return "val: {}; dim: {}; vert: {}, boundary: {}".format(
            self.val, self.dim, self.vert, self.boundary)

    def __repr__(self):
        return self.__str__()


class Filtration:

    def __init__(self, filepath):

        filtration = list()

        text_file = open(filepath, "r")
        for line in text_file:
            if not line.startswith('//'):
                line = line.split()
                simplex = Simplex(
                    float(line[0]),
                    int(line[1]),
                    str(line[2]) if len(line) == 3
                        else "-".join(map(str, sorted(line[2:])))
                )
                filtration.append(simplex)
        text_file.close

        self.filtration = sorted(filtration, key=lambda x: x.val)
        self.to_idx = dict(
            [(s.vert, i) for i, s in enumerate(self.filtration)])

    def boundary_matrix(self):

        matrix = SparseBool(len(self.filtration))

        for column, simplex in tqdm(enumerate(self.filtration),total=len(self.filtration)):
            if simplex.dim == 0:
                continue
            for name in simplex.boundary:
                row = self.to_idx[name]
                matrix.set(row, column)

        self.bm = matrix

    def reduce(self):

        pivots_dic = dict()

        for column in tqdm(range(self.bm.length)):
            row = find_pivot(self.bm, column)
            while row in pivots_dic.keys():
                self.bm.add(column, pivots_dic[row])
                row = find_pivot(self.bm, column)
            if row:
                pivots_dic[row] = column
        #print(pivots_dic)
        self.pivots = pivots_dic

    def barcode(self):

        bars = list()
        for col in tqdm(range(self.bm.length)):
            if not self.bm.d.get(col) and col not in self.pivots.keys():
                bars.append((col, np.inf))
        bars.extend(self.pivots.items())
        self.barcode = bars
        #print(bars)
        self.bc_clean = [
            (
                self.filtration[a].dim,
                self.filtration[a].val,
                (self.filtration[b].val if b != np.inf else np.inf)
            )
            for a, b in bars
        ]
