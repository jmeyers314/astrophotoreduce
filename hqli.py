# High Quality Linear Interpolation algorithm implementation
# Reference: Malvar, H.S., Li-Wei He, Cutler, R.
#            http://research.microsoft.com/pubs/102068/demosaicing_icassp04.pdf
#
#

import numpy as np
import scipy.signal as signal

# Four kernels to be convolved with CFA array
def _G_at_BR(cfa):
    kernel = [[ 0,   0, -1.0, 0,    0],
              [ 0,   0,    2, 0,    0],
              [-1.0, 2,    4, 2, -1.0],
              [ 0,   0,    2, 0,    0],
              [ 0,   0, -1.0, 0,    0]]
    return signal.convolve(cfa, kernel, mode='same')/8.

# red value at green pixel in red row and blue col or...
# blue value at green pixel in blue row and red col.
def _RB_at_G_in_RBrow_BRcol(cfa):
    kernel = [[ 0,  0, 0.5,  0,  0],
              [ 0, -1,   0, -1,  0],
              [-1,  4,   5,  4, -1],
              [ 0, -1,   0, -1,  0],
              [ 0,  0, 0.5,  0,  0]]
    return signal.convolve(cfa, kernel, mode='same')/8.

# red value at green pixel in blue row and red col or...
# blue value at green pixel in red row and blue col.
def _RB_at_G_in_BRrow_RBcol(cfa):
    kernel = [[ 0,  0, -1,  0,   0],
              [ 0, -1,  4, -1,   0],
              [0.5, 0,  5,  0, 0.5],
              [ 0, -1,  4, -1,   0],
              [ 0,  0, -1,  0,   0]]
    return signal.convolve(cfa, kernel, mode='same')/8.

# red value at blue pixel or...
# blue value at red pixel.
def _RB_at_BR(cfa):
    kernel = [[ 0,   0, -1.5, 0,    0],
              [ 0,   2,    0, 2,    0],
              [-1.5, 0,    6, 0, -1.5],
              [ 0,   2,    0, 2,    0],
              [ 0,   0, -1.5, 0,    0]]
    return signal.convolve(cfa, kernel, mode='same')/8.

def hqli(cfa):
    # initialize output arrays
    R = np.zeros_like(cfa, dtype=np.float64)
    G = np.zeros_like(cfa, dtype=np.float64)
    B = np.zeros_like(cfa, dtype=np.float64)
    # coordinate index arrays
    y, x = np.mgrid[0:cfa.shape[0], 0:cfa.shape[1]]

    # create groups of indices based on Bayer pattern
    Rloc = (np.mod(x, 2) == 0) & (np.mod(y, 2) == 0)
    Gloc = np.mod(x+y, 2) == 1
    Bloc = (np.mod(x, 2) == 1) & (np.mod(y, 2) == 1)
    G_in_Brow_Rcol = (np.mod(x, 2) == 0) & (np.mod(y, 2) == 1)
    G_in_Rrow_Bcol = (np.mod(x, 2) == 1) & (np.mod(y, 2) == 0)

    # copy data that doesn't need interpolation
    R[Rloc] = cfa[Rloc]
    G[Gloc] = cfa[Gloc]
    B[Bloc] = cfa[Bloc]

    # fill in the green data at the blue/red locations
    tmp = _G_at_BR(cfa)
    G[Rloc] = tmp[Rloc]
    G[Bloc] = tmp[Bloc]

    # fill in the blue/red data
    tmp = _RB_at_G_in_BRrow_RBcol(cfa)
    B[G_in_Rrow_Bcol] = tmp[G_in_Rrow_Bcol]
    R[G_in_Brow_Rcol] = tmp[G_in_Brow_Rcol]

    tmp = _RB_at_G_in_RBrow_BRcol(cfa)
    B[G_in_Brow_Rcol] = tmp[G_in_Brow_Rcol]
    R[G_in_Rrow_Bcol] = tmp[G_in_Rrow_Bcol]

    tmp = _RB_at_BR(cfa)
    B[Rloc] = tmp[Rloc]
    R[Bloc] = tmp[Bloc]

    R[R<0] = 0
    G[G<0] = 0
    B[B<0] = 0
    if cfa.dtype == np.uint8:
        R[R>255] = 255
        G[G>255] = 255
        B[B>255] = 255
    else:
        R[R>65535] = 65535
        G[G>65535] = 65535
        B[B>65535] = 65535

    return np.array(np.dstack([R, G, B]), dtype=cfa.dtype)
