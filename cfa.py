import numpy as np

def make_cfa(img):
    cfa = np.zeros_like(img[:,:,0])
    y, x = np.mgrid[0:cfa.shape[0], 0:cfa.shape[1]]
    Rloc = (np.mod(x, 2) == 0) & (np.mod(y, 2) == 0)
    Gloc = np.mod(x+y, 2) == 1
    Bloc = (np.mod(x, 2) == 1) & (np.mod(y, 2) == 1)
    cfa[Rloc] = img[Rloc,0]
    cfa[Gloc] = img[Gloc,1]
    cfa[Bloc] = img[Bloc,2]
    return cfa
