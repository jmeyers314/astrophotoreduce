import numpy as np

def median_absolute_deviation(x, M=None) :
    if M is None:
        M = np.median(x)
    return np.median(abs(x - M))

def _biweight_location_work(x, M, MAD, c) :
    u = (x-M) / (c*MAD)
    w = abs(u) < 1.0
    if w.sum() == 0.0:
        return M
    term = (1.0 - u[w]**2)**2
    num = ((x[w]-M)*term).sum()
    den = term.sum()
    CBI = M + num/den
    return CBI

def biweight_location(x, c=6.0, NaN=None, niter=4) :
    if NaN:
        x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.NaN
    CBI = np.median(x)
    for i in xrange(niter):
        M = CBI
        MAD = median_absolute_deviation(x, M)
        CBI = _biweight_location_work(x, M, MAD, c)
    return CBI

def _biweight_scale_work(x, M, MAD, c) :
    u = (x-M) / (c*MAD)
    w = abs(u) < 1.0
    if w.sum() == 0:
        return np.NaN
    term = u[w]**2
    num = (len(x) * ((x[w]-M)**2 * (1-term)**4).sum())**0.5
    den = abs(((1.0-term)*(1.0-5*term)).sum())
    return num/den

def biweight_scale(x, zero=None, c=9.0, NaN=None, niter=4) :
    if NaN:
        x = x[np.isfinite(x)]
    if zero is None:
        M = biweight_location(x)
    else:
        M = zero
    MAD = median_absolute_deviation(x, M)
    SBI = MAD/0.6745
    for i in xrange(niter):
        MAD = SBI*0.6745
        SBI = _biweight_scale_work(x, M, MAD, c)
    return SBI
