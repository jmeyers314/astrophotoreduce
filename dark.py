import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np

import canont3i
import biweight

def find_badpixels(fns) :
    bad_pixels = []
    outliers = {}
    for fn in fns :
        print "Working on image: ", fn
        cfa = canont3i.read_CR2_as_CFA(fn)
        print "Image shape: ", cfa.shape
        med = np.median(cfa)
        print "Median: ", med
        s = biweight.median_absolute_deviation(cfa, med)/0.6745
        print "MAD: ", s*0.6745
        print "StDev: ", s
        image_outliers = np.argwhere(cfa > (med + 8*s))
        image_outliers = np.concatenate([image_outliers, np.argwhere(cfa < (med - 8*s))])
        print "Number of 8-sigma outliers:", image_outliers.shape[0]
        print
        print
        for outlier in image_outliers:
            key = tuple(outlier)
            if key in outliers.keys():
                outliers[key].append(fn)
            else:
                outliers[key] = [fn]
    print "Bad pixels"
    print "----------"
    for pixel in outliers.keys():
        if len(outliers[pixel]) > len(fns)**0.5:
            print pixel, len(outliers[pixel])
            bad_pixels.append(pixel)
    return bad_pixels

def make_dark(fns, bad_pixels):
    cfa0 = canont3i.read_CR2_as_CFA(fns[0])
    height, width = cfa0.shape
    stack = np.empty((height, width, len(fns)), dtype=cfa0.dtype)
    for i, fn in enumerate(fns) :
        print 'i: ', i
        cfa = canont3i.read_CR2_as_CFA(fn)
        stack[:,:,i] = cfa
    print 'medianing'
    med = np.median(stack)
    print 'layer medianing'
    dark = np.median(stack, axis=2)
    for bp in bad_pixels:
        dark[bp] = med
    return dark
