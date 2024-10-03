import numpy as numpy

def intensity_histogram(img):
    histogram = numpy.histogram(img.numpy().flatten(), bins = 10, range=(0, 1))
    
    return histogram