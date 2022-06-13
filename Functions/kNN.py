import numpy as np


def distances(reference_array, sample):
    ref_rows = reference_array.shape[0]

    sample_dot = (sample**2).sum(axis=0)*np.ones(shape=(1,ref_rows))
    ref_dots = (reference_array[:, 1:]**2).sum(axis=1)
    dist_squared =  sample_dot + ref_dots - 2*np.dot(sample, reference_array[:, 1:].T)
    dist_array = np.sqrt(dist_squared)
    dist = dist_array.tolist()[0]
    return dist