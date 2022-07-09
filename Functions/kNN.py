import numpy as np


def distances(reference_array, sample):
    ref_rows = reference_array.shape[0]

    sample_dot = (sample**2).sum(axis=0)*np.ones(shape=(1,ref_rows))
    ref_dots = (reference_array[:, :]**2).sum(axis=1)
    dist_squared =  sample_dot + ref_dots - 2*np.dot(sample, reference_array[:, :].T)
    dist_array = np.sqrt(dist_squared)
    dist = dist_array.tolist()[0]
    return dist


def kNN(PCs_arr, PCs_img, k, train = True):
   """
    :param PCs_arr: Principle components of training data, without labels
    :param PCs_img: Principle components of sample image
    :k: number of nearest neighbours
    :train: "True" if sample image comes from training data
    """

   #Distance calculation
    arr_rows = PCs_arr.shape[0]
    img_dot = (PCs_img**2).sum(axis=0)*np.ones(shape=(1,arr_rows))
    arr_dot = (PCs_arr[:, :]**2).sum(axis=1)
    dist_arr =  np.sqrt(img_dot + arr_dot - 2*np.dot(PCs_img, PCs_arr[:, :].T))
    dist = dist_arr.tolist()[0]
    
    #Sorting
    counter = [0,0,0,0,0,0,0,0,0,0]
    max_indices = []
    
    if train == True:
        k_smallest = sorted(range(len(dist)), key = lambda sub: dist[sub])[1:k+1]
    
    else:
        k_smallest = sorted(range(len(dist)), key = lambda sub: dist[sub])[0:k]

    
    for i in range(0, k):
        counter[train_array[k_smallest[i],0]] += 1

    for j in range(0, 9):
        if counter[j] == max(counter):
            max_indices.append(j)
            

    if len(max_indices) == 1:
        return max_indices[0]

    else:
        if train == True:
            k_smallest = sorted(range(len(dist)), key = lambda sub: dist[sub])[1]
        else:
            k_smallest = sorted(range(len(dist)), key = lambda sub: dist[sub])[0]
        return train_array[k_smallest,0]