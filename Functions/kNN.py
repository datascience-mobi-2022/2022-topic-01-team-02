import numpy as np
import pandas as pd
#load data
train_digits = pd.read_csv("data/mnist_train.csv")
test_digits = pd.read_csv("data/mnist_test.csv")
#convert pandas Data Frame to Numpy Array
train_array = train_digits.to_numpy()
test_array = test_digits.to_numpy()


def distances(reference_array, sample):
    ref_rows = reference_array.shape[0]

    sample_dot = (sample**2).sum(axis=0)*np.ones(shape=(1,ref_rows))
    ref_dots = (reference_array[:, :]**2).sum(axis=1)
    dist_squared =  sample_dot + ref_dots - 2*np.dot(sample, reference_array[:, :].T)
    dist_array = np.sqrt(dist_squared)
    dist = dist_array.tolist()[0]
    return dist


def kNN(ref_arr, PCs_arr, PCs_img, k, train = True):
    """
    k nearest neighbours, returns the digit which had the smallest euclidean distances among k-neighbours
    :param ref_arr: Reference array with labels in the 1st column
    :param PCs_arr: Principle components of training data, without labels
    :param PCs_img: Principle components of sample image
    :k: number of nearest neighbours
    :train: "True" if sample image comes from training data
    """

   # Euclidian distance between sample image and all images in training array
    arr_rows = PCs_arr.shape[0]
    img_dot = (PCs_img**2).sum(axis=0)*np.ones(shape=(1,arr_rows))
    arr_dot = (PCs_arr[:, :]**2).sum(axis=1)
    dist_arr =  np.sqrt(img_dot + arr_dot - 2*np.dot(PCs_img, PCs_arr[:, :].T))
    dist = dist_arr.tolist()[0]
    
    counter = [0,0,0,0,0,0,0,0,0,0]
    max_indices = []
    
    # List with labels of k nearest neighbours 
    if train == True:
        k_nearest = sorted(range(len(dist)), key = lambda sub: dist[sub])[1:k+1]
    else:
        k_nearest = sorted(range(len(dist)), key = lambda sub: dist[sub])[0:k]

    # Which label occurs how often?
    for i in range(0, k):
        counter[ref_arr[k_nearest[i],0]] += 1
    for j in range(0, 9):
        if counter[j] == max(counter):
            max_indices.append(j)
            
    # Return the most featured label
    if len(max_indices) == 1:
        return max_indices[0]
        
    # Return the nearest neighbour, in case there are several labels occuring equally often
    else:
        if train == True:
            nearest = sorted(range(len(dist)), key = lambda sub: dist[sub])[1]
        else:
            nearest = sorted(range(len(dist)), key = lambda sub: dist[sub])[0]
        return ref_arr[nearest,0]