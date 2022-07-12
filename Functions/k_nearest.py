import numpy as np

def kNN(ref_arr, arr_reduced, img_reduced, k, train = True):
    """
    k nearest neighbours, returns the digit which had the smallest euclidean distances among k-neighbours
    :param ref_arr: Reference array with labels in the 1st column
    :param arr_reduced: Principle components of training data, without labels
    :param img_reduced: Principle components of sample image
    :k: number of nearest neighbours
    :train: "True" if sample image comes from training data
    """

   # Euclidian distance between sample image and all images in training array
    arr_rows = arr_reduced.shape[0]
    img_dot = (img_reduced**2).sum(axis=0)*np.ones(shape=(1,arr_rows))
    arr_dot = (arr_reduced[:, :]**2).sum(axis=1)
    dist_arr =  np.sqrt(img_dot + arr_dot - 2*np.dot(img_reduced, arr_reduced[:, :].T))
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