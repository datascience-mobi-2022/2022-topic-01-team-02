import numpy as np
from Functions import data_load as dat
from Functions import PCA as pca

train_array, test_array = dat.load_data()
z_array = dat.load_z_arr_train()

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



def validation_kNN_train(s_size, k=3, PC=30):
    """
    validates the error-rate (string) of kNN for given sample size, k, number of PC

    :param s_size: number of pictures send into kNN-code
    :param k: k nearest neighbours being selected
    :param PC: number of principle components being compared
    """

    true = 0
    false = 0
    eigenvectors_sorted = pca.create_sorted_eigenvec(PC)
    pca_arr = pca.arr_only(z_array, eigenvectors_sorted)
    
    for i in range(0, s_size):

        z_image = z_array[29500+i, :]
        pca_img = pca.image_only(z_image, eigenvectors_sorted)


        result_kNN = kNN(train_array, pca_arr, pca_img, k, train=True)
        if result_kNN == train_array[29500+i, 0]:
            true += 1
        else:
            false += 1

    return print(f'Anzahl richtig erkannter Digits: {true}\n\
Anzahl falsch erkannter Digits: {false}\n\
\nAnteil richtiger Vorhersagen: {(true/s_size)*100}%')




def validation_kNN_train_matrix(s_size, k=3, PC=30):
    """
    validates the error-rate (integer) of kNN for given sample size, k, number of PC

    :param s_size: number of pictures send into kNN-code
    :param k: k nearest neighbours being selected
    :param PC: number of principle components being compared
    """
# PCs und z_array outsourcen in die val_arr-Funktion
    true = 0
    false = 0
    eigenvectors_sorted = pca.create_sorted_eigenvec(PC)
    pca_arr = pca.arr_only(z_array, eigenvectors_sorted)
    
    for i in range(0, s_size):

        z_image = z_array[29500+i, :]
        pca_img = pca.image_only(z_image, eigenvectors_sorted)


        result_kNN = kNN(train_array, pca_arr, pca_img, k, train=True)
        if result_kNN == train_array[29500+i, 0]:
            true += 1
        else:
            false += 1

    return (true/s_size)*100

def false_digits(s_size, k=3, PC=30):
    """
    validates which digits are often not correctly predicted, relative to their appearence
    :param s_size: number of pictures send into kNN-code
    :param k: k nearest neighbours being selected
    :param PC: number of principle components being compared
    """
    digit_counter = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]])
    false_digit_counter = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]])
    eigenvectors_sorted = pca.create_sorted_eigenvec(PC)
    pca_arr = pca.arr_only(z_array, eigenvectors_sorted)

    for i in range(0, s_size):
        z_image = z_array[i, :]
        pca_img = pca.image_only(z_image, eigenvectors_sorted)
        digit_counter[train_array[i, 0]] += 1
        result_kNN = knn.kNN(train_array, pca_arr, pca_img, k, train=True)
        if result_kNN != train_array[i, 0]:
            false_digit_counter[train_array[i, 0]] += 1

    false_digit_proportion = false_digit_counter / digit_counter

    return false_digit_proportion


def validation_kNN(s_size=10000, k=3, PC=29):
    """
    validates the error-rate (string) of kNN for given sample size, k, number of PC

    :param s_size: number of pictures send into kNN-code
    :param k: k nearest neighbours being selected
    :param PC: number of principle components being compared
    """

    true = 0
    false = 0
    eigenvectors_sorted = pca.create_sorted_eigenvec(PC)
    z_array = dat.load_z_arr_train()
    pca_arr = pca.arr_only(z_array, eigenvectors_sorted)
    train_array, test_array = dat.load_data()
    z_test = pca.z_arr(test_array[:,1:])
    
    for i in range(0, s_size):

        z_image = z_test[i, :]
        pca_img = pca.image_only(z_image, eigenvectors_sorted)


        result_kNN = kNN(train_array, pca_arr, pca_img, k, train=False)
        if result_kNN == test_array[i, 0]:
            true += 1
        else:
            false += 1

    return print(f'Accuracy: {(true/s_size)*100}%')