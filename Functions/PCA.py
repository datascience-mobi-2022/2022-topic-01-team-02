import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
from Functions import data_load as dat

std0 = dat.std0_load()
train_arr_cleaned = dat.clean_train_arr()


def z_transformation(single_image):
    """
    centering and scaling, returns transformed dataset and single image for comparison
    (bissl unnötig)

    :param set: training dataset, no labels
    :param single_image: image for classification from test dataset
    """

    single_cleaned = np.delete(single_image, std0)

    z_arr = (train_arr_cleaned - np.mean(train_arr_cleaned, axis = 0))/np.std(train_arr_cleaned, axis = 0)
    z_single = (single_cleaned - np.mean(train_arr_cleaned, axis = 0))/np.std(train_arr_cleaned, axis = 0)

    return z_arr, z_single


def z_arr(arr):
    """
    centering and scaling, returns z-transformed array

    :param arr: cleaned array, without label
    """
    clean_arr = np.delete(arr, std0, 1)
    z_arr = (clean_arr - np.mean(train_arr_cleaned, axis = 0))/np.std(train_arr_cleaned, axis = 0)    
    
    return z_arr


def z_img(img):
    """
    centering and scaling, returns transformed single image

    :param img: image without label
    """
    img_cleaned = np.delete(img, std0)

    z_img = (img_cleaned - np.mean(train_arr_cleaned, axis = 0))/np.std(train_arr_cleaned, axis = 0)
    
    return z_img


def PCA(clean_set, clean_img, num_components=10):
    """
    principal component analysis, returns array of dataset and single image with defined number of components

    :param clean_set: training dataset (2D-Array; no label; after z-transformation)
    :param clean_img: single image (1D-Array; no label; after z-transformation)
    :param num_components): number of components (integer)
    :return: returns dataset and single image with reduced dimensions and PCs as features
    """

    #variance
    cov_arr = np.cov(clean_set, rowvar = False)

    #eigenvalues, eigenvectors
    eigen_val, eigen_vec = np.linalg.eigh(cov_arr)

    #sorting
    index_sorted = np.argsort(eigen_val)[::-1]
    sorted_eigenval = eigen_val[index_sorted]
    sorted_eigenvec = eigen_vec[:,index_sorted]

    #selecting subset
    eigenvec_subset = sorted_eigenvec[:, 0:num_components]

    #dimension reduction
    set_reduced = np.dot(eigenvec_subset.transpose(), clean_set.transpose()).transpose()
    img_reduced = np.dot(eigenvec_subset.transpose(), clean_img.transpose()).transpose()

    return set_reduced, img_reduced


def create_sorted_eigenvec(num_components):
    """
    needed for PCA of only the dataset or image

    :param num_components: number of components (integer)
    :return: 2D-Array of sorted eigenvectors
    """

    cov_arr = np.cov(train_arr_cleaned, rowvar = False)
    eigen_val, eigen_vec = np.linalg.eigh(cov_arr)
    index_sorted = np.argsort(eigen_val)[::-1]
    sorted_eigenvec = eigen_vec[:,index_sorted]
    eigenvec_subset = sorted_eigenvec[:, 0:num_components]
    
    return eigenvec_subset


def image_only(z_img, sorted_eigenvec):
    """
    create principal components for one image
    
    :param z_img: image from either daset (1D-Array; no label; after z-transformation)
    :param sorted_eigenvec: 2D-Array of sorted eigenvectors from train or test dataset
    :return: image with reduced dimensions and PCs as features
    """

    img_reduced = np.dot(sorted_eigenvec.transpose(), z_img.transpose()).transpose()
    return img_reduced


def arr_only(z_arr, sorted_eigenvec):
    """
    create principal components for dataset
    
    :param z_arr: training or test dataset (2D-Array; no label; after z-transformation)
    :param sorted_eigenvec: 2D-Array of sorted eigenvectors from train or test dataset
    :return: dataset with reduced dimensions and PCs as features
    """
    arr_reduced = np.dot((sorted_eigenvec).transpose(), z_arr.transpose()).transpose()
    return arr_reduced


def visualize_2d(reduced_dataset, labels, i=0, j=1):
    """
    scatterplot of principal images based on principal components

    :param reduced_dataset: dataset with PCs as features
    """
    pca_df = pd.DataFrame(data = {f'PC{i}':reduced_dataset[:, i], f'PC{j}':reduced_dataset[:, j]})
    plt.figure(figsize = (12,12))

    sb.relplot(data = pca_df, x = f'PC{i}', y = f'PC{j}', hue = labels ,s = 10, palette = 'icefire', legend='full', style=labels)


def PCA_SVD(clean_set, num_components):
    """
    principal component analysis, returns array of dataset and single image with defined number of components

    :param clean_set: training dataset (2D-Array; no label; after z-transformation)
    :param num_components): number of components (integer)
    :return: returns dataset with reduced dimensions and PCs as features
    """

    #calculate SVD
    U, S, Vt = np.linalg.svd(clean_set, full_matrices=False)


    #dimension reduction
    #@ macht matrix multiplikation
    set_reduced = clean_set @ Vt[:num_components].T
    return set_reduced