import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
import data_load as dat

std0 = dat.std0_load()
train_arr_cleaned = dat.clean_train_arr()

# import pixels with std = 0 as list
#std0_df = pd.read_csv("data/pca/std0.csv")
#std0 = list(std0_df["0"])

# import trainarray where all pixels with std = 0 are deleted
#train_arr_cleaned_df = pd.read_csv("data/pca/cleaned_train_array.csv")
#train_arr_cleaned = train_arr_cleaned_df.to_numpy()


def z_transformation(set, single_image):
    """
    centering and scaling, returns transformed dataset and single image for comparison

    :param set: training dataset, no labels
    :param single_image: image for classification from test dataset
    """
    
    std0 = []
    for i in range(0, set.shape[1]):
        if np.std(set[:, i]) == 0:
            std0.append(i)
    set_cleaned =  np.delete(set, std0, 1)
    single_cleaned = np.delete(single_image, std0)

    z_set = (set_cleaned - np.mean(set_cleaned, axis = 0))/np.std(set_cleaned, axis = 0)
    z_single = (single_cleaned - np.mean(set_cleaned, axis = 0))/np.std(set_cleaned, axis = 0)


    return z_set, z_single


def z_arr(arr):
    """
    centering and scaling, returns z-transformed array

    :param arr: cleaned array, without label
    """
    
    z_arr = (arr - np.mean(train_arr_cleaned, axis = 0))/np.std(train_arr_cleaned, axis = 0)    
    
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


def create_sorted_eigenvec(clean_set, num_components = 10):
    """
    needed for PCA of only the dataset or image

    :param clean_set: training dataset (2D-Array; no label; after z-transformation)
    :param num_components: number of components (integer); default is set to 10
    :return: 2D-Array of sorted Eigenvectors
    """

    cov_arr = np.cov(clean_set, rowvar = False)
    eigen_val, eigen_vec = np.linalg.eigh(cov_arr)
    index_sorted = np.argsort(eigen_val)[::-1]
    sorted_eigenvec = eigen_vec[:,index_sorted]
    return sorted_eigenvec


def image_only(clean_img, sorted_eigenvec):
    """
    create principal components for one image
    
    :param clean_img: image from either daset (1D-Array; no label; after z-transformation)
    :param sorted_eigenvec: 2D-Array of sorted eigenvectors from train or test dataset
    :return: image with reduced dimensions and PCs as features
    """

    img_reduced = np.dot(sorted_eigenvec.transpose(), clean_img.transpose()).transpose()
    return img_reduced


def set_only(clean_set, sorted_eigenvec):
    """
    create principal components for dataset
    
    :param clean_set: training dataset (2D-Array; no label; after z-transformation)
    :param sorted_eigenvec: 2D-Array of sorted eigenvectors from train or test dataset
    :return: dataset with reduced dimensions and PCs as features
    """

    set_reduced = np.dot((sorted_eigenvec).transpose(), clean_set.transpose()).transpose()
    return set_reduced


def visualize_2d(reduced_dataset, labels, i=0, j=1):
    """
    scatterplot of principal images based on principal components

    :param reduced_dataset: dataset with PCs as features
    """
    pca_df = pd.DataFrame(data = {f'PC{i}':reduced_dataset[:, i], f'PC{j}':reduced_dataset[:, j]})
    plt.figure(figsize = (12,12))

    sb.relplot(data = pca_df, x = f'PC{i}', y = f'PC{j}', hue = labels ,s = 10, palette = 'icefire', legend='full', style=labels)