import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from Functions import PCA as pca
from Functions import data_load as dat

train_array, test_array = dat.load_data()
train_arr_cleaned = dat.clean_train_arr()
z_arr = pca.z_arr(train_arr_cleaned)
reduced_arr = pca.arr_only(z_arr, pca.create_sorted_eigenvec(30))
var = dat.load_variance()


def show_digit(dataset, sample = 0):
    """

    :param dataset: train or test dataset
    :param sample: integer for index of sample image
    :return: shows digit
    """
    img = dataset[sample, 1:]
    img.shape = (28,28)
    plt.title(f'image at index {sample}')
    plt.imshow(img, 'gray')
    return plt.imshow(img, 'gray')


def ten_digits(dataset):
    """
    
    :param dataset: numpy array of train or test dataset
    :return: prints first ten numbers
    """

    liste = []
    for i in range(0,10):
        j = 0
        while i != dataset[j, 0]:
            j += 1
        liste.append(j)
    
    fig = plt.figure(figsize=(10,5))
    for i in range(0,10):
        img = dataset[liste[i], 1:]
        img.shape = (28,28)
        fig.add_subplot(2, 5, i+1)
        plt.imshow(img, 'gray')

    plt.show()
    
def ten_digits_z_transfo(dataset, label):
    """
    
    :param dataset: numpy array of train or test dataset, not containing label in first column
    :param label: 
    :return: prints first ten numbers
    """

    

    liste = []
    for i in range(0,10):
        j = 0
        while i != label[j]:
            j += 1
        liste.append(j)
    
    fig = plt.figure(figsize=(10,5))
    for i in range(0,10):
        img = dataset[liste[i], :]
        img.shape = (28,28)
        fig.add_subplot(2, 5, i+1)
        plt.imshow(img, 'gray')

    plt.show()

def correlation_heatmap(arr, name = 'array'):
    cov_arr = np.cov(arr, rowvar = False)
    cov_df = pd.DataFrame(cov_arr)
    sb.set(rc={"figure.dpi":200, "figure.figsize":(5, 5)})
    sb.heatmap(cov_df, cmap="viridis", annot=False, square=True, cbar_kws={"shrink": 0.8})
    plt.title(f'Correlation of {name}', fontsize =9)


def heatmap(arr, name = 'variable parts'):
    arr_df = pd.DataFrame(arr)
    sb.set(rc={"figure.dpi":200, "figure.figsize":(5, 5)})
    sb.heatmap(arr_df, cmap="viridis", annot=False, annot_kws={"size": 5}, square=True, cbar_kws={"shrink": 0.8})
    plt.title(f'Accuracy with variable {name}', fontsize =9)


def principal_comp_2d(reduced_arr, labels, i=1, j=2):
    """
    scatterplot of principal images based on principal components

    :param reduced_arr: dataset with PCs as features
    :param labels: in this case labels of digits for colour coding in plot
    """
    pca_df = pd.DataFrame(data = {f'PC{i}':reduced_arr[:, i-1], f'PC{j}':reduced_arr[:, j-1]})
    sb.set(rc={"figure.dpi":150, "figure.figsize":(5, 5)})

    sb.relplot(data = pca_df, x = f'PC{i}', y = f'PC{j}', hue = labels ,s = 1, palette = 'icefire', legend='full')


def PC_variance():
    fig = plt.figure(figsize=(8,4))
    plt.grid(True, linewidth=.5)
    plot = plt.plot([x for x in range(0,717)], var)
    plt.xlabel('Principal Components')
    plt.ylabel('obtained variance in percent')
    plt.hlines(y = var[30], xmin=0, xmax=30, color='r', linewidth=.5)
    plt.vlines(x=30, ymin=0, ymax=var[30], color='r', linewidth=.5)


def display_add_img():
    