import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from Functions import PCA as pca
from Functions import data_load as dat



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
    

def correlation_heatmap(arr, name = 'array'):
    cov_arr = np.cov(arr, rowvar = False)
    cov_df = pd.DataFrame(cov_arr)
    sb.set(rc={"figure.dpi":200, "figure.figsize":(5, 5)})
    sb.heatmap(cov_df, cmap="viridis", annot=False, square=True, cbar_kws={"shrink": 0.8})
    plt.title(f'Correlation of {name}', fontsize =9)


def principal_comp_2d(reduced_arr, labels, i=1, j=2):
    """
    scatterplot of principal images based on principal components

    :param reduced_arr: dataset with PCs as features
    :param labels: in this case labels of digits for colour coding in plot
    """
    pca_df = pd.DataFrame(data = {f'PC{i}':reduced_arr[:, i-1], f'PC{j}':reduced_arr[:, j-1]})
    sb.set(rc={"figure.dpi":250, "figure.figsize":(5, 5)})

    sb.relplot(data = pca_df, x = f'PC{i}', y = f'PC{j}', hue = labels ,s = 1, palette = 'icefire', legend='full')