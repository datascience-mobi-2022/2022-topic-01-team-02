import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from Functions import PCA as pca
from Functions import data_load as dat
from Functions import additional_code as add

train_array, test_array = dat.load_data()
train_arr_cleaned = dat.clean_train_arr()
z_arr = pca.z_arr(train_arr_cleaned)
reduced_arr = pca.arr_only(z_arr, pca.create_sorted_eigenvec(30))
var = dat.load_variance()
val_arr = dat.load_val_arr()
precise_val_arr = dat.load_precise_val_arr()


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
    
def digits_after_z(dataset):
    arr = np.zeros(dataset.shape)
    for i in range(0,3):
        for j in range(1,dataset[:,1:].shape[1]):
            if np.std(dataset[:,j]) != 0:
                arr[i,j] = (dataset[i,j]-np.mean(dataset[:,j]))/np.std(dataset[:,j])

    fig = plt.figure(figsize=(10,5))

    for l in range(0,3):
        ax = fig.add_subplot(2,3,l+1)
        img = dataset[l, 1:]
        img.shape = (28,28)
        im = ax.imshow(img, 'gray')
        fig.colorbar(im)

    for k in range(0,3):
        ax = fig.add_subplot(2,3,k+4)
        img = arr[k, 1:]
        img.shape = (28,28)
        im = ax.imshow(img, 'gray')
        fig.colorbar(im)


def correlation_heatmap(arr, name = 'array'):
    cov_arr = np.cov(arr, rowvar = False)
    cov_df = pd.DataFrame(cov_arr)
    sb.set(rc={"figure.dpi":150, "figure.figsize":(4, 4)})
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
    fig = plt.figure(figsize=(4,2))
    plt.grid(True, linewidth=.5)
    plot = plt.plot([x for x in range(0,717)], var)
    plt.xlabel('Principal Components')
    plt.ylabel('obtained variance in percent')
    plt.hlines(y = var[30], xmin=0, xmax=30, color='r', linewidth=.5)
    plt.vlines(x=30, ymin=0, ymax=var[30], color='r', linewidth=.5)


def heatmap_k_PC(arr, ind_range, col_range, large=True, medium=False, small=False):
    '''
    returns heatmap displaying accuracy for variable k's and PC's
    :param arr: array which displays accuracy for k's on x-axis and PC's on y-axis
    :param ind_range: indeces-labeling, input -> range(a, b) a = start-value, b = end-value + 1
    :param col_range: col-labeling, input -> range(a, b) a = start-value, b = end-value + 1
    '''
    arr_df = pd.DataFrame(arr)
    inds = list(ind_range)
    cols = list(col_range)
    arr_df.set_axis([inds], axis='index', inplace=True)
    arr_df.set_axis([cols], axis = 'columns', inplace=True)
    if large == True:
        sb.set(font_scale=0.5, rc={"figure.dpi":200, "figure.figsize":(2.5, 2.5)})
        plt.title('Accuracy (%) with variable PCs and ks \n 1000 samples', fontsize =8, fontweight = 'bold')

    if medium == True:
        sb.set(font_scale=0.5, rc={"figure.dpi":200, "figure.figsize":(1.75, 1.75)})
        plt.title('Accuracy (%) with variable PCs and ks \n 1000 samples', fontsize =6, fontweight = 'bold')

    if small == True:
        sb.set(font_scale=0.5, rc={"figure.dpi":200, "figure.figsize":(1.5, 1.5)})
        plt.title('Accuracy (%) with variable PCs and ks \n 10000 samples', fontsize =6, fontweight = 'bold')

    sb.heatmap(arr_df, cmap="viridis", square=False, cbar_kws={"shrink": 0.9})

    plt.xlabel('number of k')
    plt.ylabel('number of PC')


def acc_PCs():
    PCs = list(range(1, 41))
    for i in range(2, 3):
        plt.plot(PCs, val_arr[0:40 ,i], label = f"k = {i+1}")
        plt.xlabel('PCs')
        plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Prediction accuracy influenced by number of PCs', fontweight = 'bold')

def self_written_digit():
    img1 = add.load_add_img()
    img2 = 255-add.convert_add_img()
    img3 = add.convert_add_img()

    fig = plt.figure(figsize=(9,3))
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(img1, 'gray')
    ax2 = fig.add_subplot(1,3,2)
    ax2.imshow(img2, 'gray')
    ax3 = fig.add_subplot(1,3,3)
    ax3.imshow(img3, 'gray')