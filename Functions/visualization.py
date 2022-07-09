import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    