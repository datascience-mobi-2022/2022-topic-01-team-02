import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    """
    
    :return: nummpy array of train dataset and test dataset
    """
    train_digits = pd.read_csv("data/mnist_train.csv")
    test_digits = pd.read_csv("data/mnist_test.csv")
    train_array = train_digits.to_numpy()
    test_array = test_digits.to_numpy()
    return train_array, test_array


def show_digit(dataset, sample = 0):
    """

    :param dataset: train or test dataset
    :param sample: integer for index of sample image
    :return: shows digit
    """
    img = dataset[sample, 1:]
    img.shape = (28,28)
    plt.title(f'image at index {sample}')
    plt.imshow(img2, 'gray')
    return plt.imshow(img, 'gray')


def ten_digits(dataset):
    """
    
    :param dataset: numpy array of train or test dataset
    :return: prints first ten numbers
    """

    liste = []
    for i in range(0,10):
        j = 0
        while i != train_array[j, 0]:
            j += 1
        liste.append(j)
    
    fig = plt.figure(figsize=(10,5))
    for i in range(0,10):
        img = train_array[liste[i], 1:]
        img.shape = (28,28)
        fig.add_subplot(2, 5, i+1)
        plt.imshow(img, 'gray')

    plt.show()