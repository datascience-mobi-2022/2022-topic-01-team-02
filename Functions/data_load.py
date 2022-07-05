import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    """
    
    returns nummpy array of train dataset and test dataset
    """
    train_digits = pd.read_csv("data/mnist_train.csv")
    test_digits = pd.read_csv("data/mnist_test.csv")
    train_array = train_digits.to_numpy()
    test_array = test_digits.to_numpy()
    return train_array, test_array


def show_digit(dataset = train_array, sample = 0):
    """

    :param dataset: train or test dataset
    :param sample: integer for index of sample image
    shows digit
    """
    img = dataset[sample, 1:]
    img.shape = (28,28)
    return plt.imshow(img, 'gray')