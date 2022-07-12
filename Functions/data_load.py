import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    """
    
    :return: nummpy array of train dataset and test dataset
    """
    train_digits = pd.read_csv("data/mnist_train.csv", header=None)
    test_digits = pd.read_csv("data/mnist_test.csv", header=None)
    train_array = train_digits.to_numpy()
    test_array = test_digits.to_numpy()
    return train_array, test_array


def std0_load():
    """
    
    import pixels with std = 0 as list
    """
    std0_df = pd.read_csv("data/pca/std0.csv")
    std0 = list(std0_df["0"])
    return std0


def clean_train_arr():
    """

    import trainarray where all pixels with std = 0 are deleted
    """
    train_arr_cleaned_df = pd.read_csv("data/pca/cleaned_train_array.csv", header=None)
    train_arr_cleaned = train_arr_cleaned_df.to_numpy()
    return train_arr_cleaned
    