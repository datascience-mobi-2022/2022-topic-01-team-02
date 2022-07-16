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


def load_z_arr_train():
    """
    
    :return: z-transformed train-array of train dataset and test dataset
    """
    z_arr_train_df = pd.read_csv("data/pca/z_array.csv", header=None)
    z_arr_train = z_arr_train_df.to_numpy()
    return z_arr_train


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
    

def load_val_arr():
    """

    import validation array for k's and PC's
    """
    val_arr_df = pd.read_csv("data/val_array.csv", header=None)
    val_arr = val_arr_df.to_numpy()
    return val_arr


def load_val_arr_1st():
    """

    import validation array for k's and PC's
    """
    val_arr_1st_df = pd.read_csv("data/val_array2.csv", header=None)
    val_arr_1st = val_arr_1st_df.to_numpy()
    return val_arr_1st


def load_val_arr_2nd():
    """

    import validation array for k's and PC's
    """
    val_arr_2nd_df = pd.read_csv("data/val_array_PC30_40.csv", header=None)
    val_arr_2nd = val_arr_2nd_df.to_numpy()
    return val_arr_2nd


def load_precise_val_arr():
    """

    import validation array (10000 references) for k's and PC's
    """
    precise_val_arr_df = pd.read_csv("data/precise_val_array.csv", header=None)
    precise_val_arr = precise_val_arr_df.to_numpy()
    return precise_val_arr


def load_val_arr_index():
    """

    import validation array for k's and PC's with indices
    """
    val_arr_index_df = pd.read_csv("data/val_array_index.csv", index_col = 0)
    val_arr_index = val_arr_index_df.to_numpy()
    return val_arr_index


def load_avg_list():
    avg_list = []
    for i in range(0, 10):
        avg_df = pd.read_csv(f'data/avg/avg_{i}.csv', header=None)
        avg_arr = avg_df.to_numpy()
        avg_list.append(avg_arr)
    return avg_list

def load_weighting():
    weighting_list = []
    weighting_df = pd.read_csv("data/avg/weighting.csv", header=None)
    weighting_array = weighting_df.to_numpy()   
    for k in range(0, 784):
        weighting_list.append(weighting_array[k, 0])
    return weighting_list

def load_variance():
    var = pd.read_csv('data/pca/variance_pc.csv', header=None)
    var = var.to_numpy()
    var= np.insert(var, 0, 0)
    return var
