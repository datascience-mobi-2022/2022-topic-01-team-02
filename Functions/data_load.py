import numpy as np
import pandas as pd

def load_data():
    """
    
    returns array of train dataset and array of test dataset
    """
    train_digits = pd.read_csv("data/mnist_train.csv")
    test_digits = pd.read_csv("data/mnist_test.csv")
    train_array = train_digits.to_numpy()
    test_array = test_digits.to_numpy()
    return train_array, test_array