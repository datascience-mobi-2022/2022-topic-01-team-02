import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def avg_digit_img(dat, digit):

    list_digit = []
    for i in range(0, dat.shape[0]):
        if dat[i, 0] == digit:
            list_digit.append(i)

    avg = np.zeros((1,784))
    
    for j in range(0, len(list_digit)):
        avg += dat[list_digit[j], 1:]
    avg.shape = (28,28)
    avg /= len(list_digit)

    return avg

def digits(dat):
    fig = plt.figure(figsize=(10,5))
    for i in range(0,10):
        img = avg_digit_img(dat, digit = i)
        img.shape = (28,28)
        fig.add_subplot(2, 5, i+1)
        plt.imshow(img, 'gray')

    plt.show()
    