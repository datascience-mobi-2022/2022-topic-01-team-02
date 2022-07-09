import numpy as np
import pandas as pd

def avg_images_one_to_nine(dataset):
    """
    
    :param dataset: numpy array of train or test dataset
    """

    fig = plt.figure(figsize=(10,5))
    for i in range(0,10):
        list_digit = []
    for j in range(0, dataset.shape[0]):
        if dataset[j, 0] == i:
            list_digit.append(j)
        avg = np.zeros((1,784))
        for k in range(0, len(list_digit)):
            avg += dataset[list_digit[k], 1:]
        avg /= len(list_digit)

        img = avg
        img.shape = (28,28)
        fig.add_subplot(2, 5, i+1)
        plt.imshow(img, 'gray')

    plt.show()