import numpy as np
from PIL import Image as im
import pandas as pd
import Functions.PCA as pca
import Functions.k_nearest as knn
import Functions.data_load as dat

train_array, test_array = dat.load_data()
cleaned_arr = dat.clean_train_arr()
z_arr = pca.z_arr(cleaned_arr)
reduced_arr = pca.arr_only(z_arr, pca.create_sorted_eigenvec(30))

def load_jpg(file_path):
    """
    loads jpg image and converts to np array (2D)

    :param file_path: str of relative path
    :return: 2D Array od image
    """
    img = im.open(file_path)
    img = img.convert('L')
    img = np.asarray(img)
    while img.shape[0]%28 != 0:
        img = img[:img.shape[0]-1, :img.shape[1]-1]
    n = img.shape[0]//28

    img_arr = np.zeros((28,28))
    for h in range(0,28):
        for i in range(0,28):
            value = 0
            for j in range(0,n):
                for k in range(0,n):
                    value += img[j+(h*n), k+(i*n)]
            img_arr[h,i] = value//n**2
    img_arr = 255-img_arr



    return img_arr