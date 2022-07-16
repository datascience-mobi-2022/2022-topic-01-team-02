import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Functions.data_load as dat


train_array, test_array = dat.load_data()
avg_list = dat.load_avg_list()
weighting_list = dat.load_weighting()


def avg_digit_img(dat, digit):
    '''
    returns average intensities for all images, displaying the questioned digit

    :param dat: array (with labels)
    :param digit: the digit which average image will be returned
    '''
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
    '''
    returns average intensities for all images of all digits

    :param dat: array (with labels)
    '''
    fig = plt.figure(figsize=(10,5))
    for i in range(0,10):
        img = avg_digit_img(dat, digit = i)
        img.shape = (28,28)
        fig.add_subplot(2, 5, i+1)
        plt.imshow(img, 'gray')

    plt.show()


def mean_pred(array, index):
    '''
    returns prediction, by comparing input-image to average-images of all digits

    :param array: Train/Test Array as source for input image (with labels)
    :param index: image that should be inputed from the array
    '''

    intensities_list = [] 
    sample_img = array[index, 1:] 

    for i in range(0,10):
        diff_arr = sample_img - avg_list[i]
        
        diff_list = []
        for j in range(0, 784):
            diff_list.append(diff_arr[0, j])

        intensity_sum = 0
        for k in range(0, len(diff_list)):
            diff_list[k] = diff_list[k]**2
            diff_list[k] = np.sqrt(diff_list[k])
            intensity_sum += diff_list[k]

        intensities_list.append(intensity_sum)

    return intensities_list.index(min(intensities_list))


def mean_pred_weighted(array, index):
    '''
    returns prediction, by comparing input-image to average-images of all digits and weights the differences regarding each pixel's proportion to the global variance

    :param array: Train/Test Array as source for input image (with labels)
    :param index: image that should be inputed from the array
    '''

    intensities_list = [] 
    sample_img = array[index, 1:] 

    for i in range(0,10):
        diff_arr = sample_img - avg_list[i]
        
        diff_list = []
        for j in range(0, 784):
            diff_list.append(diff_arr[0, j])

        diff_list_weight = np.multiply(diff_list, weighting_list)

        intensity_sum = 0
        for k in range(0, len(diff_list_weight)):
            diff_list_weight[k] = diff_list_weight[k]**2
            diff_list_weight[k] = np.sqrt(diff_list_weight[k])
            intensity_sum += diff_list_weight[k]

        intensities_list.append(intensity_sum)

    return intensities_list.index(min(intensities_list))

def avg_validation(function, array, sample_size):
    '''
    returns the prediction accuracy

    :param function: use mean_pred or mean_pred_weighted
    :param array: use train_array or test_array
    :param sample_size: number of images to test accuracy
    '''

    true = 0
    false = 0

    for i in range(0, sample_size):
        result = function(array, i)
        if result == array[i, 0]:
            true += 1
        else:
            false +=1
    
    return print(f'Anzahl richtig erkannter Digits: {true} \n\
    Anzahl falsch erkannter Digits: {false} \n\
    Richtig: {true/sample_size*100} Prozent')
