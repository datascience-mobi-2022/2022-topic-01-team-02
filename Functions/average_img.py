import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Functions.data_load as dat
import Functions.average_img as avg


train_array, test_array = dat.load_data()


#für avg_list das nehmen ohne shape und mit for loop für alle 10 Zahlen, die dann in avg_list appended werden
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


#Noch speichern als csv und dann über data_load
def avg_list_creation(dat):
    '''
    returns list of average intensities for digit 0 to 10 in 1D arrays 
    
    '''
    avg_list = []
    for digit in range(0,10):

        list_digit = []
        for i in range(0, dat.shape[0]):
            if dat[i, 0] == digit:
                list_digit.append(i)

        avg = np.zeros((1,784))
    
        for j in range(0, len(list_digit)):
            avg += dat[list_digit[j], 1:]
        avg /= len(list_digit)
        avg_list.append(avg)
    return avg_list
    
avg_list = avg_list_creation(train_array)


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
    
    #load weight factors -> !!!transferieren zu load data skript wenn Internet!!!
    weighting_df = pd.read_csv("data/weighting.csv", header=None)
    weighting_array = weighting_df.to_numpy()
    weighting_list = []
    for k in range(0, 784):
        weighting_list.append(weighting_array[k, 0])

    intensities_list = [] 
    sample_img = array[index, 1:] 

    for i in range(0,10):
        diff_arr = sample_img - avg_list[i]
        
        diff_list = []
        for j in range(0, 784):
            diff_list.append(diff_arr[0, j])

        diff_list_weight = np.multiply(diff_list, weighting_list)

        #sum all absolute values of difference list and assign to variable intensity_sum
        intensity_sum = 0
        for k in range(0, len(diff_list_weight)):
            diff_list_weight[k] = diff_list_weight[k]**2
            diff_list_weight[k] = np.sqrt(diff_list_weight[k])
            intensity_sum += diff_list_weight[k]

        #append intensities_list by intensity sum
        #at the end of for loop, intensites_list contains 1 value for each of the 10 digits
        intensities_list.append(intensity_sum)

    #select smallest value and return as output
    return intensities_list.index(min(intensities_list))

def validation(size):
    return 1
