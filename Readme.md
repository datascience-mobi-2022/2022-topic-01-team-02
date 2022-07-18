# Implementation and evaluation of K-nearest neighbors (KNN) algorithm for handwritten digit recognition
### Data Analysis MoBi SoSe2022, Topic 01: Biomedical Image Analysis
### Tutor: Marie Becker
### Team 02: Lena Fleischhacker, Pia Röhrich, Hellen Röttgen, Benjamin Wehnert
#### July 2022

# Abstract
Digit recognition can be implemented using many different classification methods. The k-nearest neighbors (KNN) algorithm is well-known for its simplicity, however, it quickly reaches limitations when it comes to flexibility.

The main goal of the project was to write an algorithm that accurately recognizes handwritten digits from the MNIST dataset using the KNN method. Prior to the implementation of KNN, the data was z-transformed and dimensionality was reduced through Principal Component Analysis (PCA). 

In addition to digit recognition using PCA and KNN, average numbers were generated from the MNIST data set and used for digit recognition as well. Furthermore, the PCA was implemented using singular value decomposition (SVD) instead of eigenvector decomposition. Finally, the digit recognition algorithm was expanded for the recognition of self-written digits.

The KNN algorithm with previous PCA proved to be an accurate but inefficient digit recognition algorithm. It can also be used for the accurate recognition of self-written digits. However, the digit recognition based on just the average images proved to be surprisingly accurate despite its simplicity.