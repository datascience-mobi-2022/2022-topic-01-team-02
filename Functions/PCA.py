import numpy as np

def PCA(X_no_label, num_components):
    """
    :param X_no_label: training dataset (2D-Array; no label)
    :param num_components): number of components (integer)
    """
    #z-transformation
    std0 = []
    for i in range(0, X_no_label.shape[1]):
        if np.std(X_no_label[:, i]) == 0:
            std0.append(i)
    X_cleaned =  np.delete(X_no_label, std0, 1)

    X_z = (X_cleaned - np.mean(X_cleaned, axis = 0))/np.std(X_cleaned, axis = 0)

    #variance
    cov_arr = np.cov(X_z, rowvar = False)

    #eigenvalues, eigenvectors
    eigen_val, eigen_vec = np.linalg.eigh(cov_arr)

    #sorting
    index_sorted = np.argsort(eigen_val)[::-1]
    sorted_eigenval = eigen_val[index_sorted]
    sorted_eigenvec = eigen_vec[:,index_sorted]

    #selecting subset
    eigenvec_subset = sorted_eigenvec[:, 0:num_components]

    #dimension reduction
    X_reduced = np.dot(eigenvec_subset.transpose(), X_z.transpose()).transpose()

    return X_reduced