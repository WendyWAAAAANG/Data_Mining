'''
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label
'''

# import necessary packages.
import numpy as np
import pandas as pd
import operator
from os import listdir

# define k-NN function.
# inx -- vector will be classfied.
# dataSet -- training data will be used to form model.
# labels -- classification of data.
# k -- number of nearest data we desire.
def KNN(inX, data, labels, k):
    # store the size of data.
    data_size = data.shape[0]
    # calculate distance,
    # and here using Euclidean distance.
    diffMat = np.tile(inX, (data_size, 1)) - data
    sqDiffMat = diffMat ** 2
    distances = (sqDiffMat.sum(axis = 1)) ** 0.5
    # sort training items,
    # according to Euclidean distance calculated above,
    distances_sorted = distances.argsort()
    # declare class_count to store the classes of k items.
    class_count = {}
    # select k items that nearest to inX.     
    for i in range(k):
        voteIlabel = labels[distances_sorted[i]]
        class_count[voteIlabel] = class_count.get(voteIlabel, 0) + 1
    # sort the class of k nearest items.
    sorted_class_count = sorted(class_count.items(), key = operator.itemgetter(1), reverse=True)
    # return first class, the most frequent class as result.
    return sorted_class_count[0][0]

