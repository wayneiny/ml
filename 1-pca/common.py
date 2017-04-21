import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.misc as misc
import tensorflow as tf
from tensorflow.contrib.layers import *

################################################################################
# 1. Define constants and helper functions

folder_path = 'att_faces/s'

faces_list = []
faces_matrix = None
NUM_OF_FEATURES = 10304

def encode(x, w):
    '''
    Return encoded matrix k*m
    x: input matrix 10304*m
    w: encoding matrix 10304*k
    '''
    return tf.matmul(x, w)


def decode(x, w):
    '''
    Return decoded matrix 10304*m
    x: input matrix 10304*m
    w: encoding and decoding matrix 10304*k
    '''
    return tf.matmul(encode(x, w), tf.transpose(w))

################################################################################
# 2. Read in images

for i in range(1, 41):
    path = folder_path + str(i) + '/'
    for image in os.listdir(path):
        tmp = misc.imread(path + image)
        faces_list.append(tmp.flatten())


# faces_matrix is a 10304 (number of pixels) * 400 (number of examples) matrix
faces_matrix = np.array(faces_list).T
faces_mean_single = np.mean(faces_matrix, 1).reshape((NUM_OF_FEATURES, 1))
faces_mean = np.tile(faces_mean_single, (400, ))
centered_faces_matrix = (faces_matrix - faces_mean)
    