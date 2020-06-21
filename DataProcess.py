import numpy as np
from tensorflow import keras
import csv
import random
from matplotlib import pyplot as plt


# data augment
def rotation(array, angle):
    '''using Euler angles method.
    @author: renchao
    @params:
        angle: 0: no rotation, 1: rotate 90 deg, 2: rotate 180 deg, 3: rotate 270 deg
    '''
    #
    X = np.rot90(array, angle[0], axes=(0, 1))  # rotate in X-axis
    Y = np.rot90(X, angle[1], axes=(0, 2))  # rotate in Y'-axis
    Z = np.rot90(Y, angle[2], axes=(1, 2))  # rotate in Z"-axis
    return Z


def reflection(array, axis):
    '''
    @author: renchao
    @params:
        axis: -1: no flip, 0: Z-axis, 1: Y-axis, 2: X-axis
    '''
    if axis != -1:
        ref = np.flip(array, axis)
    else:
        ref = np.copy(array)
    return ref


def augment(x_train, y_train):
    for i in range(0, 465):
        angle = (random.randint(0, 3), random.randint(0, 3), random.randint(0, 3))
        rotated_data = rotation(x_train[i], angle)
        axis = random.randint(-1, 2)
        flipped_data = reflection(x_train[i], axis)
        # for j in range(0, 32, 2):
        #     plt.figure()
        #     plt.subplot(1, 3, 1)
        #     plt.imshow(rotated_data[j])
        #     plt.subplot(1, 3, 2)
        #     plt.imshow(flipped_data[j])
        #     plt.subplot(1, 3, 3)
        #     plt.imshow(x_train[i][j])
        #     plt.show()
        x_train = np.append(x_train, np.expand_dims(rotated_data, axis=0), axis=0)
        x_train = np.append(x_train, np.expand_dims(flipped_data, axis=0), axis=0)
        y_train = np.append(y_train, np.expand_dims(y_train[i], axis=0), axis=0)
        y_train = np.append(y_train, np.expand_dims(y_train[i], axis=0), axis=0)
    return x_train, y_train


def load_label():
    path = 'dataset/train_val.csv'
    y_train = np.loadtxt(path, int, delimiter=",", skiprows=1, usecols=1)
    print('Labels loaded')
    return y_train


# load feature original=1->返回增强的数据
def load_data(aug=False):
    focus = 16
    x_train = np.ones((465, 2 * focus, 2 * focus, 2 * focus))
    j = 0
    for i in range(0, 584):
        a = 'dataset/train_val/candidate' + str(i) + '.npz'
        try:
            tmp = np.load(a)
            voxel = tmp['voxel']
            seg = tmp['seg']
            x_train[j] = (voxel * seg)[50 - focus:50 + focus, 50 - focus:50 + focus, 50 - focus:50 + focus]
            j = j + 1
        except:
            continue
    y_train = load_label()
    if aug is False:
        x_train = x_train.reshape(x_train.shape[0], 2 * focus, 2 * focus, 2 * focus, 1)
        y_train = keras.utils.to_categorical(y_train, 2)
        return x_train, y_train
    x_train, y_train = augment(x_train, y_train)
    x_train, y_train = mix_up(x_train, y_train)
    x_train = x_train.reshape(x_train.shape[0], 2 * focus, 2 * focus, 2 * focus, 1)
    y_train = keras.utils.to_categorical(y_train, 2)
    return x_train, y_train


def mix_up(x_train, y_train):
    t = 0.5
    x = np.ones((400, 32, 32, 32))
    y = np.ones(400)
    i = 0
    while i < 400:
        m = random.randint(0, 464)
        n = random.randint(0, 464)
        if y_train[m] == y_train[n]:
            x[i] = (t * x_train[m] + (1 - t) * x_train[n]).copy()
            y[i] = (t * y_train[m] + (1 - t) * y_train[n]).copy()
            i = i + 1
    # i = 0
    # for j in range(0, 32, 2):
    #     plt.figure()
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(x[i][j])
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(x_train[i][j])
    #     plt.show()
    final_x = np.append(x_train, x, axis=0)
    final_y = np.append(y_train, y)
    return final_x, final_y
