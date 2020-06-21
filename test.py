import os
import numpy as np
from mylib.densenet import get_compiled
import tensorflow as tf


batch_size = 32


def load_test():
    x_test = np.ones((117, 32, 32, 32))
    # 读取测试集
    i = 0
    path = "dataset/test"  # 待读取的文件夹
    path_list = os.listdir(path)
    path_list.sort()  # 对读取的路径进行排序
    for filename in path_list:
        tmp = np.load(os.path.join(path, filename))
        voxel = tmp['voxel']
        seg = tmp['seg']
        x_test[i] = (voxel * seg)[50 - 16:50 + 16, 50 - 16:50 + 16, 50 - 16:50 + 16]
        i = i + 1
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 32, 1)
    return x_test


def save_sub(input_list, output_path):
    col0 = np.loadtxt("sampleSubmission.csv", str, delimiter=",", skiprows=1, usecols=0)
    np.savetxt(output_path, np.column_stack((col0, input_list[:, 1])), delimiter=',', fmt='%s', header='ID,Predicted', comments='')
    print('File saved')


x_test = load_test()
# compile model
model = get_compiled()
for i in range(14, 15):
    model_path = 'tmp/weights0621.' + str(i) + '.h5'
    model.load_weights(model_path)
    res = model.predict(x_test, batch_size, verbose=1)
    save_file_path = 'Submission_0621_' + str(i) + '.csv'
    save_sub(res, save_file_path)


