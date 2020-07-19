import time

import numpy as np
from scipy.ndimage import convolve

def manual_conv(x):
    for h_ in range(h):
        for w_ in range(w):
            curr_signal = nd_signal[:, h_, w_]
            curr_res = convolve(curr_signal, kernel_1d, mode='nearest')
            new_res_3d[:, h_, w_] = curr_res
    return new_res_3d

if __name__ == '__main__':
    kernel_length = 3
    h, w = 8, 8

    kernel_1d = np.ones(kernel_length, dtype=np.float64) / kernel_length
    kernel_3d = (np.ones((kernel_length, h, w))/(h * w)) / kernel_length
    signal = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    #nd_signal = np.tile(signal, h * w).reshape((h, w, w)).T
    nd_signal = np.random.random((100000, 10, 10))
    result_1d = convolve(signal, kernel_1d, mode='nearest')
    result_3d = convolve(nd_signal, kernel_3d, mode='nearest')
    new_res_3d = np.zeros(result_3d.shape)

    scipy_time = []
    manual_time = []
    repeats = 10
    for i in range(repeats):
        s = time.time()
        manual_conv(nd_signal)
        e = time.time()
        manual_time.append(e - s)

        s = time.time()
        convolve(nd_signal, kernel_3d, mode='nearest')
        e = time.time()
        scipy_time.append(e - s)

    print('scipy: ', sum(scipy_time) / repeats)
    print('manual: ', sum(manual_time) / repeats)