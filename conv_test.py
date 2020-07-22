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
    signal_a = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    signal_b = 2 * signal_a - np.sqrt(signal_a)
    signal_c = signal_b - signal_a

    result_a = np.convolve(signal_a, kernel_1d)
    result_b = np.convolve(signal_b, kernel_1d)

    m_result_c = result_b - result_a
    result_c = np.convolve(signal_c, kernel_1d)
    print()

