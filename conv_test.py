import numpy as np
from scipy.ndimage import convolve
if __name__ == '__main__':
    kernel_length = 3
    h, w = 8, 8
    kernel_1d = np.ones(kernel_length, dtype=np.float64) / kernel_length
    kernel_3d = (np.ones((kernel_length, h, w))/(h * w)) / kernel_length
    signal = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    nd_signal = np.tile(signal, h * w).reshape((h, w, w)).T
    result_1d = convolve(signal, kernel_1d, mode='nearest')
    result_3d = convolve(nd_signal, kernel_3d, mode='nearest')
