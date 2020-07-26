"""
this generator is under construction for the new stratgies
1) it will work with the IMF signal returning an 8x8x4 image
2) the debug will show the first 3 channels
3) in the __init__ function the signals will be rmsed

"""
from pathlib import Path
from random import shuffle, sample

import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from scipy.signal import resample as sc_resample
import pandas as pd
from sklearn.utils import shuffle as sk_shuffle
from albumentations import (
    IAAPerspective, CLAHE, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, OneOf, Compose
)

def permute_axes_subtract(arr, axis=1):
    """
            calculates all the differences between all combinations
            terms in the input array. output[i,j] = arr[i] - arr[j]
            for every combination if ij.

            Parameters
            ----------
            arr numpy.array
                a 1d input array

            Returns
            -------
            numpy.array
                a 2d array

            Examples
            --------
            arr = [10, 11, 12]

            diffs = [[ 0 -1 -2]
                    [ 1  0 -1]
                    [ 2  1  0]]
            """
    s = arr.shape
    if arr.ndim == 1:
        axis = 0

    # Get broadcastable shapes by introducing singleton dimensions
    s1 = np.insert(s, axis + 1, 1)
    s2 = np.insert(s, axis, 1)

    # Perform subtraction after reshaping input array to
    # broadcastable ones against each other
    return arr.reshape(s1) - arr.reshape(s2)

def multi_rms(signals, window_size=400, axis=0):
    rms_signals = np.apply_along_axis(lambda m: window_rms(m, window_size=window_size), axis=axis, arr=signals)
    return rms_signals

def window_rms(signal, window_size=400):
    signal_squared = np.power(signal, 2)
    window = np.ones(window_size) / float(window_size)
    rms = np.sqrt(np.convolve(signal_squared, window, 'same'))
    return rms

def resample(signal):
    nearest_power_2 = int(np.ceil(np.log2(len(signal))))
    resample_to = 2 ** nearest_power_2
    residual = resample_to - len(signal)
    resampled = sc_resample(signal, resample_to)
    return resampled

def multi_resample(signals, axis=0):
    rms_signals = np.apply_along_axis(lambda m: resample(m), axis=axis, arr=signals)
    return rms_signals


class EmgImageGenerator:
    def __init__(self, annotations, batch_size, num_channels=8, scaler=None, num_imfs=4, signal_window_size=256, input_size=None, is_debug=False):
        """

        :param csv_path: path to the annotations file
        :param batch_size: number of data points to yield per itereation
        :param num_channels:
            the number of EMG channels that comprise a single sample. this is defined in the annotations file
            and shouldnt be adjusted
        :param num_imfs:
            the number of IMFs chosen during the EMD preprocessing stage this is defined in the annotations file
            and shouldnt be adjusted
        :param input_size:
            leave unspecified unless there is a reason to require the base 8x8xnum_imfs to be updated
        :param is_debug:
            will save images into a debug directory. Should be False during training!
        """
        self.input_size = input_size
        self.is_debug = is_debug
        self.imf_cols_dict = {f'imf_{imf}': [f'channel_{c}_imf_{imf}' for c in range(num_channels)] for
                              imf in range(num_imfs)}
        self.id_cols = ['subject', 'signal_num']
        self.input_columns = [f'channel_{c}_imf_{imf}' for imf in range(num_imfs) for c in range(num_channels)]
        self.output_column = 'force'
        self.value_cols = self.input_columns + [self.output_column]
        self.num_channels = num_channels
        self.num_imfs = num_imfs
        self.signal_window_size = signal_window_size
        self.scaler = scaler
        self.annotations = self.process_annotations(annotations)
        self.batch_size = batch_size
        self.num_samples = int(self.annotations.index.max() +1)
        self.steps = self.num_samples // self.batch_size
        self.input_shape = (self.batch_size, self.signal_window_size, self.num_channels, self.num_channels, self.num_imfs)
        self.index_list = list(range(self.num_samples))
        shuffle(self.index_list)

    def process_annotations(self, raw_annotations):
        # fixme instead of resampling handle the residual window separately
        #  the solution for the final window will be to incorporate the missing portion from the preceding window
        annotations = raw_annotations.fillna(0)
        annotations = annotations.drop(columns=['is_valid'], axis=1)
        annotations = annotations.set_index(self.id_cols)
        data_cols = self.id_cols + ['segment_num'] + self.value_cols
        results = []
        start_seg_id = 0
        for subject, signal in annotations.index.unique():
            rmsed = multi_rms(annotations.loc[(subject, signal), self.value_cols].values)
            # calculate remainder and dst size

            residual = len(rmsed) % self.signal_window_size
            num_windows = len(rmsed) // self.signal_window_size
            whole_part = rmsed[:num_windows * self.signal_window_size]

            if residual:
                num_windows += 1
                prepend = rmsed[-self.signal_window_size:]
                whole_part = np.concatenate((whole_part, prepend))
            res_len = self.signal_window_size * num_windows
            res = np.zeros((res_len, len(data_cols)))
            segments = start_seg_id + np.arange(0, len(res)) // self.signal_window_size
            res[:, 0:2] = subject, signal
            res[:, 2] = segments
            res[:, 3:] = whole_part
            results.append(res)
            start_seg_id = segments.max() + 1

        result_array = np.vstack(results)
        annotations = pd.DataFrame(data=result_array, columns=data_cols)
        annotations['segment_num'] = annotations['segment_num'].astype(int)


        if self.scaler is None:
            force_labels = annotations[['segment_num', self.output_column]].groupby('segment_num').apply(np.mean)[
                self.output_column].values
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.scaler.fit(force_labels.reshape(-1, 1))

        annotations = annotations.set_index('segment_num', drop=True)
        return annotations


    def transform_func(self, batch_images):
        batch_maxes = batch_images.max(axis=(1, 2, 3)).reshape(-1, 1, 1, 1)
        normalized_batch = batch_images / batch_maxes
        return normalized_batch

    def resize_batch(self, batch_images):
        dst = np.array((len(batch_images), self.input_size, self.input_size, 3), dtype=batch_images.dtype)
        for i, img in enumerate(batch_images):
            resized = cv2.resize(img, (self.input_size, self.input_size), cv2.INTER_NEAREST)
            dst[i] = resized
        return dst

    def save_image(self, images, debug_tag=''):
        images = images[:,:,:, 0:3]
        debug_dir = Path(__file__).joinpath('..', '..', '..', 'files', 'deep_learning', 'debug').resolve()
        if not debug_dir.exists():
            debug_dir.mkdir(parents=True)
        last_i = len(list(debug_dir.glob('*'))) + 1
        for i, img in enumerate(images):
            min_value = img.min()
            max_value = img.max() - min_value
            img = (img - min_value) / max_value
            img = (255 * img).astype(np.uint8)
            print(last_i + i, img.min(), img.max())
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
            f_name = str(debug_dir.joinpath(f'{debug_tag}{last_i + i}.jpg').resolve())
            cv2.imwrite(f_name, img)

    def get_input_outputs(self, batch_rows, debug_tag=''):
        batch_images = np.zeros(self.input_shape)
        batch_outputs = batch_rows[self.output_column].reset_index().groupby('segment_num').mean().values.reshape(-1)
        for i, (imf, channel_cols) in enumerate(self.imf_cols_dict.items()):
            inputs = batch_rows[channel_cols].values
            input_images = permute_axes_subtract(inputs)

            # this row makes the voltage difference proportional to the channel (ai-aj) * ai
            # input_images = input_images * np.expand_dims(batch_rows[channel_cols], 2)
            input_images = input_images.reshape(self.input_shape[:4])
            batch_images[:,:, :, :, i] = input_images
        if self.input_size:
            self.resize_batch(batch_images)
        if self.is_debug:
            self.save_image(batch_images, debug_tag)
        if self.scaler:
            batch_outputs = self.scaler.transform(batch_outputs.reshape(-1, 1)).reshape(-1)
        return batch_images, batch_outputs

    def train_generator(self):
        counter = 0
        while True:
            if counter == self.steps:
                shuffle(self.index_list)
                counter = 0
            batch_indices = sample(self.index_list, self.batch_size)
            batch_rows = self.annotations.loc[batch_indices]
            input_images, outputs = self.get_input_outputs(batch_rows)
            if self.is_debug:
                self.save_image(input_images, debug_tag='')
            yield input_images, outputs

    def val_generator(self):
        # fixme, iterate over the index properly, something is broken it seems like the -1 in end_i fixes it
        #  but need to verify that counter stops when we want it to
        counter = 0
        remainder = self.num_samples % self.batch_size or self.batch_size
        while True:
            if counter == self.steps - 1:
                start_i = counter * self.batch_size
                end_i = start_i + remainder - 1
                counter = 0
            else:
                start_i = counter * self.batch_size
                end_i = start_i + self.batch_size - 1
            counter += 1
            batch_rows = self.annotations.loc[start_i:end_i]
            input_images, outputs = self.get_input_outputs(batch_rows)
            yield input_images, outputs


def show_2_images(img1, img2):
    import matplotlib.pyplot as plt
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img1)
    ax2.imshow(img2)
    plt.show()


if __name__ == '__main__':
    from pathlib import Path

    train_path = Path(__file__, '..', '..', '..', 'files', 'emd_dl_train_annotations.csv').resolve()
    val_path = Path(__file__, '..', '..', '..', 'files', 'emd_dl_val_annotations.csv').resolve()
    #train_annotations = pd.read_csv(train_path)
    #print(train_annotations['subject'].unique())
    val_annotations = pd.read_csv(val_path)
    print(val_annotations['subject'].unique())
    #train_emg_gen = EmgImageGenerator(train_annotations, 16, num_imfs=4, is_debug=False)
    val_emg_gen = EmgImageGenerator(val_annotations, 16, num_imfs=4, is_debug=False)
    for i, d in enumerate(val_emg_gen.train_generator()):
        print(i)

