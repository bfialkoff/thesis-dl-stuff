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
from scipy.ndimage import convolve
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


class EmgImageGenerator:
    def __init__(self, annotations, batch_size, num_channels=8, scaler=None, num_imfs=4, input_size=None, is_debug=False):
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
        self.num_channels = num_channels
        self.num_imfs = num_imfs
        self.input_shape = (self.num_channels, self.num_channels, self.num_imfs)
        self.output_column = 'force'
        self.scaler = scaler
        self.input_images, self.outputs = self.process_annotations(annotations)
        self.batch_size = batch_size
        self.num_samples = len(self.outputs)
        self.steps = self.num_samples // self.batch_size
        self.index_list = list(range(self.num_samples))
        self.corrupter = self.get_corrupter()
        shuffle(self.index_list)

    def process_annotations(self, raw_annotations):
        kernel_length = 201
        #rms_kernel_3d = (np.ones((kernel_length, self.num_channels, self.num_channels))/(self.num_channels * self.num_channels)) / kernel_length
        rms_kernel_3d = np.ones(kernel_length) / kernel_length
        after_rms_size = len(raw_annotations) + kernel_length - 1
        batch_images, batch_outputs = np.zeros((after_rms_size, *self.input_shape)), np.zeros(after_rms_size)
        batch_outputs = np.convolve(raw_annotations[self.output_column], rms_kernel_3d)
        if self.scaler:
            batch_outputs = self.scaler.transform(batch_outputs.reshape(-1, 1)).reshape(-1)

        for i, (imf, channel_cols) in enumerate(self.imf_cols_dict.items()):
            inputs = raw_annotations[channel_cols].values
            # this row makes the voltage difference proportional to the channel (ai-aj) * ai
            # input_images = input_images * np.expand_dims(batch_rows[channel_cols], 2)
            input_images = permute_axes_subtract(inputs)
            for r in range(self.num_channels):
                for c in range(self.num_channels):
                    if r != c: # on the diagonal the entire value is zero, no need to waste computation
                        curr_signal = input_images[:, r, c]
                        curr_res = np.convolve(curr_signal, rms_kernel_3d)
                        batch_images[:, r, c, i] = curr_res
                    else:
                        batch_images[:, r, c, i] = 0


        return batch_images, batch_outputs

    def get_corrupter(self):
        distortion_augs = OneOf([OpticalDistortion(p=1), GridDistortion(p=1)], p=1)
        effects_augs = OneOf([IAASharpen(p=1), IAAEmboss(p=1),
                              IAAPiecewiseAffine(p=1), IAAPerspective(p=1),
                              CLAHE(p=1)], p=1)
        misc_augs = OneOf([ShiftScaleRotate(p=1), HueSaturationValue(p=1), RandomBrightnessContrast(p=1)], p=1)
        blur_augs = OneOf([Blur(p=1), MotionBlur(p=1), MedianBlur(p=1), GaussNoise(p=1)], p=1)
        aug = Compose([distortion_augs, effects_augs, misc_augs, blur_augs])
        return aug


    def corrupt(self, batch_images):
        for i, img in enumerate(batch_images):
            t_img = (255 * (img + 1) / 2).astype(np.uint8)
            t_img = self.corrupter(image=t_img)['image']
            t_img = t_img / 127.5 - 1
            batch_images[i] = t_img
        return batch_images

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

    def get_input_outputs(self, batch_rows, debug_tag=''):
        batch_images, batch_outputs = np.zeros((len(batch_rows), *self.input_shape)), batch_rows[
            self.output_column].values
        for i, (imf, channel_cols) in enumerate(self.imf_cols_dict.items()):
            inputs = batch_rows[channel_cols].values
            input_images = permute_axes_subtract(inputs)

            # todo experimenting with t-diemnsional rms here

            # this row makes the voltage difference proportional to the channel (ai-aj) * ai
            # input_images = input_images * np.expand_dims(batch_rows[channel_cols], 2)
            input_images = input_images.reshape(-1, self.num_channels, self.num_channels)
            batch_images[:, :, :, i] = input_images
        print('done')
        if self.input_size:
            self.resize_batch(batch_images)
        if self.is_debug:
            self.save_image(batch_images, debug_tag)
        return batch_images, batch_outputs,

    def save_image(self, images, debug_tag=''):
        images = images[:,:,:, 0:3]
        debug_dir = Path(__file__).joinpath('..', 'files', 'deep_learning', 'debug').resolve()
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

    def get_corrupted_inputs_ouputs(self, batch_rows):
        input_images, _ = self.get_input_outputs(batch_rows, debug_tag='corrupt_')
        corrupted_images = self.corrupt(input_images)
        return corrupted_images, _

    def corruption_train_generator(self):
        counter = 0
        while True:
            if counter == self.steps:
                shuffle(self.index_list)
                counter = 0
            num_corrupted = self.batch_size // 2
            num_uncorrupted = self.batch_size - num_corrupted
            corrupted_batch_indices = sample(self.index_list, num_corrupted)
            uncorrupted_batch_indices = sample(self.index_list, num_uncorrupted)
            corrupted_batch_rows = self.annotations.iloc[corrupted_batch_indices]
            uncorrupted_batch_rows = self.annotations.iloc[uncorrupted_batch_indices]
            corrupted_input_images, _ = self.get_corrupted_inputs_ouputs(corrupted_batch_rows)
            uncorrupted_input_images, _ = self.get_input_outputs(uncorrupted_batch_rows)
            input_images = np.concatenate((corrupted_input_images, uncorrupted_input_images))
            outputs = np.array(num_corrupted * [1] + num_uncorrupted * [0])
            input_images, outputs = sk_shuffle(input_images, outputs)
            yield input_images, outputs

    def corruption_val_generator(self):
        # todo ensure that this yields one subject at a time, no shuffle, ensure all data is yielded
        batch_size = self.batch_size // 2
        counter = 0
        remainder = self.num_samples % batch_size or batch_size
        while True:
            if counter == self.steps:
                start_i = counter * batch_size
                end_i = start_i + remainder
                counter = 0
            else:
                start_i = counter * batch_size
                end_i = start_i + batch_size
            counter += 1
            uncorrupted_batch_rows = self.annotations.iloc[start_i:end_i]
            corrupted_batch_rows = self.annotations.iloc[start_i:end_i]
            uncorrupted_input_images, _ = self.get_input_outputs(uncorrupted_batch_rows)
            corrupted_input_images, _ = self.get_corrupted_inputs_ouputs(corrupted_batch_rows)
            input_images = np.concatenate((corrupted_input_images, uncorrupted_input_images))
            num_corrupted = len(corrupted_input_images)
            num_uncorrupted = len(uncorrupted_input_images)
            outputs = np.array(num_corrupted * [1] + num_uncorrupted * [0])
            yield input_images, outputs

    def train_generator(self):
        counter = 0
        while True:
            if counter == self.steps:
                shuffle(self.index_list)
                counter = 0
            batch_indices = sample(self.index_list, self.batch_size)
            input_images, outputs = self.input_images[batch_indices], self.outputs[batch_indices]
            yield input_images, outputs

    def val_generator(self):
        # todo ensure that this yields one subject at a time, no shuffle, ensure all data is yielded
        counter = 0
        remainder = self.num_samples % self.batch_size or self.batch_size
        while True:
            if counter == self.steps:
                start_i = counter * self.batch_size
                end_i = start_i + remainder
                counter = 0
            else:
                start_i = counter * self.batch_size
                end_i = start_i + self.batch_size
            counter += 1
            input_images, outputs = self.input_images[start_i:end_i], self.outputs[start_i:end_i]
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
    train_annotations = pd.read_csv(train_path)
    print(train_annotations['subject'].unique())
    val_annotations = pd.read_csv(val_path)
    print(val_annotations['subject'].unique())
    train_emg_gen = EmgImageGenerator(train_annotations, 16, num_imfs=4, is_debug=True)
    val_emg_gen = EmgImageGenerator(val_annotations, 16, num_imfs=4, is_debug=True)
    for i, d in train_emg_gen.train_generator():
        print(i)
