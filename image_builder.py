from pathlib import Path
from random import shuffle, sample

import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
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
    s1 = np.insert(s, axis, 1)
    s2 = np.insert(s, axis + 1, 1)

    # Perform subtraction after reshaping input array to
    # broadcastable ones against each other
    return arr.reshape(s1) - arr.reshape(s2)

class EmgImageGenerator:
    def __init__(self, csv_path, batch_size, is_debug=False):
        self.is_debug = is_debug
        self.input_columns = [f'channel_{c}' for c in range(8)]
        self.output_column = 'force'
        annotation = pd.read_csv(csv_path)
        self.annotation = self.scale_rows(annotation)
        self.batch_size = batch_size
        self.num_samples = len(annotation)
        self.steps = self.num_samples // self.batch_size
        self.index_list = list(range(self.num_samples))
        self.corrupter = self.get_corrupter()
        shuffle(self.index_list)

    def get_corrupter(self):
        distortion_augs = OneOf([OpticalDistortion(p=1), GridDistortion(p=1)], p=1)
        effects_augs = OneOf([IAASharpen(p=1), IAAEmboss(p=1),
                              IAAPiecewiseAffine(p=1), IAAPerspective(p=1),
                              CLAHE(p=1)], p=1)
        misc_augs = OneOf([ShiftScaleRotate(p=1), HueSaturationValue(p=1), RandomBrightnessContrast(p=1)], p=1)
        blur_augs = OneOf([Blur(p=1), MotionBlur(p=1), MedianBlur(p=1), GaussNoise(p=1)], p=1)
        aug = Compose([distortion_augs, effects_augs, misc_augs, blur_augs])
        return aug

    def scale_rows(self, df):
        self.min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.min_max_scaler.fit(df[self.output_column].values.reshape(-1, 1))  # transforms the outputs to a normal distribution
        #self.max_abs_scaler = MaxAbsScaler()
        df[self.output_column] = self.min_max_scaler.transform(df[self.output_column].values.reshape(-1, 1)) # transforms the outputs to a normal distribution
        #df[self.output_column] = self.max_abs_scaler.fit_transform(df[self.output_column].values.reshape(-1, 1))  # transforms the outputs to a normal distribution
        # df[self.input_columns] = self.gauss_scaler.fit_transform(df[self.input_columns])
        return df

    def corrupt(self, batch_images):
        for i, img in enumerate(batch_images):
            t_img = (255 * (img + 1) / 2).astype(np.uint8)
            t_img = self.corrupter(image=t_img)['image']
            t_img = t_img / 127.5 - 1
            batch_images[i] = t_img
        return batch_images

    def transform_func(self, batch_images):
        batch_maxes = batch_images.max(axis=(1, 2)).reshape(-1, 1, 1)
        normalized_batch = batch_images / batch_maxes
        return normalized_batch

    def get_input_outputs(self, batch_rows, debug_tag=''):
        inputs, outputs = batch_rows[self.input_columns].values, batch_rows[self.output_column].values
        input_images = permute_axes_subtract(inputs)
        # input_images = self.max_abs_scaler.fit_transform(input_images.reshape(-1, self.batch_size))
        input_images = self.transform_func(input_images)
        input_images = input_images.reshape(-1, 8, 8)
        input_images = np.repeat(input_images[:, :, :, np.newaxis], 3, axis=3)
        if self.is_debug:
            self.save_image(input_images, debug_tag)
        return input_images, outputs

    def save_image(self, images, debug_tag=''):
        debug_dir = Path(__file__).joinpath('..', 'files', 'deep_learning', 'debug').resolve()
        if not debug_dir.exists():
            debug_dir.mkdir(parents=True)
        last_i = len(list(debug_dir.glob('*'))) + 1
        for i, img in enumerate(images):
            min_value = img.min()
            max_malue = img.max() - min_value
            img = (img - min_value) / max_malue
            img = (255 * img).astype(np.uint8)
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
            corrupted_batch_rows = self.annotation.iloc[corrupted_batch_indices]
            uncorrupted_batch_rows = self.annotation.iloc[uncorrupted_batch_indices]
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
            uncorrupted_batch_rows = self.annotation.iloc[start_i:end_i]
            corrupted_batch_rows = self.annotation.iloc[start_i:end_i]
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
            batch_rows = self.annotation.iloc[batch_indices]
            input_images, outputs = self.get_input_outputs(batch_rows)
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
            batch_rows = self.annotation.iloc[start_i:end_i]
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

    train_path = Path(__file__, '..', 'files', 'dl_train_annotations.csv')
    val_path = Path(__file__, '..', 'files', 'dl_val_annotations.csv')
    emg_gen = EmgImageGenerator(train_path, 16, is_debug=True)
    for i, d in emg_gen.train_generator():
        print(len(d))
