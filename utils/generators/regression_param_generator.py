from random import shuffle, sample

import cv2
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class EmgImageGenerator:
    def __init__(self, csv_features_path, csv_labels_path, batch_size, scaler = None, feature_len=30, num_params=9, num_channels=8, resize=None, mirror=False, is_debug=False):
        self.feature_len = feature_len
        self.num_params = num_params
        self.mirror = mirror
        self.input_shape = (feature_len, num_channels)
        self.is_debug = is_debug
        (self.resize, self.cv2_resize), (self.mirror_shape, self.cv2_mirror_shape) = self.set_resize_params(resize)
        self.input_columns = [f'channel_{c}' for c in range(num_channels)]
        self.output_column = 'regression_params'
        self.features = pd.read_csv(csv_features_path)
        self.labels = pd.read_csv(csv_labels_path)
        self.features = self.features.drop(columns=['subject', 'signal_num'], axis=1)
        self.labels = self.labels.drop(columns=['subject', 'signal_num'], axis=1)
        self.scaler = scaler

        if self.scaler is None:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.scaler.fit(self.labels[self.output_column].values.reshape(-1, 1))

        self.sample_map = {v: i for i,v in enumerate(self.features['sample_num'].unique())}
        self.features['sample_num'] = self.features['sample_num'].map(self.sample_map)
        self.labels['sample_num'] = self.labels['sample_num'].map(self.sample_map)
        self.batch_size = batch_size
        self.index_list = self.get_index_list()
        self.num_samples = len(self.index_list)
        self.steps = self.num_samples // self.batch_size
        self.features = self.features.set_index('sample_num')
        self.labels = self.labels.set_index('sample_num')

    def set_resize_params(self, resize):
        cv2_new_shape = None
        cv2_mirror_shape = None
        new_shape = None
        mirror_shape = None
        if resize:
            (short_side, short_axis) = np.min(self.input_shape), np.argmin(self.input_shape)
            (long_side, long_axis) = np.max(self.input_shape), np.argmax(self.input_shape)
            self.ratio = long_side / short_side
            new_short_side = resize
            new_long_side = int(new_short_side * self.ratio)
            new_shape = np.zeros(2, dtype=int)
            new_shape[short_axis] = new_short_side
            new_shape[long_axis] = new_long_side
            if self.mirror:
                self.mirror_factor = int(np.ceil(self.ratio))
                mirror_shape = new_shape.copy()
                mirror_shape[short_axis] = new_short_side * self.mirror_factor
                mirror_shape = mirror_shape.tolist()
                cv2_mirror_shape = tuple(reversed(mirror_shape))

            # for some reason when using cv2.resize it expects the reverse order
            new_shape = new_shape.tolist()
            cv2_new_shape = tuple(reversed(new_shape))
        return (new_shape, cv2_new_shape), (mirror_shape, cv2_mirror_shape)

    def get_index_list(self):
        index_list = list(self.sample_map.values())
        remainder = len(index_list) % self.batch_size
        if remainder:
            missing_samples = self.batch_size - remainder
            samples_to_duplicate = sample(index_list, missing_samples)
            index_list += samples_to_duplicate
        shuffle(index_list)
        return index_list

    def scale(self, img):
        # normalize
        x = (img - img.min()) / (img.max() - img.min())

        # scale to -1, 1
        x = 2 * x - 1
        return x
    def resize_batch(self, batch_images):
        if self.mirror:
            resized_batch = np.zeros((len(batch_images), *self.mirror_shape, 3), dtype=batch_images.dtype)
        else:
            resized_batch = np.zeros((len(batch_images), *self.resize), dtype=batch_images.dtype)
        for i, img in enumerate(batch_images):
            resized = cv2.resize(img, self.cv2_resize, cv2.INTER_NEAREST)
            if self.mirror:
                resized = np.concatenate(self.mirror_factor * (resized, ), axis=1)
            resized = np.repeat(resized[:, :, np.newaxis], 3, axis=2)
            # scale to +- 1
            resized = self.scale(resized)
            resized_batch[i] = resized
        return resized_batch


    def get_input_outputs(self, batch_feature_rows, batch_label_rows, debug_tag=''):
        inputs, outputs = batch_feature_rows[self.input_columns].values, batch_label_rows[self.output_column].values
        inputs = inputs.reshape(-1, *self.input_shape)
        if self.resize:
            inputs = self.resize_batch(inputs)
        if self.scaler and False:
            outputs = self.scaler.transform(outputs.reshape(-1, 1)).reshape(-1)
        outputs = outputs.reshape(-1, self.num_params)
        return inputs, outputs



    def train_generator(self):
        counter = 0
        while True:
            if counter == self.steps:
                self.index_list = self.get_index_list()
                counter = 0
            batch_indices = sample(self.index_list, self.batch_size)
            batch_feature_rows = self.features.loc[batch_indices]
            batch_label_rows = self.labels.loc[batch_indices]
            input_images, outputs = self.get_input_outputs(batch_feature_rows, batch_label_rows)
            yield input_images, outputs

    def val_generator(self):
        # todo ensure that this yields one subject at a time, no shuffle, ensure all data is yielded
        counter = 0
        remainder = self.num_samples % self.batch_size or self.batch_size
        while True:
            if counter == self.steps - 1:
                start_i = counter * self.batch_size
                end_i = start_i + remainder
                counter = 0
            else:
                start_i = counter * self.batch_size
                end_i = start_i + self.batch_size
            counter += 1
            batch_feature_rows = self.features.loc[sorted(self.index_list)[start_i:end_i]]
            batch_label_rows = self.labels.loc[sorted(self.index_list)[start_i:end_i]]
            input_images, outputs = self.get_input_outputs(batch_feature_rows, batch_label_rows)
            yield input_images, outputs




if __name__ == '__main__':
    from pathlib import Path

    train_features_path = Path(__file__, '..', '..', '..', 'files', 'omp_generated_vanilla_annotations',
                               'val_features.csv').resolve()
    train_labels_path = Path(__file__, '..', '..', '..', 'files', 'omp_generated_vanilla_annotations', 'val_labels.csv').resolve()
    emg_gen = EmgImageGenerator(train_features_path, train_labels_path, 15, resize=80, mirror=True)

    for i, d in emg_gen.train_generator():
        print(d)
