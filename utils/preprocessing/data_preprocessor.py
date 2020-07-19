import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DataPreprocessor:
    def __init__(self, num_channels=8, num_imfs=4):
        self.input_columns = [f'channel_{c}_imf_{imf}' for imf in range(num_imfs) for c in range(num_channels)]
        self.output_column = 'force'
        self.imf_cols_dict = {f'imf_{imf}': [f'channel_{c}_imf_{imf}' for c in range(num_channels)] for
                              imf in range(num_imfs)}
        self.channel_cols_dict = {
            f'channel_{c}': [f'channel_{c}_imf_{imf}' for imf in range(num_imfs)] for c in range(num_channels)}

    @classmethod
    def multi_rms(cls, signals, window_size=400, axis=0):
        rms_signals = np.apply_along_axis(lambda m: cls.window_rms(m, window_size=window_size), axis=axis, arr=signals)
        return rms_signals

    @classmethod
    def window_rms(cls, signal, window_size=400):
        signal_squared = np.power(signal, 2)
        window = np.ones(window_size) / float(window_size)
        rms = np.sqrt(np.convolve(signal_squared, window, 'same'))
        return rms

    def set_scaler(self, df):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(df[self.output_column].values.reshape(-1, 1))
        return scaler

    def process_data(self, annotations):
        annotations = annotations.fillna(0)
        annotations = annotations.set_index(['subject', 'signal_num'])
        data_cols = self.input_columns + [self.output_column]
        for subject, signal in annotations.index.unique():
            annotations.loc[(subject, signal), data_cols] = \
                self.multi_rms(annotations.loc[(subject, signal), data_cols].values)
        self.scaler = self.set_scaler(annotations)
        return annotations