"""
this scriipt should run after running build_dataset,py
its purpose will be to ratify the CEEMDAN decomposition
therefore it will plot the 4 IMFS along with their respective spectrums
and save as pngs which will be manually reviewed offline

"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample

def get_shifted_fft_and_frequency(sampling_frequency, signal):
    dft = np.fft.fft(signal)
    abs_dft = np.abs(dft)
    len_dft = len(dft)
    discrete_frequency = np.arange(0, len_dft)
    discrete_frequency[round(len_dft / 2):] = discrete_frequency[round(len_dft / 2):] - len_dft
    time_frequency = (sampling_frequency / len_dft) * discrete_frequency
    return time_frequency, dft, abs_dft

def rms(signal, kernel_length=201):
    kernel = np.ones(kernel_length) / kernel_length
    rmsed = np.convolve(signal, kernel)
    return rmsed

def my_resample(signal):
    nearest_power_2 = int(np.ceil(np.log2(len(signal))))
    resample_to = 2 ** nearest_power_2
    resampled = resample(signal, resample_to)
    return resampled

if __name__ == '__main__':
    emd_csv_path = Path(__file__, '..', 'files', 'emd_annotations.csv')
    graph_path = Path(__file__).joinpath('..', 'files', 'validate', 'resample')
    df = pd.read_csv(emd_csv_path)
    num_subjects_gen = df['subject'].unique().tolist()
    num_signals_gen = df['signal_num'].unique().tolist()
    num_channels_gen= range(8)
    df = df.set_index(['subject', 'signal_num'])
    channel_cols_dict = {
        f'channel_{c}': [f'channel_{c}_imf_{imf}' for imf in range(4)] for c in num_channels_gen
    }

    for subject in tqdm(num_subjects_gen):
        for signal in num_signals_gen:
            subject_df = df.loc[subject, signal]
            for channel in num_channels_gen:
                path = graph_path.joinpath(f'subject_{subject}', f'channel_{channel}').resolve()
                if not path.exists():
                    path.mkdir(parents=True)
                cols = channel_cols_dict[f'channel_{channel}']
                subject_path = path.joinpath(f'signal_{signal}.png').resolve()
                resampled_subject_path = path.joinpath(f'resampled_signal_{signal}.png').resolve()
                if resampled_subject_path.exists() and subject_path.exists():
                    continue
                f, ax = plt.subplots(4, 2)
                f1, ax1 = plt.subplots(4, 2)
                for i in range(4):
                    imf = subject_df[cols[i]].values
                    rms_imf = rms(imf)
                    resampled = my_resample(rms_imf)
                    time_frequency, _, abs_dft = get_shifted_fft_and_frequency(1980, rms_imf)
                    time_frequency_rms, _, abs_dft_rms = get_shifted_fft_and_frequency(1980, resampled)

                    ax[i, 0].plot(rms_imf)
                    ax[i, 1].plot(time_frequency, abs_dft)
                    plt.savefig(subject_path)

                    ax1[i,0].plot(resampled)
                    ax1[i, 1].plot(time_frequency_rms, abs_dft_rms)
                    plt.savefig(resampled_subject_path)
                    plt.close()
