import mne
import scipy
import scipy.signal
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from numpy import matlib
import random

# Doing the spectrogram for the whole signal and then producing the windows of 5 epochs

plt.ion()

EPOCH_LENGTH = 4  # seconds
fs = 128  # Hz.
stft_stride = 16  # samples
stft_size = 256  # samples
file_path = "C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data/CohortA/recordings/A1.edf"


def load_raw_recording(file_path):
    data = mne.io.read_raw_edf(file_path)
    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names

    return raw_data


def load_labels(labels_path, keep_artifacts=False):
    df = pd.read_csv(labels_path, header=None)

    labels_1 = pd.get_dummies(df[1])
    labels_2 = pd.get_dummies(df[2])

    if not keep_artifacts:
        labels_1 = labels_1.iloc[:, -3:]
        labels_2 = labels_2.iloc[:, -3:]

    # column names: {1, 2, 3, n, r, w}
    # 1=wake artifact, 2=NREM artifact, 3=REM artifact

    return [labels_1, labels_2]

def epochs_analysis(labels_1, labels_2):
    # Select only epochs with agreement between scorers and optionally without artifacts
    # Label arrays should contain all the six labels

    labels_1 = labels_1.to_numpy()
    labels_2 = labels_2.to_numpy()

    # Number of epochs where experts disagree
    disagreement_epochs = labels_1 == labels_2
    disagreement_epochs = np.sum(labels_1 == labels_2, axis=1)
    disagreement_epochs = np.where(disagreement_epochs != 3)[0]
    n_disagreement_epochs = len(disagreement_epochs)

    # Number of epochs that are labeled as artifacts for every scorer
    n_artifacts_1 = len(np.where(np.sum(labels_1, axis=1) != 1)[0])
    n_artifacts_2 = len(np.where(np.sum(labels_2, axis=1) != 1)[0])

    # Number of epochs that are labeled as artifact by one expert but not by the other




    a = pd.merge(labels_1, labels_2, how='inner')

def visualize_5_epoch_windows(signal, y, labels):
    fig, ax = plt.subplots(3, 6, figsize=(15, 10), gridspec_kw={'width_ratios': [60, 1, 60, 1, 60, 1]})
    fig.subplots_adjust(hspace=0.8, wspace=0.8)

    # if plot_artifacts:
    labels_dict = {0: 'W_art', 1: 'N_art', 2: 'R_art', 3: 'NREM', 4: 'REM', 5: 'WAKE'}
    # else:
    #   labels_dict = {0: 'NREM', 1: 'REM', 2: 'WAKE'}

    for q in range(3, 6):
        counter = 0
        while counter < 3:
            cax = ax[(q - 3), counter * 2]

            # Select random epoch
            rdm_epoch_idx = np.random.randint(2, len(signal) / 4 / fs - 2)
            rdm_epoch_label = labels.to_numpy()[rdm_epoch_idx, :]
            rdm_epoch_label = np.where(rdm_epoch_label == 1)[0]
            rdm_epoch_label = labels_dict[int(rdm_epoch_label)]

            if rdm_epoch_label == labels_dict[q]:
                rdm_epoch_neighb_labels = labels.to_numpy()[rdm_epoch_idx - 2:rdm_epoch_idx + 3, :]
                rdm_epoch_neighb_labels = np.where(rdm_epoch_neighb_labels == 1)[1]
                rdm_epoch_neighb_labels = [labels_dict[i] for i in rdm_epoch_neighb_labels]

                time_axis = np.linspace((rdm_epoch_idx - 2) * fs * 4, (rdm_epoch_idx + 3) * fs * 4, 18) / fs

                rdm_epoch_spect = y[:, (rdm_epoch_idx - 2) * 32: (rdm_epoch_idx + 3) * 32]

                img = cax.imshow(rdm_epoch_spect, cmap='jet', aspect='auto')
                cax.set_title(rdm_epoch_label)
                cax.invert_yaxis()
                cax.set_xlabel('Time (s)')
                cax.set_ylabel('Frequency (Hz.)')
                cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
                cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
                cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
                cax.set_yticklabels(['24', '12', '0'])
                cax.vlines(x=cax.get_xticks()[1:-1],
                           ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
                epoch_labels_ax = cax.twiny()
                epoch_labels_ax.set_xlim(cax.get_xlim())
                epoch_labels_ax.set_xticks(np.linspace(cax.get_xlim()[1] / 10, 9 * cax.get_xlim()[1] / 10, 5))
                epoch_labels_ax.set_xticklabels(rdm_epoch_neighb_labels)
                epoch_labels_ax.tick_params(length=0)
                fig.colorbar(img, cax=ax[q - 3, (counter * 2) + 1],
                             ticks=[np.min(rdm_epoch_spect), np.max(rdm_epoch_spect)])
                counter = counter + 1
    plt.show()

def visualize_single_epochs_EEG(signal, y, labels, same_scale=True, n_samples=5):
    if not same_scale:
        for rr in range(5):
            width_ratios = matlib.repmat(np.array([60, 1]), 1, n_samples)
            fig, ax = plt.subplots(3, n_samples*2, figsize=(15, 10), gridspec_kw={'width_ratios': width_ratios.tolist()[0]})
            fig.subplots_adjust(hspace=0.8, wspace=0.8)

            # if plot_artifacts:
            labels_dict = {0: 'W_art', 1: 'N_art', 2: 'R_art', 3: 'NREM', 4: 'REM', 5: 'WAKE'}
            # else:
            #   labels_dict = {0: 'NREM', 1: 'REM', 2: 'WAKE'}

            for q in range(3, 6):
                counter = 0
                while counter < n_samples:
                    cax = ax[(q - 3), counter * 2]

                    # Select random epoch
                    rdm_epoch_idx = np.random.randint(2, len(signal) / 4 / fs - 2)
                    rdm_epoch_label = labels.to_numpy()[rdm_epoch_idx, :]
                    rdm_epoch_label = np.where(rdm_epoch_label == 1)[0]
                    rdm_epoch_label = labels_dict[int(rdm_epoch_label)]

                    if rdm_epoch_label == labels_dict[q]:
                        time_axis = np.linspace((rdm_epoch_idx) * fs * 4, (rdm_epoch_idx + 1) * fs * 4, 18) / fs

                        rdm_epoch_spect = y[:, rdm_epoch_idx * 32: (rdm_epoch_idx + 1) * 32]

                        img = cax.imshow(rdm_epoch_spect, cmap='jet', aspect='auto')

                        cax.set_title(rdm_epoch_label)
                        cax.invert_yaxis()
                        cax.set_xlabel('Time (s)')
                        cax.set_ylabel('Frequency (Hz.)')
                        cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
                        cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
                        cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
                        cax.set_yticklabels(['24', '12', '0'])
                        fig.colorbar(img, cax=ax[q - 3, (counter * 2) + 1],
                                     ticks=[np.min(rdm_epoch_spect), np.max(rdm_epoch_spect)], pad=0.001)
                        counter = counter + 1

    else:
        for rr in range(5):
            fig, ax = plt.subplots(3, n_samples, figsize=(15, 10))
            plt.tight_layout()
            fig.subplots_adjust(hspace=0.5)#, wspace=0.8)

            # if plot_artifacts:
            labels_dict = {0: 'W_art', 1: 'N_art', 2: 'R_art', 3: 'NREM', 4: 'REM', 5: 'WAKE'}
            # else:
            #   labels_dict = {0: 'NREM', 1: 'REM', 2: 'WAKE'}

            for q in range(3, 6):
                counter = 0
                while counter < n_samples:
                    cax = ax[(q - 3), counter]

                    # Select random epoch
                    rdm_epoch_idx = np.random.randint(2, len(signal) / 4 / fs - 2)
                    rdm_epoch_label = labels.to_numpy()[rdm_epoch_idx, :]
                    rdm_epoch_label = np.where(rdm_epoch_label == 1)[0]
                    rdm_epoch_label = labels_dict[int(rdm_epoch_label)]

                    if rdm_epoch_label == labels_dict[q]:
                        time_axis = np.linspace((rdm_epoch_idx) * fs * 4, (rdm_epoch_idx + 1) * fs * 4, 18) / fs

                        rdm_epoch_spect = y[:, rdm_epoch_idx * 32: (rdm_epoch_idx + 1) * 32]

                        img = cax.imshow(rdm_epoch_spect, cmap='jet', aspect='auto', vmin=-5, vmax=3)

                        cax.set_title(rdm_epoch_label)
                        cax.invert_yaxis()
                        cax.set_xlabel('Time (s)')
                        cax.set_ylabel('Frequency (Hz.)')
                        cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
                        cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
                        cax.set_yticks(
                            [cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
                        cax.set_yticklabels(['24', '12', '0'])
                        # fig.colorbar(img, cax=ax[q - 3, (counter * 2) + 1],
                        #              ticks=[np.min(rdm_epoch_spect), np.max(rdm_epoch_spect)], pad=0.001)
                        counter = counter + 1

    plt.show()

def visualize_single_epochs_EMG(signal, y, labels, same_scale=True, n_samples=5):
    if not same_scale:
        for rr in range(5):
            width_ratios = matlib.repmat(np.array([60, 1]), 1, n_samples)
            fig, ax = plt.subplots(3, n_samples*2, figsize=(15, 10), gridspec_kw={'width_ratios': width_ratios.tolist()[0]})
            fig.subplots_adjust(hspace=0.8, wspace=0.4)
            plt.tight_layout()

            # if plot_artifacts:
            labels_dict = {0: 'W_art', 1: 'N_art', 2: 'R_art', 3: 'NREM', 4: 'REM', 5: 'WAKE'}
            # else:
            #   labels_dict = {0: 'NREM', 1: 'REM', 2: 'WAKE'}

            for q in range(3, 6):
                counter = 0
                while counter < n_samples:
                    cax = ax[(q - 3), counter * 2]

                    # Select random epoch
                    rdm_epoch_idx = np.random.randint(2, len(signal) / 4 / fs - 2)
                    rdm_epoch_label = labels.to_numpy()[rdm_epoch_idx, :]
                    rdm_epoch_label = np.where(rdm_epoch_label == 1)[0]
                    rdm_epoch_label = labels_dict[int(rdm_epoch_label)]

                    if rdm_epoch_label == labels_dict[q]:
                        time_axis = np.linspace((rdm_epoch_idx) * fs * 4, (rdm_epoch_idx + 1) * fs * 4, 18) / fs

                        rdm_epoch_spect = y[:, rdm_epoch_idx * 32: (rdm_epoch_idx + 1) * 32]

                        img = cax.imshow(rdm_epoch_spect, cmap='jet', aspect='auto')

                        cax.set_title(rdm_epoch_label)
                        cax.invert_yaxis()
                        cax.set_xlabel('Time (s)')
                        cax.set_ylabel('Frequency (Hz.)')
                        cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
                        cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
                        cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
                        cax.set_yticklabels(['24', '12', '0'])
                        fig.colorbar(img, cax=ax[q - 3, (counter * 2) + 1],
                                     ticks=[np.min(rdm_epoch_spect), np.max(rdm_epoch_spect)], pad=0.1)
                        counter = counter + 1

    else:
        for rr in range(5):
            fig, ax = plt.subplots(3, n_samples, figsize=(15, 10))
            plt.tight_layout()
            fig.subplots_adjust(hspace=0.5)#, wspace=0.8)

            # if plot_artifacts:
            labels_dict = {0: 'W_art', 1: 'N_art', 2: 'R_art', 3: 'NREM', 4: 'REM', 5: 'WAKE'}
            # else:
            #   labels_dict = {0: 'NREM', 1: 'REM', 2: 'WAKE'}

            for q in range(3, 6):
                counter = 0
                while counter < n_samples:
                    cax = ax[(q - 3), counter]

                    # Select random epoch
                    rdm_epoch_idx = np.random.randint(2, len(signal) / 4 / fs - 2)
                    rdm_epoch_label = labels.to_numpy()[rdm_epoch_idx, :]
                    rdm_epoch_label = np.where(rdm_epoch_label == 1)[0]
                    rdm_epoch_label = labels_dict[int(rdm_epoch_label)]

                    if rdm_epoch_label == labels_dict[q]:
                        time_axis = np.linspace((rdm_epoch_idx) * fs * 4, (rdm_epoch_idx + 1) * fs * 4, 18) / fs

                        rdm_epoch_spect = y[:, rdm_epoch_idx * 32: (rdm_epoch_idx + 1) * 32]

                        img = cax.imshow(rdm_epoch_spect, cmap='jet', aspect='auto', vmin=-1, vmax=1.5)

                        cax.set_title(rdm_epoch_label)
                        cax.invert_yaxis()
                        cax.set_xlabel('Time (s)')
                        cax.set_ylabel('Frequency (Hz.)')
                        cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
                        cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
                        cax.set_yticks(
                            [cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
                        cax.set_yticklabels(['24', '12', '0'])
                        # fig.colorbar(img, cax=ax[q - 3, (counter * 2) + 1],
                        #              ticks=[np.min(rdm_epoch_spect), np.max(rdm_epoch_spect)], pad=0.001)
                        counter = counter + 1

    plt.show()

def preprocess_EEG(signal,
                   fs=128,
                   stft_size=256,
                   stft_stride=16,
                   lowcut=0.5,
                   highcut=24,
                   visualize=False,
                   labels=None,
                   plot_artifacts=False):

    # STFT
    f, t, Z = scipy.signal.stft(signal,
                                fs=128,
                                window='hamming',
                                nperseg=stft_size,
                                noverlap=stft_size - stft_stride
                                )

    # Bandpass (crop)
    Z = Z[np.where(f == lowcut)[0][0]: np.where(f == highcut)[0][0] + 1, :]

    # PSD
    y = np.abs(Z) ** 2

    # Log-scale
    y = 10 * np.log10(y)

    # Standardize
    y_mean = np.mean(y, axis=1, keepdims=True)
    y_std = np.std(y, axis=1, keepdims=True)

    y = (y - y_mean) / y_std

    if visualize == 'windows':
        visualize_5_epoch_windows(signal, y, labels)
    elif visualize == 'epochs':
        visualize_single_epochs_EEG(signal, y, labels)

    return y

def preprocess_EMG(signal,
                   fs=128,
                   stft_size=256,
                   stft_stride=16,
                   lowcut=0.5,
                   highcut=30,
                   visualize=False,
                   labels=None,
                   plot_artifacts=False):


    # STFT
    f, t, Z = scipy.signal.stft(signal,
                                fs=128,
                                window='hamming',
                                nperseg=stft_size,
                                noverlap=stft_size - stft_stride
                                )

    # Bandpass (crop)
    Z = Z[np.where(f == lowcut)[0][0]: np.where(f == highcut)[0][0] + 1, :]

    # PSD
    y = np.abs(Z) ** 2

    # Integration
    y = np.sum(y, axis=0)

    # Stack rows to have 2 dimensions
    y = np.expand_dims(y, axis=0)
    # y = np.repeat(y, eeg_dimensions[0], axis=0)
    y = np.repeat(y, 48, axis=0)

    # Log-scale
    y = 10*np.log10(y)

    # Standardize
    y_mean = np.mean(y, axis=1, keepdims=True)
    y_std = np.std(y, axis=1, keepdims=True)

    y = (y - y_mean) / y_std

    # if visualize == 'windows':
    #     visualize_5_epoch_windows(signal, y, labels)
    if visualize == 'epochs':
        visualize_single_epochs_EMG(signal, y, labels)

    return y


def windowing(signal, window_size=2560, window_stride=512, fs=128):
    n_windows = signal.size // window_stride - 2 * (window_size / window_stride // 2)

    windowed_signal = np.zeros((int(n_windows), window_size))

    # signal = signal[window_size//2 : -window_size//2],

    for i in range(int(n_windows)):
        windowed_signal[i, :] = signal[i * window_stride: i * window_stride + window_size]

    return windowed_signal


# --------------------------------------------------------------------------------------------------
file_path = "C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data/CohortA/recordings/A1.edf"
labels_path = 'C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data/CohortA/scorings/A1.csv'
y_1, y_2 = load_labels(labels_path, keep_artifacts=True)
# filter_epochs(y_1, y_2)
raw_data = load_raw_recording(file_path)
eeg_1 = preprocess_EEG(raw_data[0, :], labels=y_1, visualize='epochs', plot_artifacts=True)
# eeg_2 = preprocess_EEG(raw_data[1, :], labels=y_1, visualize='epochs',  plot_artifacts=True)
# emg = preprocess_EMG(raw_data[2, :], labels=y_1, visualize='epochs',  plot_artifacts=True)
