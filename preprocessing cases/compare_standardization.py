import mne
import scipy
import scipy.signal
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Input, MaxPool2D, Conv2D, Dense, Softmax, Flatten
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


def preprocess_EEG(signal,
                   fs=128,
                   stft_size=256,
                   stft_stride=16,
                   lowcut=0.5,
                   highcut=24,
                   visualize=False,
                   labels=None,
                   plot_artifacts=False):

    for q in range(3):
        if visualize:
            # Select random epoch
            rdm_epoch_idx = np.random.randint(2, len(signal) / 4 / fs - 2)
            rdm_epoch_labels = labels.to_numpy()[rdm_epoch_idx - 2:rdm_epoch_idx + 3, :]

            if plot_artifacts:
                labels_dict = {0: 'W_art', 1: 'N_art', 2: 'R_art', 3: 'NREM', 4: 'REM', 5: 'WAKE'}
            else:
                labels_dict = {0: 'NREM', 1: 'REM', 2: 'WAKE'}
            rdm_epoch_labels = np.where(rdm_epoch_labels == 1)[1]
            rdm_epoch_labels = [labels_dict[i] for i in rdm_epoch_labels]

            rdm_epoch_signal = signal[(rdm_epoch_idx - 2) * fs * 4: (rdm_epoch_idx + 3) * fs * 4]
            time_axis = np.linspace((rdm_epoch_idx - 2) * fs * 4, (rdm_epoch_idx + 3) * fs * 4, fs * 4 * 5) / fs

            fig, ax = plt.subplots(6, 2, figsize=(15, 10), gridspec_kw={'width_ratios': [60, 1]})
            fig.subplots_adjust(hspace=0.8)
            cax = ax[0, 0]
            cax.plot(time_axis, rdm_epoch_signal)
            cax.vlines(x=np.linspace(time_axis[0] + 4, time_axis[-1] - 4, 4),
                       ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
            cax.set_title('Raw 5 epochs window')
            # cax.set_xlabel('Time (s)')
            cax.set_xticks(np.linspace(time_axis[0], time_axis[-1], 6))
            cax.set_xlim((time_axis[0], time_axis[-1]))
            epoch_labels_ax = cax.twiny()
            epoch_labels_ax.set_xlim(cax.get_xlim())
            epoch_labels_ax.set_xticks(np.linspace(time_axis[0] + 2, time_axis[-1]-2, 5))
            epoch_labels_ax.set_xticklabels(rdm_epoch_labels)
            epoch_labels_ax.tick_params(length=0)
            ax[0, 1].axis('off')

        # STFT
        f, t, Z = scipy.signal.stft(signal,
                                    fs=128,
                                    window='hamming',
                                    nperseg=stft_size,
                                    noverlap=stft_size - stft_stride
                                    )

        if visualize:
            cax = ax[1, 0]

            rdm_epoch_spect = Z[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]
            # time_axis = np.linspace((rdm_epoch_idx-2)*32, (rdm_epoch_idx+3)*32, 32*5)
            time_axis = np.linspace((rdm_epoch_idx - 2) * fs * 4, (rdm_epoch_idx + 3) * fs * 4, 18) / fs

            img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
            cax.set_title('Spectrogram')
            cax.invert_yaxis()
            # cax.set_xlabel('Time (s)')
            cax.set_ylabel('Frequency (Hz.)')
            cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
            cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
            cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
            cax.set_yticklabels([str(f[-1]), str(f[-1] / 2), str(f[0])])
            cax.vlines(x=cax.get_xticks()[1:-1],
                       ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
            fig.colorbar(img, cax=ax[1, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])

        # Bandpass (crop)
        Z = Z[np.where(f == lowcut)[0][0]: np.where(f == highcut)[0][0] + 1, :]

        if visualize:
            cax = ax[2, 0]

            rdm_epoch_spect = Z[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]

            img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
            cax.set_title('Bandpass')
            cax.invert_yaxis()
            # cax.set_xlabel('Time (s)')
            cax.set_ylabel('Frequency (Hz.)')
            cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
            cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
            cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
            cax.set_yticklabels(['24', '12', '0'])
            cax.vlines(x=cax.get_xticks()[1:-1],
                       ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
            fig.colorbar(img, cax=ax[2, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])

        # PSD
        y = np.abs(Z) ** 2

        if visualize:
            cax = ax[3, 0]

            rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]

            img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
            cax.set_title('PSD')
            cax.invert_yaxis()
            # cax.set_xlabel('Time (s)')
            cax.set_ylabel('Frequency (Hz.)')
            cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
            cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
            cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
            cax.set_yticklabels(['24', '12', '0'])
            cax.vlines(x=cax.get_xticks()[1:-1],
                       ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
            fig.colorbar(img, cax=ax[3, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])

        # Log-scale
        y = 10 * np.log10(y)

        if visualize:
            cax = ax[4, 0]

            rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]

            img = cax.imshow(rdm_epoch_spect, cmap='jet', aspect='auto')
            cax.set_title('Log transformation')
            cax.invert_yaxis()
            # cax.set_xlabel('Time (s)')
            cax.set_ylabel('Frequency (Hz.)')
            cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
            cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
            cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
            cax.set_yticklabels(['24', '12', '0'])
            cax.vlines(x=cax.get_xticks()[1:-1],
                       ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
            fig.colorbar(img, cax=ax[4, 1], ticks=[np.min(rdm_epoch_spect), np.max(rdm_epoch_spect)])

        # Standardize
        y_mean = np.mean(y, axis=1, keepdims=True)
        y_std = np.std(y, axis=1, keepdims=True)

        y = (y - y_mean) / y_std

        if visualize:
            cax = ax[5, 0]

            rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]

            img = cax.imshow(rdm_epoch_spect, cmap='jet', aspect='auto')
            cax.set_title('Standardization')
            cax.invert_yaxis()
            cax.set_xlabel('Time (s)')
            cax.set_ylabel('Frequency (Hz.)')
            cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
            cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
            cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
            cax.set_yticklabels(['24', '12', '0'])
            cax.vlines(x=cax.get_xticks()[1:-1],
                       ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
            fig.colorbar(img, cax=ax[5, 1], ticks=[np.min(rdm_epoch_spect), np.max(rdm_epoch_spect)])
            plt.show()





        if visualize:
            rdm_epoch_signal = signal[(rdm_epoch_idx - 2) * fs * 4: (rdm_epoch_idx + 3) * fs * 4]
            time_axis = np.linspace((rdm_epoch_idx - 2) * fs * 4, (rdm_epoch_idx + 3) * fs * 4, fs * 4 * 5) / fs

            fig, ax = plt.subplots(6, 2, figsize=(15, 10), gridspec_kw={'width_ratios': [60, 1]})
            fig.subplots_adjust(hspace=0.8)
            cax = ax[0, 0]
            cax.plot(time_axis, rdm_epoch_signal)
            cax.vlines(x=np.linspace(time_axis[0] + 4, time_axis[-1] - 4, 4),
                       ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
            cax.set_title('Raw 5 epochs window')
            # cax.set_xlabel('Time (s)')
            cax.set_xticks(np.linspace(time_axis[0], time_axis[-1], 6))
            cax.set_xlim((time_axis[0], time_axis[-1]))
            epoch_labels_ax = cax.twiny()
            epoch_labels_ax.set_xlim(cax.get_xlim())
            epoch_labels_ax.set_xticks(np.linspace(time_axis[0]+ 2, time_axis[-1]-2, 5))
            epoch_labels_ax.set_xticklabels(rdm_epoch_labels)
            epoch_labels_ax.tick_params(length=0)
            ax[0, 1].axis('off')

        # STFT
        f, t, Z = scipy.signal.stft(signal,
                                    fs=128,
                                    window='hamming',
                                    nperseg=stft_size,
                                    noverlap=stft_size - stft_stride
                                    )

        if visualize:
            cax = ax[1, 0]

            rdm_epoch_spect = Z[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]
            # time_axis = np.linspace((rdm_epoch_idx-2)*32, (rdm_epoch_idx+3)*32, 32*5)
            time_axis = np.linspace((rdm_epoch_idx - 2) * fs * 4, (rdm_epoch_idx + 3) * fs * 4, 18) / fs

            img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
            cax.set_title('Spectrogram')
            cax.invert_yaxis()
            # cax.set_xlabel('Time (s)')
            cax.set_ylabel('Frequency (Hz.)')
            cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
            cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
            cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
            cax.set_yticklabels([str(f[-1]), str(f[-1] / 2), str(f[0])])
            cax.vlines(x=cax.get_xticks()[1:-1],
                       ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
            fig.colorbar(img, cax=ax[1, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])

        # Bandpass (crop)
        Z = Z[np.where(f == lowcut)[0][0]: np.where(f == highcut)[0][0] + 1, :]

        if visualize:
            cax = ax[2, 0]

            rdm_epoch_spect = Z[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]

            img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
            cax.set_title('Bandpass')
            cax.invert_yaxis()
            # cax.set_xlabel('Time (s)')
            cax.set_ylabel('Frequency (Hz.)')
            cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
            cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
            cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
            cax.set_yticklabels(['24', '12', '0'])
            cax.vlines(x=cax.get_xticks()[1:-1],
                       ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
            fig.colorbar(img, cax=ax[2, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])

        # PSD
        y = np.abs(Z) ** 2

        if visualize:
            cax = ax[3, 0]

            rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]

            img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
            cax.set_title('PSD')
            cax.invert_yaxis()
            # cax.set_xlabel('Time (s)')
            cax.set_ylabel('Frequency (Hz.)')
            cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
            cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
            cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
            cax.set_yticklabels(['24', '12', '0'])
            cax.vlines(x=cax.get_xticks()[1:-1],
                       ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
            fig.colorbar(img, cax=ax[3, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])

        # Log-scale
        y = 10 * np.log10(y)

        if visualize:
            cax = ax[4, 0]

            rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]

            img = cax.imshow(rdm_epoch_spect, cmap='jet', aspect='auto')
            cax.set_title('Log transformation')
            cax.invert_yaxis()
            # cax.set_xlabel('Time (s)')
            cax.set_ylabel('Frequency (Hz.)')
            cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
            cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
            cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
            cax.set_yticklabels(['24', '12', '0'])
            cax.vlines(x=cax.get_xticks()[1:-1],
                       ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
            fig.colorbar(img, cax=ax[4, 1], ticks=[np.min(rdm_epoch_spect), np.max(rdm_epoch_spect)])

        # Standardize
        y_mean = np.mean(rdm_epoch_spect, axis=1, keepdims=True)
        y_std = np.std(rdm_epoch_spect, axis=1, keepdims=True)

        y = (y - y_mean) / y_std

        if visualize:
            cax = ax[5, 0]

            rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]

            img = cax.imshow(rdm_epoch_spect, cmap='jet', aspect='auto')
            cax.set_title('Standardization')
            cax.invert_yaxis()
            cax.set_xlabel('Time (s)')
            cax.set_ylabel('Frequency (Hz.)')
            cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
            cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
            cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
            cax.set_yticklabels(['24', '12', '0'])
            cax.vlines(x=cax.get_xticks()[1:-1],
                       ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
            fig.colorbar(img, cax=ax[5, 1], ticks=[np.min(rdm_epoch_spect), np.max(rdm_epoch_spect)])
            plt.show()


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

    for q in range(3):
        if visualize:
            # Select random epoch
            rdm_epoch_idx = np.random.randint(2, len(signal) / 4 / fs - 2)
            rdm_epoch_labels = labels.to_numpy()[rdm_epoch_idx - 2:rdm_epoch_idx + 3, :]

            if plot_artifacts:
                labels_dict = {0: 'W_art', 1: 'N_art', 2: 'R_art', 3: 'NREM', 4: 'REM', 5: 'WAKE'}
            else:
                labels_dict = {0: 'NREM', 1: 'REM', 2: 'WAKE'}
            rdm_epoch_labels = np.where(rdm_epoch_labels == 1)[1]
            rdm_epoch_labels = [labels_dict[i] for i in rdm_epoch_labels]

            rdm_epoch_signal = signal[(rdm_epoch_idx - 2) * fs * 4: (rdm_epoch_idx + 3) * fs * 4]
            time_axis = np.linspace((rdm_epoch_idx - 2) * fs * 4, (rdm_epoch_idx + 3) * fs * 4, fs * 4 * 5) / fs

            fig, ax = plt.subplots(7, 2, figsize=(15, 10), gridspec_kw={'width_ratios': [60, 1]})
            fig.subplots_adjust(hspace=0.8)
            cax = ax[0, 0]
            cax.plot(time_axis, rdm_epoch_signal)
            cax.vlines(x=np.linspace(time_axis[0] + 4, time_axis[-1] - 4, 4),
                       ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
            cax.set_title('Raw 5 epochs window')
            # cax.set_xlabel('Time (s)')
            cax.set_xticks(np.linspace(time_axis[0], time_axis[-1], 6))
            cax.set_xlim((time_axis[0], time_axis[-1]))
            epoch_labels_ax = cax.twiny()
            epoch_labels_ax.set_xlim(cax.get_xlim())
            epoch_labels_ax.set_xticks(np.linspace(time_axis[0] + 2, time_axis[-1]-2, 5))
            epoch_labels_ax.set_xticklabels(rdm_epoch_labels)
            epoch_labels_ax.tick_params(length=0)
            ax[0, 1].axis('off')

        # STFT
        f, t, Z = scipy.signal.stft(signal,
                                    fs=128,
                                    window='hamming',
                                    nperseg=stft_size,
                                    noverlap=stft_size - stft_stride
                                    )

        if visualize:
            cax = ax[1, 0]

            rdm_epoch_spect = Z[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]
            # time_axis = np.linspace((rdm_epoch_idx-2)*32, (rdm_epoch_idx+3)*32, 32*5)
            time_axis = np.linspace((rdm_epoch_idx - 2) * fs * 4, (rdm_epoch_idx + 3) * fs * 4, 18) / fs

            img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
            cax.set_title('Spectrogram')
            cax.invert_yaxis()
            # cax.set_xlabel('Time (s)')
            cax.set_ylabel('Frequency (Hz.)')
            cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
            cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
            cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
            cax.set_yticklabels([str(f[-1]), str(f[-1] / 2), str(f[0])])
            cax.vlines(x=cax.get_xticks()[1:-1],
                       ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
            fig.colorbar(img, cax=ax[1, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])

        # Bandpass (crop)
        Z = Z[np.where(f == lowcut)[0][0]: np.where(f == highcut)[0][0] + 1, :]

        if visualize:
            cax = ax[2, 0]

            rdm_epoch_spect = Z[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]

            img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
            cax.set_title('Bandpass')
            cax.invert_yaxis()
            # cax.set_xlabel('Time (s)')
            cax.set_ylabel('Frequency (Hz.)')
            cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
            cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
            cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
            cax.set_yticklabels(['24', '12', '0'])
            cax.vlines(x=cax.get_xticks()[1:-1],
                       ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
            fig.colorbar(img, cax=ax[2, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])

        # PSD
        y = np.abs(Z) ** 2

        if visualize:
            cax = ax[3, 0]

            rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]

            img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
            cax.set_title('PSD')
            cax.invert_yaxis()
            # cax.set_xlabel('Time (s)')
            cax.set_ylabel('Frequency (Hz.)')
            cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
            cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
            cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
            cax.set_yticklabels(['24', '12', '0'])
            cax.vlines(x=cax.get_xticks()[1:-1],
                       ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
            fig.colorbar(img, cax=ax[3, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])

        # Integration
        y = np.sum(y, axis=0)

        # Stack rows to have 2 dimensions
        y = np.expand_dims(y, axis=0)
        # y = np.repeat(y, eeg_dimensions[0], axis=0)
        y = np.repeat(y, 48, axis=0)

        if visualize:
            cax = ax[4, 0]

            rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]

            img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
            cax.set_title('Integration')
            cax.invert_yaxis()
            # cax.set_xlabel('Time (s)')
            cax.set_ylabel('Frequency (Hz.)')
            cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
            cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
            cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
            cax.set_yticklabels(['24', '12', '0'])
            cax.vlines(x=cax.get_xticks()[1:-1],
                       ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
            fig.colorbar(img, cax=ax[4, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])

        # Log-scale
        y = 10*np.log10(y)

        if visualize:
            cax = ax[5, 0]

            rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]

            img = cax.imshow(rdm_epoch_spect, cmap='jet', aspect='auto')
            cax.set_title('Log transformation')
            cax.invert_yaxis()
            # cax.set_xlabel('Time (s)')
            cax.set_ylabel('Frequency (Hz.)')
            cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
            cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
            cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
            cax.set_yticklabels(['24', '12', '0'])
            cax.vlines(x=cax.get_xticks()[1:-1],
                       ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
            fig.colorbar(img, cax=ax[5, 1], ticks=[np.min(rdm_epoch_spect), np.max(rdm_epoch_spect)])

        # Standardize
        y_mean = np.mean(y, axis=1, keepdims=True)
        y_std = np.std(y, axis=1, keepdims=True)

        y = (y - y_mean) / y_std

        if visualize:
            cax = ax[6, 0]

            rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]

            img = cax.imshow(rdm_epoch_spect, cmap='jet', aspect='auto')
            cax.set_title('Standardization')
            cax.invert_yaxis()
            cax.set_xlabel('Time (s)')
            cax.set_ylabel('Frequency (Hz.)')
            cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
            cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
            cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
            cax.set_yticklabels(['24', '12', '0'])
            cax.vlines(x=cax.get_xticks()[1:-1],
                       ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
            fig.colorbar(img, cax=ax[6, 1], ticks=[np.min(rdm_epoch_spect), np.max(rdm_epoch_spect)])
            plt.show()





        if visualize:
            rdm_epoch_signal = signal[(rdm_epoch_idx - 2) * fs * 4: (rdm_epoch_idx + 3) * fs * 4]
            time_axis = np.linspace((rdm_epoch_idx - 2) * fs * 4, (rdm_epoch_idx + 3) * fs * 4, fs * 4 * 5) / fs

            fig, ax = plt.subplots(7, 2, figsize=(15, 10), gridspec_kw={'width_ratios': [60, 1]})
            fig.subplots_adjust(hspace=0.8)
            cax = ax[0, 0]
            cax.plot(time_axis, rdm_epoch_signal)
            cax.vlines(x=np.linspace(time_axis[0] + 4, time_axis[-1] - 4, 4),
                       ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
            cax.set_title('Raw 5 epochs window')
            # cax.set_xlabel('Time (s)')
            cax.set_xticks(np.linspace(time_axis[0], time_axis[-1], 6))
            cax.set_xlim((time_axis[0], time_axis[-1]))
            epoch_labels_ax = cax.twiny()
            epoch_labels_ax.set_xlim(cax.get_xlim())
            epoch_labels_ax.set_xticks(np.linspace(time_axis[0] + 2, time_axis[-1] - 2, 5))
            epoch_labels_ax.set_xticklabels(rdm_epoch_labels)
            epoch_labels_ax.tick_params(length=0)
            ax[0, 1].axis('off')

        # STFT
        f, t, Z = scipy.signal.stft(signal,
                                    fs=128,
                                    window='hamming',
                                    nperseg=stft_size,
                                    noverlap=stft_size - stft_stride
                                    )

        if visualize:
            cax = ax[1, 0]

            rdm_epoch_spect = Z[:, (rdm_epoch_idx - 2) * 32: (rdm_epoch_idx + 3) * 32]
            # time_axis = np.linspace((rdm_epoch_idx-2)*32, (rdm_epoch_idx+3)*32, 32*5)
            time_axis = np.linspace((rdm_epoch_idx - 2) * fs * 4, (rdm_epoch_idx + 3) * fs * 4, 18) / fs

            img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
            cax.set_title('Spectrogram')
            cax.invert_yaxis()
            # cax.set_xlabel('Time (s)')
            cax.set_ylabel('Frequency (Hz.)')
            cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
            cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
            cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
            cax.set_yticklabels([str(f[-1]), str(f[-1] / 2), str(f[0])])
            cax.vlines(x=cax.get_xticks()[1:-1],
                       ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
            fig.colorbar(img, cax=ax[1, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])

        # Bandpass (crop)
        Z = Z[np.where(f == lowcut)[0][0]: np.where(f == highcut)[0][0] + 1, :]

        if visualize:
            cax = ax[2, 0]

            rdm_epoch_spect = Z[:, (rdm_epoch_idx - 2) * 32: (rdm_epoch_idx + 3) * 32]

            img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
            cax.set_title('Bandpass')
            cax.invert_yaxis()
            # cax.set_xlabel('Time (s)')
            cax.set_ylabel('Frequency (Hz.)')
            cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
            cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
            cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
            cax.set_yticklabels(['24', '12', '0'])
            cax.vlines(x=cax.get_xticks()[1:-1],
                       ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
            fig.colorbar(img, cax=ax[2, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])

        # PSD
        y = np.abs(Z) ** 2

        if visualize:
            cax = ax[3, 0]

            rdm_epoch_spect = y[:, (rdm_epoch_idx - 2) * 32: (rdm_epoch_idx + 3) * 32]

            img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
            cax.set_title('PSD')
            cax.invert_yaxis()
            # cax.set_xlabel('Time (s)')
            cax.set_ylabel('Frequency (Hz.)')
            cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
            cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
            cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
            cax.set_yticklabels(['24', '12', '0'])
            cax.vlines(x=cax.get_xticks()[1:-1],
                       ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
            fig.colorbar(img, cax=ax[3, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])

        # Integration
        y = np.sum(y, axis=0)

        # Stack rows to have 2 dimensions
        y = np.expand_dims(y, axis=0)
        # y = np.repeat(y, eeg_dimensions[0], axis=0)
        y = np.repeat(y, 48, axis=0)

        if visualize:
            cax = ax[4, 0]

            rdm_epoch_spect = y[:, (rdm_epoch_idx - 2) * 32: (rdm_epoch_idx + 3) * 32]

            img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
            cax.set_title('Integration')
            cax.invert_yaxis()
            # cax.set_xlabel('Time (s)')
            cax.set_ylabel('Frequency (Hz.)')
            cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
            cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
            cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
            cax.set_yticklabels(['24', '12', '0'])
            cax.vlines(x=cax.get_xticks()[1:-1],
                       ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
            fig.colorbar(img, cax=ax[4, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])

        # Log-scale
        y = 10 * np.log10(y)

        if visualize:
            cax = ax[5, 0]

            rdm_epoch_spect = y[:, (rdm_epoch_idx - 2) * 32: (rdm_epoch_idx + 3) * 32]

            img = cax.imshow(rdm_epoch_spect, cmap='jet', aspect='auto')
            cax.set_title('Log transformation')
            cax.invert_yaxis()
            # cax.set_xlabel('Time (s)')
            cax.set_ylabel('Frequency (Hz.)')
            cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
            cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
            cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
            cax.set_yticklabels(['24', '12', '0'])
            cax.vlines(x=cax.get_xticks()[1:-1],
                       ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
            fig.colorbar(img, cax=ax[5, 1], ticks=[np.min(rdm_epoch_spect), np.max(rdm_epoch_spect)])

        # Standardize
        y_mean = np.mean(rdm_epoch_spect, axis=1, keepdims=True)
        y_std = np.std(rdm_epoch_spect, axis=1, keepdims=True)

        y = (y - y_mean) / y_std

        if visualize:
            cax = ax[6, 0]

            rdm_epoch_spect = y[:, (rdm_epoch_idx - 2) * 32: (rdm_epoch_idx + 3) * 32]

            img = cax.imshow(rdm_epoch_spect, cmap='jet', aspect='auto')
            cax.set_title('Standardization')
            cax.invert_yaxis()
            cax.set_xlabel('Time (s)')
            cax.set_ylabel('Frequency (Hz.)')
            cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
            cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
            cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
            cax.set_yticklabels(['24', '12', '0'])
            cax.vlines(x=cax.get_xticks()[1:-1],
                       ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
            fig.colorbar(img, cax=ax[6, 1], ticks=[np.min(rdm_epoch_spect), np.max(rdm_epoch_spect)])
            plt.show()

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
eeg_1 = preprocess_EEG(raw_data[0, :], labels=y_1, visualize=True, plot_artifacts=True)
# eeg_2 = preprocess_EEG(raw_data[1, :], labels=y_1, visualize=True,  plot_artifacts=True)
emg = preprocess_EMG(raw_data[2, :], labels=y_1, visualize=True,  plot_artifacts=True)
