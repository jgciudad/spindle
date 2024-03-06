from kornum_data_loading import load_recording
import os
import string
import random
import pandas as pd
import numpy as np
# from final_preprocessing import preprocess_EEG, preprocess_EMG
from spindle_data.spindle_data_loading import windowing, load_raw_recording
import scipy.signal
from matplotlib import pyplot as plt

dataset_folder = r'C:\Users\javig\Documents\THESIS_DATA\Raw kornum lab data\Laura-EEGdata (original)\data-Kornum'
destination_folder = r'C:\Users\javig\Documents\THESIS DATA\Raw kornum lab data\to_numpy\spectrograms'

training_validation_labels = [
    # r'2DTUSERVER-Alexandra\tsv\M23-b1.tsv',
    #                           r'2DTUSERVER-Alexandra\tsv\M23-b2.tsv',
    #                           r'2DTUSERVER-Alexandra\tsv\M23-b3.tsv',
    #                           r'2DTUSERVER-Alexandra\tsv\M29-b1.tsv',
    #                           r'2DTUSERVER-Alexandra\tsv\M29-b2.tsv',
    #                           r'2DTUSERVER-Alexandra\tsv\M29-b3.tsv',
    #                           r'2DTUSERVER-Alexandra\tsv\M48-b1.tsv',
    #                           r'2DTUSERVER-Alexandra\tsv\M48-b2.tsv',
    #                           r'2DTUSERVER-Alexandra\tsv\M48-b3.tsv',
    #                           r'2DTUSERVER-Alexandra\tsv\M52-b1.tsv',
    #                           r'2DTUSERVER-Alexandra\tsv\M52-b3.tsv',
    #                           r'2DTUSERVER-Alexandra\tsv\M58-b1.tsv',
    #                           r'2DTUSERVER-Alexandra\tsv\M58-b3.tsv',
    #                           r'2DTUSERVER-CH\tsv\m1-b1.tsv',
    #                           r'2DTUSERVER-CH\tsv\m11-b1.tsv',
    #                           r'2DTUSERVER-CH\tsv\m12-b1.tsv',
    #                           r'2DTUSERVER-CH\tsv\m13-b1.tsv',
    #                           r'2DTUSERVER-CH\tsv\m14-b1.tsv',
    #                           r'2DTUSERVER-CH\tsv\m14-r3.tsv',
    #                           r'2DTUSERVER-CH\tsv\m15-r3.tsv',
    #                           r'2DTUSERVER-CH\tsv\m2-b1.tsv',
    #                           r'2DTUSERVER-CH\tsv\m3-b1.tsv',
    #                           r'2DTUSERVER-CH\tsv\m4-b1.tsv',
    #                           r'2DTUSERVER-CH\tsv\m5-b1.tsv',
    #                           r'2DTUSERVER-CH\tsv\m6-b1.tsv',
    #                           r'2DTUSERVER-CH\tsv\m6-r3.tsv',
    #                           r'2DTUSERVER-CH\tsv\m7-b1.tsv',
    #                           r'2DTUSERVER-CH\tsv\m8-b1.tsv',
    #                           r'2DTUSERVER-CH\tsv\m8-r3.tsv',
    #                           r'2DTUSERVER-LOUISE\tsv\M16-b2.tsv',
    #                           r'2DTUSERVER-LOUISE\tsv\M16-b3.tsv',
    #                           r'2DTUSERVER-LOUISE\tsv\M18-b3.tsv',
    #                           r'2DTUSERVER-LOUISE\tsv\M20-b3.tsv',
    r'2DTUSERVER-LOUISE\tsv\M309-b1.tsv',
    #                           r'2DTUSERVER-Maria\tsv\m121-b1.tsv',
    #                           r'2DTUSERVER-Maria\tsv\m121-b2.tsv',
    # r'2DTUSERVER-Maria\tsv\m61-b1.tsv',
    # r'2DTUSERVER-Maria\tsv\m63-b1.tsv',
    # r'2DTUSERVER-Maria\tsv\m63-b2.tsv',
    # r'2DTUSERVER-Maria\tsv\m86-b1.tsv',
    # r'2DTUSERVER-Maria\tsv\m86-b2.tsv',
    r'2DTUSERVER-Maria\tsv\m88-b1.tsv',
    # r'2DTUSERVER-Maria\tsv\m88-b2.tsv',
    # r'2DTUSERVER-Maria\tsv\m94-b2.tsv',
    # r'2DTUSERVER-Maria\tsv\m96-b1.tsv',
    # r'2DTUSERVER-Maria\tsv\m96-b2.tsv',
    # r'2DTUSERVER-Marieke\tsv\m2-b1.tsv',
    # r'2DTUSERVER-Marieke\tsv\m21-b1.tsv'
]

training_validation_signals = [
    # r'2DTUSERVER-Alexandra\EDF\M23-b1.edf',
    #                            r'2DTUSERVER-Alexandra\EDF\M23-b2.edf',
    #                            r'2DTUSERVER-Alexandra\EDF\M23-b3.edf',
    #                            r'2DTUSERVER-Alexandra\EDF\M29-b1.edf',
    #                            r'2DTUSERVER-Alexandra\EDF\M29-b2.edf',
    #                            r'2DTUSERVER-Alexandra\EDF\M29-b3.edf',
    #                            r'2DTUSERVER-Alexandra\EDF\M48-b1.edf',
    #                            r'2DTUSERVER-Alexandra\EDF\M48-b2.edf',
    #                            r'2DTUSERVER-Alexandra\EDF\M48-b3.edf',
    #                            r'2DTUSERVER-Alexandra\EDF\M52-b1.edf',
    #                            r'2DTUSERVER-Alexandra\EDF\M52-b3.edf',
    #                            r'2DTUSERVER-Alexandra\EDF\M58-b1.edf',
    #                            r'2DTUSERVER-Alexandra\EDF\M58-b3.edf',
    #                            r'2DTUSERVER-CH\EDF\m1-b1.edf',
    #                            r'2DTUSERVER-CH\EDF\m11-b1.edf',
    #                            r'2DTUSERVER-CH\EDF\m12-b1.edf',
    #                            r'2DTUSERVER-CH\EDF\m13-b1.edf',
    #                            r'2DTUSERVER-CH\EDF\m14-b1.edf',
    #                            r'2DTUSERVER-CH\EDF\m14-r3.edf',
    #                            r'2DTUSERVER-CH\EDF\m15-r3.edf',
    #                            r'2DTUSERVER-CH\EDF\m2-b1.edf',
    #                            r'2DTUSERVER-CH\EDF\m3-b1.edf',
    #                            r'2DTUSERVER-CH\EDF\m4-b1.edf',
    #                            r'2DTUSERVER-CH\EDF\m5-b1.edf',
    #                            r'2DTUSERVER-CH\EDF\m6-b1.edf',
    #                            r'2DTUSERVER-CH\EDF\m6-r3.edf',
    #                            r'2DTUSERVER-CH\EDF\m7-b1.edf',
    #                            r'2DTUSERVER-CH\EDF\m8-b1.edf',
    #                            r'2DTUSERVER-CH\EDF\m8-r3.edf',
    #                            r'2DTUSERVER-LOUISE\EDF\M16-b2.edf',
    #                            r'2DTUSERVER-LOUISE\EDF\M16-b3.edf',
    #                            r'2DTUSERVER-LOUISE\EDF\M18-b3.edf',
    #                            r'2DTUSERVER-LOUISE\EDF\M20-b3.edf',
    r'2DTUSERVER-LOUISE\EDF\M309-b1.edf',
    #                            r'2DTUSERVER-Maria\EDF\m121-b1.edf',
    #                            r'2DTUSERVER-Maria\EDF\m121-b2.edf',
    # r'2DTUSERVER-Maria\EDF\m61-b1.edf',
    # r'2DTUSERVER-Maria\EDF\m63-b1.edf',
    # r'2DTUSERVER-Maria\EDF\m63-b2.edf',
    # r'2DTUSERVER-Maria\EDF\m86-b1.edf',
    # r'2DTUSERVER-Maria\EDF\m86-b2.edf',
    r'2DTUSERVER-Maria\EDF\m88-b1.edf',
    # r'2DTUSERVER-Maria\EDF\m88-b2.edf',
    # r'2DTUSERVER-Maria\EDF\m94-b2.edf',
    # r'2DTUSERVER-Maria\EDF\m96-b1.edf',
    # r'2DTUSERVER-Maria\EDF\m96-b2.edf',
    # r'2DTUSERVER-Marieke\EDF\m2-b1.edf',
    # r'2DTUSERVER-Marieke\EDF\m21-b1.edf'
]

test_signals = [r'2DTUSERVER-Alexandra\EDF\M52-b2.edf',
                #                 r'2DTUSERVER-Alexandra\EDF\M58-b2.edf',
                #                 r'2DTUSERVER-CH\EDF\m15-b1.edf',
                #                 r'2DTUSERVER-LOUISE\EDF\M18-b2.edf',
                #                 r'2DTUSERVER-LOUISE\EDF\M313-b1.edf',
                r'2DTUSERVER-Maria\EDF\m61-b2.edf',
                #                 r'2DTUSERVER-Maria\EDF\m94-b1.edf'
                ]

test_labels = [r'2DTUSERVER-Alexandra\tsv\M52-b2.tsv',
               #                r'2DTUSERVER-Alexandra\tsv\M58-b2.tsv',
               #                r'2DTUSERVER-CH\tsv\m15-b1.tsv',
               #                r'2DTUSERVER-LOUISE\tsv\M18-b2.tsv',
               #                r'2DTUSERVER-LOUISE\tsv\M313-b1.tsv',
               r'2DTUSERVER-Maria\tsv\m61-b2.tsv',
               #                r'2DTUSERVER-Maria\tsv\m94-b1.tsv'
               ]

number_of_files = len(training_validation_signals) + len(test_signals)
file_counter = 1


def load_labels(labels_path,
                just_artifact_labels,
                just_stage_labels):
    df = pd.read_csv(labels_path, skiprows=9, engine='python', sep='\t', index_col=False).iloc[:, 4]
    df.loc[(df != 1) & (df != 2) & (df != 3)] = 4
    df = df.iloc[2:-2].copy()  # Drop 2 first and 2 last epochs

    labels = pd.get_dummies(df)
    if labels.shape[
        1] == 4:  # Assing name and reorder columns. If shape[1]<4 is because there isn't any artifact, so the artifact column is added 'artificially'
        labels.columns = ['WAKE', 'NREM', 'REM', 'Art']  # Assign name to columns
        labels = labels[['NREM', 'REM', 'WAKE', 'Art']]  # Reorder columns to have same order as in spindle data
    elif labels.shape[1] < 4:
        labels.columns = ['WAKE', 'NREM', 'REM']  # Assign name to columns
        labels = labels[['NREM', 'REM', 'WAKE']]  # Reorder columns to have same order as in spindle data
        labels.insert(3, "Art", 0)

    if just_artifact_labels == True:
        labels = labels['Art']

    elif just_stage_labels == True:
        labels = labels[['NREM', 'REM', 'WAKE']]

    return labels

def preprocess_EEG(signal,
                   fs=128,
                   stft_size=256,
                   stft_stride=16,
                   lowcut=0.5,
                   highcut=24,
                   visualize=False,
                   labels=None,
                   plot_artifacts=False):
    if visualize:
        # Select random epoch
        # rdm_epoch_idx = np.random.randint(2, len(signal) / 4 / fs - 2)
        rdm_epoch_idx = 3405
        rdm_epoch_labels = labels.to_numpy()[rdm_epoch_idx - 2:rdm_epoch_idx + 3, :]

        if plot_artifacts:
            labels_dict = {0: 'NREM', 1: 'REM', 2: 'WAKE', 3: 'ART'}
        else:
            labels_dict = {0: 'NREM', 1: 'REM', 2: 'WAKE'}
        rdm_epoch_labels = np.where(rdm_epoch_labels == 1)[1]
        rdm_epoch_labels = [labels_dict[i] for i in rdm_epoch_labels]

        rdm_epoch_signal = signal[(rdm_epoch_idx - 2) * fs * 4: (rdm_epoch_idx + 3) * fs * 4]
        time_axis = np.linspace((rdm_epoch_idx - 2) * fs * 4, (rdm_epoch_idx + 3) * fs * 4, fs * 4 * 5) / fs

        fig, ax = plt.subplots(6, 2, figsize=(15, 10), gridspec_kw={'width_ratios': [60, 1]})
        fig.subplots_adjust(hspace=0.8)
        cax = ax[0, 0]
        cax.plot(time_axis, rdm_epoch_signal*1000000, zorder=0)
        cax.tick_params(labelsize=12)
        cax.set_ylim(cax.get_ylim())
        cax.vlines(x=np.linspace(time_axis[0] + 4, time_axis[-1] - 4, 4),
                   ymin=cax.get_ylim()[0]+1, ymax=cax.get_ylim()[1]-1, color='k', linewidth=2.5)
        cax.set_title('Step 0: Raw 5 epochs window', fontsize=15)
        cax.set_ylabel('', fontsize=15)
        # cax.set_xlabel('Time (s)', fontsize=15)
        cax.set_ylabel(r'Voltage ($\mu$V)', fontsize=13)
        cax.set_xticks(np.linspace(time_axis[0], time_axis[-1], 6))
        cax.set_xlim((time_axis[0], time_axis[-1]))
        cax.get_xaxis().set_visible(False)
        epoch_labels_ax = cax.twiny()
        epoch_labels_ax.set_xlim(cax.get_xlim())
        epoch_labels_ax.set_xticks(np.linspace(time_axis[0]+ 2, time_axis[-1]-2, 5))
        epoch_labels_ax.set_xticklabels(rdm_epoch_labels, fontsize=13)
        epoch_labels_ax.tick_params(length=0)
        epoch_labels_ax.get_xticklabels()[2].set_fontsize(15)
        # epoch_labels_ax.get_xticklabels()[2].set_weight("bold")
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
        cax.set_title('Step 2: Spectrogram', fontsize=15)
        cax.tick_params(labelsize=13)
        cax.invert_yaxis()
        # cax.set_xlabel('Time (s)')
        cax.set_ylabel('Freq. (Hz)', fontsize=13)
        cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
        cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
        cax.get_xaxis().set_visible(False)
        cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
        cax.set_yticklabels([str(f[-1]), str(f[-1] / 2), str(f[0])])
        cax.set_yticklabels(['64', '32', '0'])
        cax.set_ylim(cax.get_ylim())
        cax.vlines(x=cax.get_xticks()[1:-1],
                   ymin=cax.get_ylim()[0]-50, ymax=cax.get_ylim()[1]+50, color='k', linewidth=2.5)
        fig.colorbar(img, cax=ax[1, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])

    # Bandpass (crop)
    Z = Z[np.where(f == lowcut)[0][0]: np.where(f == highcut)[0][0] + 1, :]

    if visualize:
        cax = ax[2, 0]

        rdm_epoch_spect = Z[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]

        img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
        cax.set_title('Step 3: Frequency range selection', fontsize=15)
        cax.tick_params(labelsize=13)
        cax.invert_yaxis()
        # cax.set_xlabel('Time (s)')
        cax.set_ylabel('Freq. (Hz)', fontsize=13)
        cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
        cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
        cax.get_xaxis().set_visible(False)
        cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
        cax.set_yticklabels(['24', '12', '0.5'])
        cax.set_ylim(cax.get_ylim())
        cax.vlines(x=cax.get_xticks()[1:-1],
                   ymin=cax.get_ylim()[0]-50, ymax=cax.get_ylim()[1]+50, color='k', linewidth=2.5)
        fig.colorbar(img, cax=ax[2, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])

    # PSD
    y = np.abs(Z) ** 2

    if visualize:
        cax = ax[3, 0]

        rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]

        img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
        cax.set_title('Step 4: PSD', fontsize=15)
        cax.tick_params(labelsize=13)
        cax.invert_yaxis()
        # cax.set_xlabel('Time (s)')
        cax.set_ylabel('Freq. (Hz)', fontsize=13)
        cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
        cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
        cax.get_xaxis().set_visible(False)
        cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
        cax.set_yticklabels(['24', '12', '0.5'])
        cax.set_ylim(cax.get_ylim())
        cax.vlines(x=cax.get_xticks()[1:-1],
                   ymin=cax.get_ylim()[0]-50, ymax=cax.get_ylim()[1]+50, color='k', linewidth=2.5)
        fig.colorbar(img, cax=ax[3, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])

    # Log-scale
    y = 10 * np.log10(y)

    if visualize:
        cax = ax[4, 0]

        rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]

        img = cax.imshow(rdm_epoch_spect, cmap='jet', aspect='auto')
        cax.set_title('Step 6: Log transformation', fontsize=15)
        cax.tick_params(labelsize=13)
        cax.invert_yaxis()
        # cax.set_xlabel('Time (s)')
        cax.set_ylabel('Freq. (Hz)', fontsize=13)
        cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
        cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
        cax.get_xaxis().set_visible(False)
        cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
        cax.set_yticklabels(['24', '12', '0.5'])
        cax.set_ylim(cax.get_ylim())
        cax.vlines(x=cax.get_xticks()[1:-1],
                   ymin=cax.get_ylim()[0]-50, ymax=cax.get_ylim()[1]+50, color='k', linewidth=2.5)
        fig.colorbar(img, cax=ax[4, 1], ticks=[np.min(rdm_epoch_spect), np.max(rdm_epoch_spect)])

    # Standardize
    y_mean = np.mean(y, axis=1, keepdims=True)
    y_std = np.std(y, axis=1, keepdims=True)

    y = (y - y_mean) / y_std

    if visualize:
        cax = ax[5, 0]

        rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]

        img = cax.imshow(rdm_epoch_spect, cmap='jet', aspect='auto')
        cax.set_title('Step 7: Standardization', fontsize=15)
        cax.tick_params(labelsize=13)
        cax.invert_yaxis()
        cax.set_xlabel('Time (s)', fontsize=13)
        cax.set_ylabel('Freq. (Hz)', fontsize=13)
        cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
        cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
        # cax.get_xaxis().set_visible(False)
        cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
        cax.set_yticklabels(['24', '12', '0.5'])
        cax.set_xticklabels(['0', '4', '8', '12', '16', '20'])
        cax.set_ylim(cax.get_ylim())
        cax.vlines(x=cax.get_xticks()[1:-1],
                   ymin=cax.get_ylim()[0]-50, ymax=cax.get_ylim()[1]+50, color='k', linewidth=2.5)
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

    if visualize:

        # nice_epoch = False
        # Select random epoch
        rdm_epoch_idx = np.random.randint(2, len(signal) / 4 / fs - 2)
        rdm_epoch_idx = 3419
        rdm_epoch_labels = labels.to_numpy()[rdm_epoch_idx - 2:rdm_epoch_idx + 3, :]


        if plot_artifacts:
            labels_dict = {0: 'NREM', 1: 'REM', 2: 'WAKE', 3: 'ART'}
        else:
            labels_dict = {0: 'NREM', 1: 'REM', 2: 'WAKE'}
        rdm_epoch_labels = np.where(rdm_epoch_labels == 1)[1]
        rdm_epoch_labels = [labels_dict[i] for i in rdm_epoch_labels]


        rdm_epoch_signal = signal[(rdm_epoch_idx - 2) * fs * 4: (rdm_epoch_idx + 3) * fs * 4]
        time_axis = np.linspace((rdm_epoch_idx - 2) * fs * 4, (rdm_epoch_idx + 3) * fs * 4, fs * 4 * 5) / fs

        fig, ax = plt.subplots(7, 2, figsize=(15, 10), gridspec_kw={'width_ratios': [60, 1]})
        fig.subplots_adjust(hspace=0.8)
        cax = ax[0, 0]
        cax.plot(time_axis, rdm_epoch_signal*1000000, zorder=0)
        cax.tick_params(labelsize=13)
        cax.set_ylim((-cax.get_ylim()[1], cax.get_ylim()[1]))
        cax.vlines(x=np.linspace(time_axis[0] + 4, time_axis[-1] - 4, 4),
                   ymin=cax.get_ylim()[0]+1, ymax=cax.get_ylim()[1]-1, color='k', linewidth=2.5)
        cax.set_title('Step 0: Raw 5 epochs window', fontsize=15)
        # cax.set_xlabel('Time (s)')
        cax.set_ylabel(r'Voltage ($\mu$V)', fontsize=13)
        cax.set_xticks(np.linspace(time_axis[0], time_axis[-1], 6))
        cax.set_xlim((time_axis[0], time_axis[-1]))
        cax.get_xaxis().set_visible(False)
        epoch_labels_ax = cax.twiny()
        epoch_labels_ax.set_xlim(cax.get_xlim())
        epoch_labels_ax.set_xticks(np.linspace(time_axis[0]+ 2, time_axis[-1]-2, 5))
        epoch_labels_ax.set_xticklabels(rdm_epoch_labels, fontsize=13)
        epoch_labels_ax.tick_params(length=0)
        epoch_labels_ax.get_xticklabels()[2].set_fontsize(15)
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
        cax.set_title('Step 2: Spectrogram', fontsize=15)
        cax.tick_params(labelsize=13)
        cax.invert_yaxis()
        # cax.set_xlabel('Time (s)')
        cax.set_ylabel('Freq. (Hz)', fontsize=13)
        cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
        cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
        cax.get_xaxis().set_visible(False)
        cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
        cax.set_yticklabels(['64', '32', '0'])
        # cax.set_yticklabels([str(f[-1]), str(f[-1] / 2), str(f[0])])
        cax.set_ylim(cax.get_ylim())
        cax.vlines(x=cax.get_xticks()[1:-1],
                   ymin=cax.get_ylim()[0]-50, ymax=cax.get_ylim()[1]+50, color='k', linewidth=2.5)
        fig.colorbar(img, cax=ax[1, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])

    # Bandpass (crop)
    Z = Z[np.where(f == lowcut)[0][0]: np.where(f == highcut)[0][0] + 1, :]

    if visualize:
        cax = ax[2, 0]

        rdm_epoch_spect = Z[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]

        img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
        cax.set_title('Step 3: Frequency range selection', fontsize=15)
        cax.tick_params(labelsize=13)
        cax.invert_yaxis()
        # cax.set_xlabel('Time (s)')
        cax.set_ylabel('Freq. (Hz)', fontsize=13)
        cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
        cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
        cax.get_xaxis().set_visible(False)
        cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
        cax.set_yticklabels(['30', '15', '0.5'])
        cax.set_ylim(cax.get_ylim())
        cax.vlines(x=cax.get_xticks()[1:-1],
                   ymin=cax.get_ylim()[0]-50, ymax=cax.get_ylim()[1]+50, color='k', linewidth=2.5)
        fig.colorbar(img, cax=ax[2, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])

    # PSD
    y = np.abs(Z) ** 2

    if visualize:
        cax = ax[3, 0]

        rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]

        img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
        cax.set_title('Step 4: PSD', fontsize=15)
        cax.tick_params(labelsize=13)
        cax.invert_yaxis()
        # cax.set_xlabel('Time (s)')
        cax.set_ylabel('Freq. (Hz)', fontsize=13)
        cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
        cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
        cax.get_xaxis().set_visible(False)
        cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
        cax.set_yticklabels(['30', '15', '0.5'])
        cax.set_ylim(cax.get_ylim())
        cax.vlines(x=cax.get_xticks()[1:-1],
                   ymin=cax.get_ylim()[0]-50, ymax=cax.get_ylim()[1]+50, color='k', linewidth=2.5)
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
        cax.set_title('Step 5: Integration', fontsize=15)
        cax.tick_params(labelsize=13)
        cax.invert_yaxis()
        # cax.set_xlabel('Time (s)')
        cax.set_ylabel('Freq. (Hz)', fontsize=13)
        cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
        cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
        cax.get_xaxis().set_visible(False)
        cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
        cax.set_yticklabels(['30', '15', '0.5'])
        cax.set_ylim(cax.get_ylim())
        cax.vlines(x=cax.get_xticks()[1:-1],
                   ymin=cax.get_ylim()[0]-50, ymax=cax.get_ylim()[1]+50, color='k', linewidth=2.5)
        fig.colorbar(img, cax=ax[4, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])

    # Log-scale
    y = 10*np.log10(y)

    if visualize:
        cax = ax[5, 0]

        rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]

        img = cax.imshow(rdm_epoch_spect, cmap='jet', aspect='auto')
        cax.set_title('Step 6: Log transformation', fontsize=15)
        cax.tick_params(labelsize=13)
        cax.invert_yaxis()
        # cax.set_xlabel('Time (s)')
        cax.set_ylabel('Freq. (Hz)', fontsize=13)
        cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
        cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
        cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
        cax.set_yticklabels(['30', '15', '0.5'])
        cax.set_ylim(cax.get_ylim())
        cax.get_xaxis().set_visible(False)
        cax.vlines(x=cax.get_xticks()[1:-1],
                   ymin=cax.get_ylim()[0]-50, ymax=cax.get_ylim()[1]+50, color='k', linewidth=2.5)
        fig.colorbar(img, cax=ax[5, 1], ticks=[np.min(rdm_epoch_spect), np.max(rdm_epoch_spect)])

    # Standardize
    y_mean = np.mean(y, axis=1, keepdims=True)
    y_std = np.std(y, axis=1, keepdims=True)

    y = (y - y_mean) / y_std

    if visualize:
        cax = ax[6, 0]

        rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]

        img = cax.imshow(rdm_epoch_spect, cmap='jet', aspect='auto')
        cax.set_title('Step 7: Standardization', fontsize=15)
        cax.tick_params(labelsize=13)
        cax.invert_yaxis()
        cax.set_xlabel('Time (s)')
        cax.set_ylabel('Freq. (Hz.)', fontsize=13)
        cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
        cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
        # cax.get_xaxis().set_visible(False)
        cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
        cax.set_yticklabels(['30', '15', '0.5'])
        cax.set_xticklabels(['0', '4', '8', '12', '16', '20'])
        cax.set_xlabel('Time (s)', fontsize=13)
        cax.set_ylim(cax.get_ylim())
        cax.vlines(x=cax.get_xticks()[1:-1],
                   ymin=cax.get_ylim()[0]-50, ymax=cax.get_ylim()[1]+50, color='k', linewidth=2.5)
        fig.colorbar(img, cax=ax[6, 1], ticks=[np.min(rdm_epoch_spect), np.max(rdm_epoch_spect)])
        plt.show()

    return y


def load_recording(signal_path,
                   labels_path,
                   resample_rate,
                   just_artifact_labels,
                   just_stage_labels,
                   validation_split):
    y = load_labels(labels_path,
                    just_artifact_labels=just_artifact_labels,
                    just_stage_labels=just_stage_labels)

    raw_data = load_raw_recording(signal_path, resample_rate)
    # eeg_1 = preprocess_EEG(raw_data[0, :],
    #                        labels=y,
    #                        visualize=True)
    # eeg_2 = preprocess_EEG(raw_data[1, :],
    #                        labels=y,
    #                        visualize=True)
    emg = preprocess_EMG(raw_data[2, :],
                         labels=y,
                         visualize=True)
    x = np.stack((eeg_1, eeg_2, emg))
    x = windowing(x, window_size=32 * 5, window_stride=32)
    x = np.transpose(x, (0, 3, 2, 1))

    y = load_labels(labels_path,
                    just_artifact_labels=just_artifact_labels,
                    just_stage_labels=just_stage_labels)
    y = y.to_numpy()

    if x.shape[0] != y.shape[0]:
        print(signal_path)
        raise Exception("x and y don't have same number of epochs")

    if validation_split != 0:
        rdm_indexes = np.arange(x.shape[0])
        np.random.shuffle(rdm_indexes)
        train_indexes = rdm_indexes[:int(len(rdm_indexes) * (1 - validation_split))]
        val_indexes = rdm_indexes[int(len(rdm_indexes) * (1 - validation_split)):]

        x_train = x[train_indexes]
        labels_train = y[train_indexes]
        # save_to_numpy(x_train, labels_train,
        #               r'C:\Users\javig\Desktop\SPINDLE dataset\SPINDLE dataset\preprocessed\d5\training')
        x_val = x[val_indexes]
        labels_val = y[val_indexes]
        # save_to_numpy(x_val, labels_val,
        #               r'C:\Users\javig\Desktop\SPINDLE dataset\SPINDLE dataset\preprocessed\d1\validation')

        return x_train, x_val, labels_train, labels_val

    else:
        # save_to_numpy(x, y, r'C:\Users\javig\Desktop\SPINDLE dataset\SPINDLE dataset\preprocessed\d1\testing\Cohort A')
        return x, y


for i in range(len(test_signals)):
    print('Processing file ', file_counter)
    print('Remaning files: ', number_of_files - file_counter)

    x, y = load_recording(dataset_folder + os.sep + test_signals[i],
                          dataset_folder + os.sep + test_labels[i],
                          resample_rate=128,
                          just_artifact_labels=False,
                          just_stage_labels=False,
                          validation_split=0)
