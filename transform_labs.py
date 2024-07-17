#!/usr/bin/env python

import argparse
import sys
import time
from glob import glob
import os
from os.path import join, basename, isfile, realpath, dirname
import mne 
import h5py
import numpy as np
import tables
import scipy.io
import pandas as pd

sys.path.insert(0, realpath(join(dirname(__file__), '..')))

from base.data.downsample import downsample
from base.config_loader import ConfigLoader
from base.data.data_table import create_table_description , COLUMN_LABEL, COLUMN_MOUSE_ID, COLUMN_LAB


def parse():
    parser = argparse.ArgumentParser(description='data transformation script')
    parser.add_argument('--experiment', '-e', required=False, default='standard_config',
                        help='name of experiment to transform data to')

    return parser.parse_args()


def preprocess_EEG(signal): 

    lowcut      = 0.5
    highcut     = 24
    stft_stride = 16
    stft_size   = 256
  
    f, t, Z = scipy.signal.stft(signal,
                                fs=128,
                                window='hamming',
                                nperseg=stft_size,
                                noverlap=stft_size - stft_stride
                                )
  
    Z = Z[np.where(f == lowcut)[0][0]: np.where(f == highcut)[0][0] + 1, :]
    
    # PSD
    y = np.abs(Z) ** 2

    # Log-scale
    zero_elements = np.where(y == 0)
    if zero_elements[0].size > 0:
        print(f"Zero elements found at indices: {zero_elements}")
        epsilon = 1e-12
        y = y + epsilon

    y = 10 * np.log10(y)

    # Standardize
    y_mean = np.mean(y, axis=1, keepdims=True)
    y_std  = np.std(y, axis=1, keepdims=True)
    y      = (y - y_mean) / y_std

    return y 

def preprocess_EMG(signal): 

    stft_size=256
    stft_stride=16
    lowcut=0.5
    highcut=30

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

    zero_elements = np.where(y == 0)
    if zero_elements[0].size > 0:
        print(f"Zero elements found at indices: {zero_elements}")
        epsilon = 1e-12
        y = y + epsilon
    # Log-scale
    y = 10*np.log10(y)

    # Standardize
    y_mean = np.mean(y, axis=1, keepdims=True)
    y_std = np.std(y, axis=1, keepdims=True)

    y = (y - y_mean) / y_std

    return y

def windowing(signal, window_size=32*5, window_stride=32, fs=128):
    n_windows = np.floor((signal.shape[2] - window_size) / window_stride).astype(int)+1
    # if ((n_windows - 1 + 1)*window_stride + window_size) < signal.shape[2]:
    #     n_windows += 1

    windowed_signal = np.zeros((n_windows, 3, 48, window_size))

    # signal = signal[window_size//2 : -window_size//2],

    for i in range(n_windows):
        windowed_signal[i, :, :, :] = signal[:, :, (i*window_stride) : (i*window_stride) + window_size]

    return windowed_signal
    

def read_recording(mat_path: str, sr_old: int, labs_channels: dict, lab_name: str):
    rec = scipy.io.loadmat(mat_path)['signal']

    features = {}
    s_       = {} 

    for channel in labs_channels[lab_name].keys():
        signal = rec[labs_channels[lab_name][channel],:]

        s = downsample(signal,
                       sr_old=sr_old,
                       sr_new=config.SAMPLING_RATE,
                       fmax=0.4 * config.SAMPLING_RATE,
                       outtype='sos',
                       method='pad')
        
        s_[channel] = s
        #s = (s - np.mean(s, keepdims=True)) / np.std(s, keepdims=True)
    
    if lab_name=='Maiken' or lab_name=='Alessandro':
        eeg_1 = preprocess_EEG(s_["EEG1"]) # eeg1 
        eeg_2 = preprocess_EEG(s_["EEG1"]) # eeg2
        emg   = preprocess_EMG(s_["EMG"]) # emg
    
    elif lab_name=='Kornum' or lab_name=='Antoine'or lab_name=='mus' or lab_name=='Sebastian': 
        eeg_1 = preprocess_EEG(s_["EEG1"]) # eeg1 
        eeg_2 = preprocess_EEG(s_["EEG2"]) # eeg2
        emg   = preprocess_EMG(s_["EMG"]) # emg
 
    x = np.stack((eeg_1, eeg_2, emg))
    x = windowing(x, window_size=32 * 5, window_stride=32)
    x = np.transpose(x, (0, 3, 2, 1))

    features["set1"] = x

    #sample_start_times = np.arange(0, features[channel].shape[0], config.SAMPLING_RATE*config.SAMPLE_DURATION, dtype=int)
    sample_start_times = np.arange(0, features["set1"].shape[0], 1, dtype=int)

    return features, sample_start_times.tolist()


def read_labels(labels_path: str): # tjek om disse er korrekte 
    labels_correspondence = {1: 'Wake',
                            2: 'Non REM',
                            3: 'REM',
                            4: 'Art'}
        
    if labels_path.split(os.sep)[-1][-3:] == 'mat':
        labels = scipy.io.loadmat(labels_path)[os.path.basename(labels_path).split('.')[0]]
    elif labels_path.split(os.sep)[-1][-3:] == 'csv':
        labels = pd.read_csv(labels_path, header=None, index_col=0).iloc[:,0].to_numpy()
        if not isinstance(labels[0], int): 
            labels = labels[1:].astype(float)

    labels_str = [labels_correspondence[l] for l in labels.squeeze()]
    assert len(np.unique(labels))<=4 
    # drop the first and last two 
    labels_str = labels_str[2:-2].copy() # Drop 2 first and 2 last epochs

    return labels_str

def write_data_to_table(table: tables.Table, features: dict, labels: list, start_times: list, mouse_id: int, lab: str, labs_channels: dict):
    """writes given data to the passed table, each sample is written in a new row"""
    sample = table.row

    # iterate over samples and create rows
    for sample_start, label in zip(start_times, labels):
        # determine idxs of data to load from features
        #sample_end = int(sample_start + config.SAMPLE_DURATION * config.SAMPLING_RATE)
        # try to load data from sample_start to sample_end, if there is not enough data, ignore the sample
        try:
            sample[COLUMN_MOUSE_ID] = mouse_id
            sample[COLUMN_LAB] = lab
            sample[COLUMN_LABEL] = label
            sample["x"] = features["set1"][sample_start,:,:,:]
            sample.append()
        except ValueError:
            print(f"""
            While processing epoch [{sample_start}] with label {label}:
            not enough datapoints in file (n = {len(list(features.values())[0])})
            This epoch is ignored.
            """)
    # write data to table
    table.flush()

def transform():
    """transform files in DATA_DIR to pytables table"""
    # load description of table columns
    table_desc = create_table_description(config)

    # if the transformed data file already exists, ask the user if he wants to overwrite it
    #if isfile(config.DATA_FILE):
        #question = f"{realpath(config.DATA_FILE)} already exists, do you want to override? (y/N): "
        #response = input(question)
        #if response.lower() != 'y':
          #  exit()

    print(f'data is loaded from {realpath(config.DATA_DIR)}')

    # open pytables DATA_FILE
    with tables.open_file(config.DATA_FILE, mode='w', title='data from multiple labs') as f:

        # create tables for every dataset
        table = f.create_table(f.root, 'multiple_labs', table_desc, 'multiple_labs_data')

        # determine which files to transform for each dataset based on DATA_SPLIT
        labs = [f for f in os.listdir(config.DATA_DIR) if not f.startswith('.') and 'MACOSX' not in f]
        # iterate over files, load them and write them to the created table
        for l in labs:
            #l = labs[3]

            lab_name = l.split('-')[0]
            
            mice = [f for f in os.listdir(os.path.join(config.DATA_DIR, l)) if not f.startswith('.') and 'MACOSX' not in f]

            for m in mice:
                #m = mice[20] 
                
                try:
                    assert os.path.isfile(os.path.join(config.DATA_DIR, l, m, 'cleaned', 'signal.mat'))
                except:
                    print(os.path.join(l, m), ' does not exist')
                    continue

                if os.path.isfile(os.path.join(config.DATA_DIR, l, m, 'cleaned', 'fs.mat')):
                    original_fs = scipy.io.loadmat(os.path.join(config.DATA_DIR, l, m, 'cleaned', 'fs.mat'))['fs'][0][0]
                else:
                    original_fs = config.ORIGINAL_FS[lab_name]
        
                print('mouse {:s}'.format(os.path.join(l, m)))
                start = time.time()

                if lab_name == 'Kornum':
                    labels_name = 'labels.csv'
                elif lab_name == 'Maiken':
                    labels_name = 'newSS.mat'
                else:
                    labels_name = 'labels.mat'

                features, times = read_recording(os.path.join(config.DATA_DIR, l, m, 'cleaned', 'signal.mat'), original_fs, config.LABS_CHANNELS, lab_name)
                labels = read_labels(os.path.join(config.DATA_DIR, l, m, 'cleaned', labels_name))

                # write loaded data to table
                assert len(labels)==features["set1"].shape[0]
                write_data_to_table(table, features, labels, times, m, lab_name, config.LABS_CHANNELS)

                print('execution time: {:.2f}'.format(time.time() - start))
                print()

        print(f)


if __name__ == '__main__':
    args = parse()
    # load config, dirs are not needed here, because we do not write a log
    config = ConfigLoader(experiment=args.experiment, create_dirs=False)

    # transform files
    transform()
