'''
Cleans the recordings that are somehow corrupted
'''

import pandas as pd
import os
import mne
import pathlib

clean_data_path = r'C:\Users\javig\Documents\Kornum lab data\Laura-EEGdata (cleaned)\data-Kornum'

# CASE 1: FILES WHERE THE SIGNAL IS LONGER THAN THE LABELS:
# I checked that it works properly

labels = [r"C:\Users\javig\Documents\Kornum lab data\Laura-EEGdata (original)\data-Kornum\2DTUSERVER-Alexandra\tsv\M23-b3.tsv",
          r"C:\Users\javig\Documents\Kornum lab data\Laura-EEGdata (original)\data-Kornum\2DTUSERVER-Alexandra\tsv\M29-b3.tsv",
          r"C:\Users\javig\Documents\Kornum lab data\Laura-EEGdata (original)\data-Kornum\2DTUSERVER-Alexandra\tsv\M52-b3.tsv",
          r"C:\Users\javig\Documents\Kornum lab data\Laura-EEGdata (original)\data-Kornum\2DTUSERVER-Alexandra\tsv\M58-b3.tsv"]


for i in range(len(labels)):
    l = labels[i]

    tsv_data = pd.read_csv(l, skiprows=9, engine='python', sep='\t', index_col=False).iloc[:, 4]

    edf_path = os.path.join(pathlib.Path.home().drive, os.sep, *l.split(os.sep)[:-2], 'EDF',
                            l.split(os.sep)[-1][:-3] + 'edf')
    edf_signal = mne.io.read_raw_edf(edf_path)
    sfreq = edf_signal.info['sfreq']
    raw_signal = edf_signal.get_data()

    n_epochs_label = len(tsv_data)
    # n_epochs_signal = raw_signal.shape[1] / sfreq / 4
    # print(n_epochs_label)
    # print(n_epochs_signal)

    edf_signal.crop(tmax=len(tsv_data) * 4, include_tmax=False)

    n_epochs_signal = edf_signal.get_data().shape[1] / sfreq / 4
    print(n_epochs_label)
    print(n_epochs_signal)

    signal_path = os.path.join(clean_data_path, *edf_path.split(os.sep)[-3:])
    label_path = os.path.join(clean_data_path, *l.split(os.sep)[-3:])

    # signal_path = 'C:/Users/javig/Desktop/cleaned data/' + edf_path.split(os.sep)[-1]
    # label_path = 'C:/Users/javig/Desktop/cleaned data/' + l.split(os.sep)[-1]

    edf_signal.export(signal_path, overwrite=True)


# CASE 2: FILES WHERE UNSCORED LABELS NEED TO DISCARDED FROM THE END OF THE SIGNAL (AND THE LABELS)
# I checked that it works properly

labels = [r"C:\Users\javig\Documents\Kornum lab data\Laura-EEGdata (original)\data-Kornum\2DTUSERVER-LOUISE\tsv\M20-b3.tsv",
          r"C:\Users\javig\Documents\Kornum lab data\Laura-EEGdata (original)\data-Kornum\2DTUSERVER-LOUISE\tsv\M309-b1.tsv",
          r"C:\Users\javig\Documents\Kornum lab data\Laura-EEGdata (original)\data-Kornum\2DTUSERVER-LOUISE\tsv\M313-b1.tsv"]


for i in range(len(labels)):

    l = labels[i]

    tsv_data = pd.read_csv(l, skiprows=9, engine='python', sep='\t', index_col=False).iloc[:, 4]

    edf_path = os.path.join(pathlib.Path.home().drive, os.sep, *l.split(os.sep)[:-2], 'EDF',
                            l.split(os.sep)[-1][:-3] + 'edf')
    edf_signal = mne.io.read_raw_edf(edf_path)
    sfreq = edf_signal.info['sfreq']
    raw_signal = edf_signal.get_data()

    # n_epochs_label = len(tsv_data)
    # n_epochs_signal = raw_signal.shape[1] / sfreq / 4

    index_first = tsv_data[tsv_data == 255].index[0]

    edf_signal.crop(tmax=index_first * 4, include_tmax=False)
    tsv_data_2 = pd.read_csv(l, header=None, skipfooter=len(tsv_data)-index_first, engine='python')

    # n_epochs_label = len(tsv_data_2) - 9
    # n_epochs_signal = edf_signal.get_data().shape[1] / sfreq / 4
    # print(n_epochs_label)
    # print(n_epochs_signal)

    signal_path = os.path.join(clean_data_path, *edf_path.split(os.sep)[-3:])
    label_path = os.path.join(clean_data_path, *l.split(os.sep)[-3:])

    # signal_path = 'C:/Users/javig/Desktop/cleaned data/' + edf_path.split(os.sep)[-1]
    # label_path = 'C:/Users/javig/Desktop/cleaned data/' + l.split(os.sep)[-1]

    edf_signal.export(signal_path, overwrite=True)
    tsv_data_2.to_csv(label_path, header=False, index=False)


# CASE 3:FILES WHERE UNSCORED LABELS NEED TO DISCARDED FROM THE BEGINNING OF THE SIGNAL (AND THE LABELS)
# I checked that it works properly

labels = [r"C:\Users\javig\Documents\Kornum lab data\Laura-EEGdata (original)\data-Kornum\2DTUSERVER-Maria\tsv\m86-b1.tsv",
          r"C:\Users\javig\Documents\Kornum lab data\Laura-EEGdata (original)\data-Kornum\2DTUSERVER-Maria\tsv\m88-b1.tsv",
          r"C:\Users\javig\Documents\Kornum lab data\Laura-EEGdata (original)\data-Kornum\2DTUSERVER-Maria\tsv\m94-b1.tsv",
          r"C:\Users\javig\Documents\Kornum lab data\Laura-EEGdata (original)\data-Kornum\2DTUSERVER-Maria\tsv\m96-b1.tsv"]


for i in range(len(labels)):

    l = labels[i]

    tsv_data = pd.read_csv(l, skiprows=9, engine='python', sep='\t', index_col=False).iloc[:, 4]

    edf_path = os.path.join(pathlib.Path.home().drive, os.sep, *l.split(os.sep)[:-2], 'EDF',
                            l.split(os.sep)[-1][:-3] + 'edf')
    edf_signal = mne.io.read_raw_edf(edf_path)
    sfreq = edf_signal.info['sfreq']
    raw_signal = edf_signal.get_data()

    # n_epochs_label = len(tsv_data)
    # n_epochs_signal = raw_signal.shape[1] / sfreq / 4

    index_last = tsv_data[tsv_data == 255].index[-1]

    edf_signal.crop(tmin=(index_last+1) * 4)
    tsv_data_2 = pd.read_csv(l, engine='python', header=None).iloc[:10]
    tsv_data_3 = pd.read_csv(l, skiprows=12+index_last, engine='python', index_col=False, header=None)
    tsv_data_4 = pd.concat((tsv_data_2, tsv_data_3), axis=0)

    # n_epochs_label = len(tsv_data_4)-10
    # n_epochs_signal = edf_signal.get_data().shape[1] / sfreq / 4
    # print(n_epochs_label)
    # print(n_epochs_signal)

    signal_path = os.path.join(clean_data_path, *edf_path.split(os.sep)[-3:])
    label_path = os.path.join(clean_data_path, *l.split(os.sep)[-3:])

    # signal_path = 'C:/Users/javig/Desktop/cleaned data/' + edf_path.split(os.sep)[-1]
    # label_path = 'C:/Users/javig/Desktop/cleaned data/' + l.split(os.sep)[-1]

    edf_signal.export(signal_path, overwrite=True)
    tsv_data_4.to_csv(label_path, header=False, index=False)


# CASE 4: FILES WHERE THE SIGNAL IS LONGER THAN THE LABELS AND THE UNSCORED LABELS NEED TO DISCARDED FROM THE END OF THE SIGNAL (AND THE LABELS)
# Is like case 2 above actually, but there is an unscored label in the middle of the signal so the code is different in line

labels = [r"C:\Users\javig\Documents\Kornum lab data\Laura-EEGdata (original)\data-Kornum\2DTUSERVER-Alexandra\tsv\M48-b3.tsv"]

for i in range(len(labels)):

    l = labels[i]

    tsv_data = pd.read_csv(l, skiprows=9, engine='python', sep='\t', index_col=False).iloc[:, 4]

    edf_path = os.path.join(pathlib.Path.home().drive, os.sep, *l.split(os.sep)[:-2], 'EDF',
                            l.split(os.sep)[-1][:-3] + 'edf')
    edf_signal = mne.io.read_raw_edf(edf_path)
    sfreq = edf_signal.info['sfreq']
    raw_signal = edf_signal.get_data()

    n_epochs_label = len(tsv_data)
    n_epochs_signal = raw_signal.shape[1] / sfreq / 4
    print(n_epochs_label)
    print(n_epochs_signal)

    index_first = tsv_data[tsv_data == 255].index[1] # index is 1 because there is a single unscored label in the middle of the signal

    edf_signal.crop(tmax=index_first * 4, include_tmax=False)
    tsv_data_2 = pd.read_csv(l, header=None, skipfooter=len(tsv_data)-index_first, engine='python')

    # n_epochs_label = len(tsv_data_2) - 10
    # n_epochs_signal = edf_signal.get_data().shape[1] / sfreq / 4
    # print(n_epochs_label)
    # print(n_epochs_signal)

    signal_path = os.path.join(clean_data_path, *edf_path.split(os.sep)[-3:])
    label_path = os.path.join(clean_data_path, *l.split(os.sep)[-3:])

    # signal_path = 'C:/Users/javig/Desktop/cleaned data/' + edf_path.split(os.sep)[-1]
    # label_path = 'C:/Users/javig/Desktop/cleaned data/' + l.split(os.sep)[-1]

    edf_signal.export(signal_path, overwrite=True)
    tsv_data_2.to_csv(label_path, header=False, index=False)