import mne
import scipy
import scipy.signal
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Input, MaxPool2D, Conv2D, Dense, Softmax, Flatten
import random


# Creating windows of 5 epochs from raw signal, then doing the spectrogram for each of the windows
# A = with scipy stft

# plt.ion()

EPOCH_LENGTH = 4  # seconds
fs = 128  # Hz.
stft_stride = 16  # samples

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


def windowing(signal, window_size=512, window_stride=512, fs=128):
    n_windows = signal.size // window_stride - 2 * (window_size / window_stride // 2)

    windowed_signal = np.zeros((int(n_windows), window_size))

    # signal = signal[window_size//2 : -window_size//2],

    for i in range(int(n_windows)):
        windowed_signal[i, :] = signal[i * window_stride: i * window_stride + window_size]

    return windowed_signal


labels_path = 'C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data/CohortA/scorings/A1.csv'
y_1, y_2 = load_labels(labels_path)


# -------------------------------- option 1: do spectrogram for each window (too memory consuming) --------------------------------
def preprocess_EEG(signal, fs=128, stft_size=256, stft_stride=16, lowcut=0.5, highcut=24):
    # STFT
    f, t, Z = scipy.signal.stft(signal,
                                fs=128,
                                window='hamming',
                                nperseg=stft_size,
                                noverlap=stft_size - stft_stride,
                                boundary=None
                                # padded=False
                                )
    # Bandpass (crop)
    Z = Z[np.where(f == lowcut)[0][0]: np.where(f == highcut)[0][0] + 1, :]

    # PSD
    y = np.abs(Z) ** 2

    # Log-scale
    y = 10*np.log10(y)

    # Standardize
    y_mean = np.mean(y, axis=1)
    y_std = np.std(y, axis=1)

    # y = (y - np.expand_dims(y_mean, axis=1)) / np.expand_dims(y_std, axis=1)

    return y


def preprocess_EMG(signal, fs=128, stft_size=256, stft_stride=16, lowcut=0.5, highcut=30):
    # STFT
    f, t, Z = scipy.signal.stft(signal,
                                fs=128,
                                window='hamming',
                                nperseg=stft_size,
                                noverlap=stft_size - stft_stride,
                                boundary=None
                                # padded=False
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
    y_mean = np.mean(y, axis=1)
    y_std = np.std(y, axis=1)

    # y = (y - np.expand_dims(y_mean, axis=1)) / np.expand_dims(y_std, axis=1)

    return y

def load_and_preprocess_data_1(file_path):
    raw_data = load_raw_recording(file_path)

    eeg1_w = windowing(raw_data[0, :])
    eeg2_w = windowing(raw_data[1, :])
    emg_w = windowing(raw_data[2, :])

    wrt = np.zeros((21600, 48, 17, 3))
    for i in range(eeg1_w.shape[0]):
        wrt[i, :, :, 0] = preprocess_EEG(eeg1_w[i, :])
        wrt[i, :, :, 1] = preprocess_EEG(eeg2_w[i, :])
        wrt[i, :, :, 2] = preprocess_EMG(emg_w[i, :])

    input_dataset = tf.keras.utils.timeseries_dataset_from_array(
        data=wrt,
        targets=None,
        sequence_length=5,
        sequence_stride=1,
        batch_size=None)
    # input_dataset = input_dataset.batch(32)

    return input_dataset

input_dataset = load_and_preprocess_data_1(file_path)

# -------------------------------- option 2: do spectrogram on tf dataset and transform it --------------------------------

# def get_eeg_spectrogram(waveform, fs=128, stft_size=256, stft_stride=16, lowcut=0.5, highcut=24):
#     # STFT
#     Z = tf.signal.stft(
#         waveform, frame_length=stft_size, frame_step=stft_stride, window_fn=tf.signal.hamming_window)
#
#     f = np.linspace(0, 64, 129)
#
#     # Bandpass (crop)
#     Z = Z[:, :, np.where(f == lowcut)[0][0]: np.where(f == highcut)[0][0] + 1]
#
#     # PSD
#     y = tf.math.abs(Z)**2
#
#     # Log-scale
#     y = 10*tf.math.log(y)
#
#     # Standardize
#     y_mean = tf.math.reduce_mean(y, axis=1)
#     y_std = tf.math.reduce_std(y, axis=1)
#
#     y = (y - tf.expand_dims(y_mean, axis=1)) / tf.expand_dims(y_std, axis=1)
#
#     # CONVENTION: low frequencies are at the top. Need to reverse Y-axis when plotting
#
#     return y
#
# def get_eeg_spectrogram_no_batch(waveform, fs=128, stft_size=256, stft_stride=16, lowcut=0.5, highcut=24):
#     # STFT
#     f, t, Z = scipy.signal.stft(waveform,
#                                 fs=128,
#                                 window='hamming',
#                                 nperseg=stft_size,
#                                 noverlap=stft_size - stft_stride
#                                 )
#
#     # Bandpass (crop)
#     Z = Z[np.where(f == lowcut)[0][0]: np.where(f == highcut)[0][0] + 1, :]
#
#     # PSD
#     y = np.abs(Z) ** 2
#
#     # Log-scale
#     y = 10*np.log10(y)
#
#     # Standardize
#     y_mean = np.mean(y, axis=1)
#     y_std = np.std(y, axis=1)
#
#     y = (y - np.expand_dims(y_mean, axis=1)) / np.expand_dims(y_std, axis=1)
#
#     # CONVENTION: low frequencies are at the top. Need to reverse Y-axis when plotting
#
#     return y
#
# def get_emg_spectrogram(waveform, fs=128, stft_size=256, stft_stride=16, lowcut=0.5, highcut=30):
#     # STFT
#     Z = tf.signal.stft(
#         waveform, frame_length=stft_size, frame_step=stft_stride, window_fn=tf.signal.hamming_window)
#
#     f = np.linspace(0, 64, 129)
#
#     # Bandpass (crop)
#     Z = Z[:, :, np.where(f == lowcut)[0][0]: np.where(f == highcut)[0][0] + 1]
#
#     # PSD
#     y = tf.math.abs(Z)**2
#
#     # Integration
#     y = tf.reduce_sum(y, axis=2)
#
#     # Stack rows to have 2 dimensions
#     y = tf.expand_dims(y, axis=2)
#     # y = np.repeat(y, eeg_dimensions[0], axis=0)
#     y = tf.repeat(y, 48, axis=2)
#
#     # Log-scale
#     # y = 10*tf.math.log(y)
#     y = 10*tf.experimental.numpy.log10(y)
#
#     # Standardize
#     y_mean = tf.math.reduce_mean(y, axis=1)
#     y_std = tf.math.reduce_std(y, axis=1)
#
#     y = (y - tf.expand_dims(y_mean, axis=1)) / tf.expand_dims(y_std, axis=1)
#
#     # CONVENTION: low frequencies are at the top. Need to reverse Y-axis when plotting
#
#     return y
#
# def get_emg_spectrogram_no_batch(waveform, fs=128, stft_size=256, stft_stride=16, lowcut=0.5, highcut=30):
#     # STFT
#     f, t, Z = scipy.signal.stft(signal,
#                                 fs=128,
#                                 window='hamming',
#                                 nperseg=stft_size,
#                                 noverlap=stft_size - stft_stride
#                                 )
#
#     # Bandpass (crop)
#     Z = Z[np.where(f == lowcut)[0][0]: np.where(f == highcut)[0][0] + 1, :]
#
#     # PSD
#     y = np.abs(Z) ** 2
#
#     # Integration
#     y = np.sum(y, axis=0)
#
#     # Stack rows to have 2 dimensions
#     y = np.expand_dims(y, axis=0)
#     # y = np.repeat(y, eeg_dimensions[0], axis=0)
#     y = np.repeat(y, 48, axis=0)
#
#     # Log-scale
#     y = 10*np.log10(y)
#
#     # Standardize
#     y_mean = np.mean(y, axis=1)
#     y_std = np.std(y, axis=1)
#
#     y = (y - np.expand_dims(y_mean, axis=1)) / np.expand_dims(y_std, axis=1)
#
#     return y
#
# def load_and_preprocess_data_2(file_path):
#     raw_data = load_raw_recording(file_path)
#
#     eeg1_w = windowing(raw_data[0, :])
#     eeg2_w = windowing(raw_data[1, :])
#     emg_w = windowing(raw_data[2, :])
#
#     input_dataset_eeg1 = tf.data.Dataset.from_tensor_slices(eeg1_w)  # start_index=int(wp/2), targets=y_1[2:-2],
#     # input_dataset_eeg1 = input_dataset_eeg1.batch(32)
#     input_dataset_eeg2 = tf.data.Dataset.from_tensor_slices(eeg2_w)  # start_index=int(wp/2), targets=y_1[2:-2],
#     # input_dataset_eeg2 = input_dataset_eeg2.batch(32)
#     input_dataset_emg = tf.data.Dataset.from_tensor_slices(emg_w)  # start_index=int(wp/2), targets=y_1[2:-2],
#     # input_dataset_emg = input_dataset_emg.batch(32)
#
#     # for b in input_dataset_emg:
#     #     get_eeg_spectrogram_no_batch(b)
#
#     def make_spec_ds(ds, signal_type):
#         if signal_type == 'eeg':
#             return ds.map(
#                 map_func=lambda signal: get_eeg_spectrogram_no_batch(signal))
#         elif signal_type == 'emg':
#             return ds.map(
#                 map_func=lambda signal: get_emg_spectrogram_no_batch(signal))
#
#     input_dataset_eeg1 = make_spec_ds(input_dataset_eeg1, 'eeg')
#     input_dataset_eeg2 = make_spec_ds(input_dataset_eeg2, 'eeg')
#     input_dataset_emg = make_spec_ds(input_dataset_emg, 'emg')
#
#     d = tf.data.Dataset.zip((input_dataset_eeg1, input_dataset_eeg2, input_dataset_emg))
#
#     d = d.map(lambda x, y, z: tf.stack([x, y, z], axis=-1))
#
#     return d
#
#
# input_dataset = load_and_preprocess_data_2(file_path)
#
# for a in input_dataset:
#     b=0


# --------------------------------------------------------------------------------------------------------------------------------
labels_dataset = tf.data.Dataset.from_tensor_slices(y_1[2:-2])
# labels_dataset = labels_dataset.batch(32)

dataset = tf.data.Dataset.zip((input_dataset, labels_dataset), name=None)
# dataset = dataset.shuffle(1000).batch(32)
dataset = dataset.batch(32)


def visualize_dataset(dataset,  same_scale=True):
    stage_dict = {'3': 'NREM', '4': 'REM', '5': 'WAKE'}

    for i in range(3): # for the 3 signals (EEG_1, EEG_2, EMG)
        # dataset = dataset.shuffle(22000)
        # i=0

        vmin_eeg = -6
        vmax_eeg = 2.5
        vmin_emg = -5
        vmax_emg = 2

        fig, ax = plt.subplots(3, 6, figsize=(18, 9))

        # for j in range(3,6): # for the three non-artifact categories
        for j in range(3): # for the three non-artifact categories

        # j = 3


            counter = 0

            for b, c in dataset:
                if counter < 6:
                    category_indexes = tf.where(c == 1)[:, 1]

                    idx = tf.where(category_indexes == j)

                    if len(idx) != 0:# and len(idx)>20:
                        # idx = idx[random.randint(0,len(idx)-1)]
                        idx = idx[0]
                        b = tf.concat(
                            [b[:, 0, :, :, :], b[:, 1, :, :, :], b[:, 2, :, :, :], b[:, 3, :, :, :], b[:, 4, :, :, :]],
                            axis=2)
                        spect = b[int(idx), :, :, i] #,:,:,:]

                        # ax[j-3, counter].imshow(tf.reverse(tf.transpose(spect), axis=[0]), cmap='jet')
                        # ax[j-3, counter].set_title(stage_dict[str(j)])
                        if same_scale:
                            if i == 2:
                                img = ax[j, counter].imshow(spect, cmap='jet', vmin=vmin_emg, vmax=vmax_emg)
                            else:
                                img = ax[j, counter].imshow(spect, cmap='jet', vmin=vmin_eeg, vmax=vmax_eeg)
                        else:
                            img = ax[j, counter].imshow(spect, cmap='jet')

                        ax[j, counter].set_title(stage_dict[str(j+3)])
                        # plt.colorbar(ax=ax[j-3,counter])
                        ax[j, counter].invert_yaxis()
                        ax[j, counter].set_yticks([ax[j, counter].get_ylim()[1], abs(ax[j, counter].get_ylim()[1] - ax[j, counter].get_ylim()[0])/2, ax[j, counter].get_ylim()[0]])
                        ax[j, counter].set_yticklabels(['24', '12', '0,5'])
                        ax[j, counter].set_xticks([ax[j, counter].get_xlim()[1],
                                                  ax[j, counter].get_xlim()[0]])
                        ax[j, counter].set_xticklabels([str(ax[j, counter].get_xlim()[1] + 0.5),
                                                        str(ax[j, counter].get_xlim()[0] + 0.5)])
                        # ax[j, counter].vlines(x=np.linspace(spect.shape[1]/5, spect.shape[1]/5*4, 4), ymin=ax[j, counter].get_ylim()[0], ymax=ax[j, counter].get_ylim()[1], color='k')
                        fig.colorbar(img, ax=ax[j, counter])

                        counter = counter + 1
                else:
                    break

        if i == 0:
            suptitle = 'EEG_1'
        elif i == 1:
            suptitle = 'EEG_2'
        else:
            suptitle = 'EMG'

        fig.suptitle(suptitle)
        # plt.show()
        fig.savefig('v2_' + suptitle + '_preprocessing')


visualize_dataset(dataset, False)

#-------------------------------------------------------------------------------------------------------------------------

input_dataset_2 = load_and_preprocess_data_2(r'C:\Users\javig\Desktop\SPINDLE dataset\SPINDLE dataset\data\CohortA\recordings\A2.edf')
y_1, y_2 = load_labels('C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data/CohortA/scorings/A2.csv')

labels_dataset_2 = tf.data.Dataset.from_tensor_slices(y_1[2:-2])
# labels_dataset_2 = labels_dataset_2.batch(32)

dataset_2 = tf.data.Dataset.zip((input_dataset_2, labels_dataset_2), name=None)

dataset = dataset.concatenate(dataset_2)
dataset = dataset.shuffle(43200).batch(32)

del input_dataset_2
del dataset_2

spindle_model = tf.keras.Sequential([
    Input((145, 48, 3)),
    MaxPool2D(pool_size=(2, 3), strides=(2, 3)),
    Conv2D(filters=50, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
    MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    Dense(units=1000, activation='relu'),
    Dense(units=1000, activation='relu'),
    Dense(units=3, activation='softmax')
])


spindle_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5*1e-5),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.CategoricalAccuracy()]
)

spindle_model.fit(dataset, epochs=4, verbose=1)


input_dataset_3 = load_and_preprocess_data_2(r'C:\Users\javig\Desktop\SPINDLE dataset\SPINDLE dataset\data\CohortA\recordings\A3.edf')
y_1, y_2 = load_labels('C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data/CohortA/scorings/A3.csv')

labels_dataset_3 = tf.data.Dataset.from_tensor_slices(y_1[2:-2])
labels_dataset_3 = labels_dataset_3.batch(32)

evaluation_outputs = spindle_model.evaluate(input_dataset_3, verbose=1)


# HIDDEN MARKOV MODEL --------------------------------------------------------------------

# TENSORFLOW
initial_distribution = tfd.Categorical(probs=[1/3, 1/3, 1/3])
transition_distribution = tfd.Categorical(probs=[[0.5, 0, 0.5],
                                                 [1/3, 1/3, 1/3],
                                                 [0, 0.5, 0.5]])

observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])

hmm_model = tfp.distributions.HiddenMarkovModel(initial_distribution,
                                                transition_distribution,
                                                observation_distribution,
                                                num_steps,
    validate_args=False,
    allow_nan_stats=True,
    time_varying_transition_distribution=False,
    time_varying_observation_distribution=False,
    mask=None,
    name='HiddenMarkovModel'
)

posterior_mode(
    observations, mask=None, name='posterior_mode'
)


# STACKOVERFLOW (https://stackoverflow.com/questions/9729968/python-implementation-of-viterbi-algorithm)

def viterbi(y, A, B, Pi=None):
    """
    Return the MAP estimate of state trajectory of Hidden Markov Model.

    Parameters
    ----------
    y : array (T,)
        Observation state sequence. int dtype.
    A : array (K, K)
        State transition matrix. See HiddenMarkovModel.state_transition  for
        details.
    B : array (K, M)
        Emission matrix. See HiddenMarkovModel.emission for details.
    Pi: optional, (K,)
        Initial state probabilities: Pi[i] is the probability x[0] == i. If
        None, uniform initial distribution is assumed (Pi[:] == 1/K).

    Returns
    -------
    x : array (T,)
        Maximum a posteriori probability estimate of hidden state trajectory,
        conditioned on observation sequence y under the model parameters A, B,
        Pi.
    T1: array (K, T)
        the probability of the most likely path so far
    T2: array (K, T)
        the x_j-1 of the most likely path so far
    """
    # Cardinality of the state space
    K = A.shape[0]
    # Initialize the priors with default (uniform dist) if not given by caller
    Pi = Pi if Pi is not None else np.full(K, 1 / K)
    T = len(y)
    T1 = np.empty((K, T), 'd')
    T2 = np.empty((K, T), 'B')

    # Initilaize the tracking tables from first observation
    T1[:, 0] = Pi * B[:, y[0]]
    T2[:, 0] = 0

    # Iterate throught the observations updating the tracking tables
    for i in range(1, T):
        T1[:, i] = np.max(T1[:, i - 1] * A.T * B[np.newaxis, :, y[i]].T, 1)
        T2[:, i] = np.argmax(T1[:, i - 1] * A.T, 1)

    # Build the output, optimal model trajectory
    x = np.empty(T, 'B')
    x[-1] = np.argmax(T1[:, T - 1])
    for i in reversed(range(1, T)):
        x[i - 1] = T2[x[i], i]

    return x, T1, T2





