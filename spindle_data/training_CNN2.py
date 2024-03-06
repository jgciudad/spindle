from tensorflow.keras.layers import Input, MaxPool2D, Conv2D, Dense, Flatten, Dropout
from spindle_data.spindle_data_loading import SequenceDataset
from metrics import *
from tools import *

plt.ion()

save_path = r'/results/3 - new round of results after meeting'
model_name = 'B_1'


# signal_path1 = "C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data/CohortA/recordings/A1.edf"
# labels_path1 = 'C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data/CohortA/scorings/A1.csv'
# signal_path2 = "C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data/CohortA/recordings/A2.edf"
# labels_path2 = 'C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data/CohortA/scorings/A2.csv'
#
# train_dataset_1, val_dataset_1 = load_recording_to_dataset(signal_path=signal_path1,
#                                                            labels_path=labels_path1,
#                                                            validation_split=0.1,
#                                                            scorer=1,
#                                                            just_artifact_labels=True,
#                                                            artifact_to_stages=False,
#                                                            balance_artifacts=False)
# train_dataset_2, val_dataset_2 = load_recording_to_dataset(signal_path=signal_path2,
#                                                            labels_path=labels_path2,
#                                                            validation_split=0.1,
#                                                            scorer=1,
#                                                            just_artifact_labels=True,
#                                                            artifact_to_stages=False,
#                                                            balance_artifacts=False)
# train_dataset = train_dataset_1.concatenate(train_dataset_2)
# val_dataset = val_dataset_1.concatenate(val_dataset_2)
#
# del train_dataset_1
# del val_dataset_1
# del train_dataset_2
# del val_dataset_2

# # signal_paths = [r"C:\Users\javig\Desktop\SPINDLE dataset\SPINDLE dataset\data (original)\CohortA\recordings\A1.edf"]
# signal_paths = [r"C:\Users\javig\Desktop\SPINDLE dataset\SPINDLE dataset\data (original)\CohortA\recordings\A2.edf"]
# # labels_paths = [r"C:\Users\javig\Desktop\SPINDLE dataset\SPINDLE dataset\data (original)\CohortA\scorings\A1.csv"]
# labels_paths = [r"C:\Users\javig\Desktop\SPINDLE dataset\SPINDLE dataset\data (original)\CohortA\scorings\A2.csv"]
#
# train_dataset, val_dataset = load_to_dataset(signal_paths=signal_paths,
#                                              labels_paths=labels_paths,
#                                              scorer=4,
#                                              just_artifact_labels=True,
#                                              artifact_to_stages=False,
#                                              balance_artifacts=False,
#                                              validation_split=0.1)
#
# batch_size = 100
# train_dataset = train_dataset.batch(batch_size)
# val_dataset = val_dataset.batch(batch_size)

# -------------------------------------------------------------------------------------------------------------------------


train_sequence = SequenceDataset(r'C:\Users\javig\Desktop\SPINDLE dataset\SPINDLE dataset\preprocessed\d5\training',
                                 100,
                                 binary=True)
val_sequence = SequenceDataset(r'C:\Users\javig\Desktop\SPINDLE dataset\SPINDLE dataset\preprocessed\d5\validation',
                               100,
                               binary=True)


# -------------------------------------------------------------------------------------------------------------------------

spindle_model2 = tf.keras.Sequential([
    Input((160, 48, 3)),
    MaxPool2D(pool_size=(2, 3), strides=(2, 3)),
    Conv2D(filters=50, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
    MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    Dense(units=1000, activation='relu', kernel_initializer='glorot_uniform'),
    Dropout(0.5),
    Dense(units=1000, activation='relu', kernel_initializer='glorot_uniform'),
    Dropout(0.5),
    Dense(units=1, activation='sigmoid', kernel_initializer='glorot_uniform')
])
# tf.config.run_functions_eagerly(True)
spindle_model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5 * 1e-5,
                                                          beta_1=0.9,
                                                          beta_2=0.999),
                       loss=BinaryWeightedCrossEntropy(),  # BinaryWeightedCrossEntropy # tf.keras.losses.BinaryCrossentropy()
                       metrics=[tf.keras.metrics.BinaryAccuracy(),
                                BinaryBalancedAccuracy(),
                                BinaryF1Score()])
                       # run_eagerly=True)

history1 = spindle_model2.fit(x=train_sequence,
                              validation_data=val_sequence,
                              epochs=5,
                              verbose=1)

plot_history_cnn2(history1.history, model_name, save_path, epochs=5)

model_path = os.path.join(save_path, model_name, model_name + "_5epochs" + ".h5")
if not os.path.exists(os.path.dirname(model_path)):
    os.makedirs(os.path.dirname(model_path))
spindle_model2.save_weights(model_path)

# FIND OPTIMAL THRESHOLD



history2 = spindle_model2.fit(x=train_sequence,
                              validation_data=val_sequence,
                              epochs=5,
                              verbose=1)

plot_history_cnn2(concatenate_histories(history1.history, history2.history), model_name, save_path, epochs=10)
spindle_model2.save_weights(os.path.join(save_path, model_name, model_name + "_10epochs" + ".h5"))

history3 = spindle_model2.fit(x=train_dataset,
                              validation_data=val_dataset,
                              epochs=5,
                              verbose=1)

plot_history_cnn2(concatenate_histories(concatenate_histories(history1.history, history2.history), history3.history), model_name, save_path, epochs=15)
spindle_model2.save_weights(os.path.join(save_path, model_name, model_name + "_15epochs" + ".h5"))

history4 = spindle_model2.fit(x=train_dataset,
                              validation_data=val_dataset,
                              epochs=5,
                              verbose=1)

plot_history(concatenate_histories(concatenate_histories(concatenate_histories(history1.history, history2.history), history3.history), history4.history), model_name, save_path, epochs=20)
spindle_model2.save_weights(os.path.join(save_path, model_name, model_name + "_20epochs" + ".h5"))


























############################################ REMOVED FUNCTIONS ####################################################


# def load_labels(labels_path, artifact_to_stages=False, just_artifact_labels=False):
#     df = pd.read_csv(labels_path, header=None)
#
#     labels_1 = pd.get_dummies(df[1])
#     labels_2 = pd.get_dummies(df[2])
#
#     # column names: {1, 2, 3, n, r, w}
#     # 1=wake artifact, 2=NREM artifact, 3=REM artifact
#
#     if artifact_to_stages:
#         labels_1.loc[labels_1["1"] == 1, 'w'] = 1
#         labels_2.loc[labels_2["1"] == 1, 'w'] = 1
#         labels_1.loc[labels_1["2"] == 1, 'n'] = 1
#         labels_2.loc[labels_2["2"] == 1, 'n'] = 1
#         labels_1.loc[labels_1["3"] == 1, 'r'] = 1
#         labels_2.loc[labels_2["3"] == 1, 'r'] = 1
#
#         labels_1 = labels_1.iloc[:, -3:]
#         labels_2 = labels_2.iloc[:, -3:]
#     elif just_artifact_labels:
#         labels_1.loc[(labels_1['1'] == 1) | (labels_1['2'] == 1) | (labels_1['3'] == 1), 'art'] = 1
#         labels_1.loc[(labels_1['1'] == 0) & (labels_1['2'] == 0) & (labels_1['3'] == 0), 'art'] = 0
#
#         labels_2.loc[(labels_2['1'] == 1) | (labels_2['2'] == 1) | (labels_2['3'] == 1), 'art'] = 1
#         labels_2.loc[(labels_2['1'] == 0) & (labels_2['2'] == 0) & (labels_2['3'] == 0), 'art'] = 0
#
#         labels_1 = labels_1['art']
#         labels_2 = labels_2['art']
#
#     return [labels_1, labels_2]


# def get_artifact_epochs(labels):
#     labels = labels_1
#     artifact_epochs = labels.index[(labels['1'] == 1) | (labels['2'] == 1) | (labels['3'] == 1)].tolist()
#
#     return artifact_epochs


# def epochs_analysis(labels_1, labels_2):
#     # Select only epochs with agreement between scorers and optionally without artifacts
#     # Label arrays should contain all the six labels
#
#     labels_1 = labels_1.to_numpy()
#     labels_2 = labels_2.to_numpy()
#
#     # Number of epochs where experts disagree
#     disagreement_epochs = labels_1 == labels_2
#     disagreement_epochs = np.sum(labels_1 == labels_2, axis=1)
#     disagreement_epochs = np.where(disagreement_epochs != 3)[0]
#     n_disagreement_epochs = len(disagreement_epochs)
#
#     # Number of epochs that are labeled as artifacts for every scorer
#     n_artifacts_1 = len(np.where(np.sum(labels_1, axis=1) != 1)[0])
#     n_artifacts_2 = len(np.where(np.sum(labels_2, axis=1) != 1)[0])
#
#     # Number of epochs that are labeled as artifact by one expert but not by the other
#
#
#     a = pd.merge(labels_1, labels_2, how='inner')


# def preprocess_EEG(signal,
#                    fs=128,
#                    stft_size=256,
#                    stft_stride=16,
#                    lowcut=0.5,
#                    highcut=24,
#                    visualize=False,
#                    labels=None,
#                    plot_artifacts=False):
#     if visualize:
#         # Select random epoch
#         rdm_epoch_idx = np.random.randint(2, len(signal) / 4 / fs - 2)
#         rdm_epoch_labels = labels.to_numpy()[rdm_epoch_idx - 2:rdm_epoch_idx + 3, :]
#
#         if plot_artifacts:
#             labels_dict = {0: 'W_art', 1: 'N_art', 2: 'R_art', 3: 'NREM', 4: 'REM', 5: 'WAKE'}
#         else:
#             labels_dict = {0: 'NREM', 1: 'REM', 2: 'WAKE'}
#         rdm_epoch_labels = np.where(rdm_epoch_labels == 1)[1]
#         rdm_epoch_labels = [labels_dict[i] for i in rdm_epoch_labels]
#
#         rdm_epoch_signal = signal[(rdm_epoch_idx - 2) * fs * 4: (rdm_epoch_idx + 3) * fs * 4]
#         time_axis = np.linspace((rdm_epoch_idx - 2) * fs * 4, (rdm_epoch_idx + 3) * fs * 4, fs * 4 * 5) / fs
#
#         fig, ax = plt.subplots(6, 2, figsize=(15, 10), gridspec_kw={'width_ratios': [60, 1]})
#         fig.subplots_adjust(hspace=0.8)
#         cax = ax[0, 0]
#         cax.plot(time_axis, rdm_epoch_signal)
#         cax.vlines(x=np.linspace(time_axis[0] + 4, time_axis[-1] - 4, 4),
#                    ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
#         cax.set_title('Raw 5 epochs window')
#         # cax.set_xlabel('Time (s)')
#         cax.set_xticks(np.linspace(time_axis[0], time_axis[-1], 6))
#         cax.set_xlim((time_axis[0], time_axis[-1]))
#         epoch_labels_ax = cax.twiny()
#         epoch_labels_ax.set_xlim(cax.get_xlim())
#         epoch_labels_ax.set_xticks(np.linspace(time_axis[0]+ 2, time_axis[-1]-2, 5))
#         epoch_labels_ax.set_xticklabels(rdm_epoch_labels)
#         epoch_labels_ax.tick_params(length=0)
#         ax[0, 1].axis('off')
#
#     # STFT
#     f, t, Z = scipy.signal.stft(signal,
#                                 fs=128,
#                                 window='hamming',
#                                 nperseg=stft_size,
#                                 noverlap=stft_size - stft_stride
#                                 )
#
#     if visualize:
#         cax = ax[1, 0]
#
#         rdm_epoch_spect = Z[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]
#         # time_axis = np.linspace((rdm_epoch_idx-2)*32, (rdm_epoch_idx+3)*32, 32*5)
#         time_axis = np.linspace((rdm_epoch_idx - 2) * fs * 4, (rdm_epoch_idx + 3) * fs * 4, 18) / fs
#
#         img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
#         cax.set_title('Spectrogram')
#         cax.invert_yaxis()
#         # cax.set_xlabel('Time (s)')
#         cax.set_ylabel('Frequency (Hz.)')
#         cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
#         cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
#         cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
#         cax.set_yticklabels([str(f[-1]), str(f[-1] / 2), str(f[0])])
#         cax.vlines(x=cax.get_xticks()[1:-1],
#                    ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
#         fig.colorbar(img, cax=ax[1, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])
#
#     # Bandpass (crop)
#     Z = Z[np.where(f == lowcut)[0][0]: np.where(f == highcut)[0][0] + 1, :]
#
#     if visualize:
#         cax = ax[2, 0]
#
#         rdm_epoch_spect = Z[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]
#
#         img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
#         cax.set_title('Bandpass')
#         cax.invert_yaxis()
#         # cax.set_xlabel('Time (s)')
#         cax.set_ylabel('Frequency (Hz.)')
#         cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
#         cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
#         cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
#         cax.set_yticklabels(['24', '12', '0'])
#         cax.vlines(x=cax.get_xticks()[1:-1],
#                    ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
#         fig.colorbar(img, cax=ax[2, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])
#
#     # PSD
#     y = np.abs(Z) ** 2
#
#     if visualize:
#         cax = ax[3, 0]
#
#         rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]
#
#         img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
#         cax.set_title('PSD')
#         cax.invert_yaxis()
#         # cax.set_xlabel('Time (s)')
#         cax.set_ylabel('Frequency (Hz.)')
#         cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
#         cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
#         cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
#         cax.set_yticklabels(['24', '12', '0'])
#         cax.vlines(x=cax.get_xticks()[1:-1],
#                    ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
#         fig.colorbar(img, cax=ax[3, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])
#
#     # Log-scale
#     y = 10 * np.log10(y)
#
#     if visualize:
#         cax = ax[4, 0]
#
#         rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]
#
#         img = cax.imshow(rdm_epoch_spect, cmap='jet', aspect='auto')
#         cax.set_title('Log transformation')
#         cax.invert_yaxis()
#         # cax.set_xlabel('Time (s)')
#         cax.set_ylabel('Frequency (Hz.)')
#         cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
#         cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
#         cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
#         cax.set_yticklabels(['24', '12', '0'])
#         cax.vlines(x=cax.get_xticks()[1:-1],
#                    ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
#         fig.colorbar(img, cax=ax[4, 1], ticks=[np.min(rdm_epoch_spect), np.max(rdm_epoch_spect)])
#
#     # Standardize
#     y_mean = np.mean(y, axis=1, keepdims=True)
#     y_std = np.std(y, axis=1, keepdims=True)
#
#     y = (y - y_mean) / y_std
#
#     if visualize:
#         cax = ax[5, 0]
#
#         rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]
#
#         img = cax.imshow(rdm_epoch_spect, cmap='jet', aspect='auto')
#         cax.set_title('Standardization')
#         cax.invert_yaxis()
#         cax.set_xlabel('Time (s)')
#         cax.set_ylabel('Frequency (Hz.)')
#         cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
#         cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
#         cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
#         cax.set_yticklabels(['24', '12', '0'])
#         cax.vlines(x=cax.get_xticks()[1:-1],
#                    ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
#         fig.colorbar(img, cax=ax[5, 1], ticks=[np.min(rdm_epoch_spect), np.max(rdm_epoch_spect)])
#         plt.show()
#
#     return y

# def preprocess_EMG(signal,
#                    fs=128,
#                    stft_size=256,
#                    stft_stride=16,
#                    lowcut=0.5,
#                    highcut=30,
#                    visualize=False,
#                    labels=None,
#                    plot_artifacts=False):
#
#     if visualize:
#         # Select random epoch
#         rdm_epoch_idx = np.random.randint(2, len(signal) / 4 / fs - 2)
#         rdm_epoch_labels = labels.to_numpy()[rdm_epoch_idx - 2:rdm_epoch_idx + 3, :]
#
#         if plot_artifacts:
#             labels_dict = {0: 'W_art', 1: 'N_art', 2: 'R_art', 3: 'NREM', 4: 'REM', 5: 'WAKE'}
#         else:
#             labels_dict = {0: 'NREM', 1: 'REM', 2: 'WAKE'}
#         rdm_epoch_labels = np.where(rdm_epoch_labels == 1)[1]
#         rdm_epoch_labels = [labels_dict[i] for i in rdm_epoch_labels]
#
#         rdm_epoch_signal = signal[(rdm_epoch_idx - 2) * fs * 4: (rdm_epoch_idx + 3) * fs * 4]
#         time_axis = np.linspace((rdm_epoch_idx - 2) * fs * 4, (rdm_epoch_idx + 3) * fs * 4, fs * 4 * 5) / fs
#
#         fig, ax = plt.subplots(7, 2, figsize=(15, 10), gridspec_kw={'width_ratios': [60, 1]})
#         fig.subplots_adjust(hspace=0.8)
#         cax = ax[0, 0]
#         cax.plot(time_axis, rdm_epoch_signal)
#         cax.vlines(x=np.linspace(time_axis[0] + 4, time_axis[-1] - 4, 4),
#                    ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
#         cax.set_title('Raw 5 epochs window')
#         # cax.set_xlabel('Time (s)')
#         cax.set_xticks(np.linspace(time_axis[0], time_axis[-1], 6))
#         cax.set_xlim((time_axis[0], time_axis[-1]))
#         epoch_labels_ax = cax.twiny()
#         epoch_labels_ax.set_xlim(cax.get_xlim())
#         epoch_labels_ax.set_xticks(np.linspace(time_axis[0]+ 2, time_axis[-1]-2, 5))
#         epoch_labels_ax.set_xticklabels(rdm_epoch_labels)
#         epoch_labels_ax.tick_params(length=0)
#         ax[0, 1].axis('off')
#
#     # STFT
#     f, t, Z = scipy.signal.stft(signal,
#                                 fs=128,
#                                 window='hamming',
#                                 nperseg=stft_size,
#                                 noverlap=stft_size - stft_stride
#                                 )
#
#     if visualize:
#         cax = ax[1, 0]
#
#         rdm_epoch_spect = Z[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]
#         # time_axis = np.linspace((rdm_epoch_idx-2)*32, (rdm_epoch_idx+3)*32, 32*5)
#         time_axis = np.linspace((rdm_epoch_idx - 2) * fs * 4, (rdm_epoch_idx + 3) * fs * 4, 18) / fs
#
#         img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
#         cax.set_title('Spectrogram')
#         cax.invert_yaxis()
#         # cax.set_xlabel('Time (s)')
#         cax.set_ylabel('Frequency (Hz.)')
#         cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
#         cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
#         cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
#         cax.set_yticklabels([str(f[-1]), str(f[-1] / 2), str(f[0])])
#         cax.vlines(x=cax.get_xticks()[1:-1],
#                    ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
#         fig.colorbar(img, cax=ax[1, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])
#
#     # Bandpass (crop)
#     Z = Z[np.where(f == lowcut)[0][0]: np.where(f == highcut)[0][0] + 1, :]
#
#     if visualize:
#         cax = ax[2, 0]
#
#         rdm_epoch_spect = Z[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]
#
#         img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
#         cax.set_title('Bandpass')
#         cax.invert_yaxis()
#         # cax.set_xlabel('Time (s)')
#         cax.set_ylabel('Frequency (Hz.)')
#         cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
#         cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
#         cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
#         cax.set_yticklabels(['24', '12', '0'])
#         cax.vlines(x=cax.get_xticks()[1:-1],
#                    ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
#         fig.colorbar(img, cax=ax[2, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])
#
#     # PSD
#     y = np.abs(Z) ** 2
#
#     if visualize:
#         cax = ax[3, 0]
#
#         rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]
#
#         img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
#         cax.set_title('PSD')
#         cax.invert_yaxis()
#         # cax.set_xlabel('Time (s)')
#         cax.set_ylabel('Frequency (Hz.)')
#         cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
#         cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
#         cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
#         cax.set_yticklabels(['24', '12', '0'])
#         cax.vlines(x=cax.get_xticks()[1:-1],
#                    ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
#         fig.colorbar(img, cax=ax[3, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])
#
#     # Integration
#     y = np.sum(y, axis=0)
#
#     # Stack rows to have 2 dimensions
#     y = np.expand_dims(y, axis=0)
#     # y = np.repeat(y, eeg_dimensions[0], axis=0)
#     y = np.repeat(y, 48, axis=0)
#
#     if visualize:
#         cax = ax[4, 0]
#
#         rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]
#
#         img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
#         cax.set_title('Integration')
#         cax.invert_yaxis()
#         # cax.set_xlabel('Time (s)')
#         cax.set_ylabel('Frequency (Hz.)')
#         cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
#         cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
#         cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
#         cax.set_yticklabels(['24', '12', '0'])
#         cax.vlines(x=cax.get_xticks()[1:-1],
#                    ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
#         fig.colorbar(img, cax=ax[4, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])
#
#     # Log-scale
#     y = 10*np.log10(y)
#
#     if visualize:
#         cax = ax[5, 0]
#
#         rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]
#
#         img = cax.imshow(rdm_epoch_spect, cmap='jet', aspect='auto')
#         cax.set_title('Log transformation')
#         cax.invert_yaxis()
#         # cax.set_xlabel('Time (s)')
#         cax.set_ylabel('Frequency (Hz.)')
#         cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
#         cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
#         cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
#         cax.set_yticklabels(['24', '12', '0'])
#         cax.vlines(x=cax.get_xticks()[1:-1],
#                    ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
#         fig.colorbar(img, cax=ax[5, 1], ticks=[np.min(rdm_epoch_spect), np.max(rdm_epoch_spect)])
#
#     # Standardize
#     y_mean = np.mean(y, axis=1, keepdims=True)
#     y_std = np.std(y, axis=1, keepdims=True)
#
#     y = (y - y_mean) / y_std
#
#     if visualize:
#         cax = ax[6, 0]
#
#         rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]
#
#         img = cax.imshow(rdm_epoch_spect, cmap='jet', aspect='auto')
#         cax.set_title('Standardization')
#         cax.invert_yaxis()
#         cax.set_xlabel('Time (s)')
#         cax.set_ylabel('Frequency (Hz.)')
#         cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
#         cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
#         cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
#         cax.set_yticklabels(['24', '12', '0'])
#         cax.vlines(x=cax.get_xticks()[1:-1],
#                    ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
#         fig.colorbar(img, cax=ax[6, 1], ticks=[np.min(rdm_epoch_spect), np.max(rdm_epoch_spect)])
#         plt.show()
#
#     return y


# def windowing(signal, window_size=32*5, window_stride=32, fs=128):
#     n_windows = 21600 - 4
#
#     windowed_signal = np.zeros((n_windows, 3, 48, window_size))
#
#     # signal = signal[window_size//2 : -window_size//2],
#
#     for i in range(n_windows):
#         windowed_signal[i, :, :, :] = signal[:, :, (i*window_stride) : (i*window_stride) + window_size]
#
#     return windowed_signal


# def load_recording_to_dataset_1(signal_path, labels_path, stft_size, stft_stride, fs, epoch_length):
#
#     y_1, y_2 = load_labels(labels_path)
#     # filter_epochs(y_1, y_2)
#     raw_data = load_raw_recording(signal_path)
#     eeg_1 = preprocess_EEG(raw_data[0, :], labels=y_1)#, visualize=True)
#     eeg_2 = preprocess_EEG(raw_data[1, :], labels=y_1)#, visualize=True)
#     emg = preprocess_EMG(raw_data[2, :], labels=y_1)#, visualize=True)
#     x = np.stack((eeg_1, eeg_2, emg))
#
#     # number of stft windows per epoch
#     wp = ((fs * EPOCH_LENGTH) - stft_size) / stft_stride + 2 * (stft_size / 2 / stft_stride)
#
#     dataset = tf.keras.utils.timeseries_dataset_from_array(
#         data=x.T,
#         # targets=None,
#         sequence_length=5 * wp,
#         sequence_stride=wp,
#         batch_size=None,
#         targets=y_1[2:-2],
#         shuffle=True
#     )
#
#     # labels_dataset = tf.data.Dataset.from_tensor_slices(y_1[2:-2])
#     # labels_dataset = labels_dataset.batch(32)
#     #
#     # dataset = tf.data.Dataset.zip((input_dataset, labels_dataset), name=None)
#
#     return dataset


# def load_recording_to_dataset_2(signal_path, labels_path, validation_split=None, to_numpy=False, just_artifact_labels=False): # stft_size, stft_stride, fs, epoch_length,
#     y_1, y_2 = load_labels(labels_path, just_artifact_labels=just_artifact_labels)
#     # filter_epochs(y_1, y_2)
#     raw_data = load_raw_recording(signal_path)
#     eeg_1 = preprocess_EEG(raw_data[0, :], labels=y_1)  # , visualize=True)
#     eeg_2 = preprocess_EEG(raw_data[1, :], labels=y_1)  # , visualize=True)
#     emg = preprocess_EMG(raw_data[2, :], labels=y_1)  # , visualize=True)
#     x = np.stack((eeg_1, eeg_2, emg))
#
#     x = windowing(x, window_size=32 * 5, window_stride=32)
#     x = np.transpose(x, (0, 3, 2, 1))
#
#     if validation_split is not None:
#         rdm_indexes = np.arange(x.shape[0])
#         np.random.shuffle(rdm_indexes)
#         train_indexes = rdm_indexes[:int(len(rdm_indexes)*(1-validation_split))]
#         val_indexes = rdm_indexes[int(len(rdm_indexes)*(1-validation_split)):]
#
#         x_train = x[train_indexes]
#         labels_train = y_1.to_numpy()[2:-2][train_indexes]
#         x_val = x[val_indexes]
#         labels_val = y_1.to_numpy()[2:-2][val_indexes]
#
#         input_dataset_train = tf.data.Dataset.from_tensor_slices(x_train)
#         labels_dataset_train = tf.data.Dataset.from_tensor_slices(labels_train)
#         input_dataset_val = tf.data.Dataset.from_tensor_slices(x_val)
#         labels_dataset_val = tf.data.Dataset.from_tensor_slices(labels_val)
#         # labels_dataset = labels_dataset.batch(32)
#
#         train_dataset = tf.data.Dataset.zip((input_dataset_train, labels_dataset_train))
#         val_dataset = tf.data.Dataset.zip((input_dataset_val, labels_dataset_val))
#
#         return train_dataset, val_dataset
#     else:
#         input_dataset = tf.data.Dataset.from_tensor_slices(x)
#         labels_dataset = tf.data.Dataset.from_tensor_slices(y_1[2:-2])
#         dataset = tf.data.Dataset.zip((input_dataset, labels_dataset))
#
#         return dataset

# def load_labels_2(labels_path,
#                   scorer,
#                   just_artifact_labels,
#                   artifact_to_stages):
#
#     df = pd.read_csv(labels_path, header=None)
#
#     df = df.iloc[2:-2].copy() # Drop 2 first and 2 last epochs
#
#     labels_1 = pd.get_dummies(df[1])
#     labels_2 = pd.get_dummies(df[2])
#
#     if scorer==0:
#         labels = labels_1[(labels_1 == labels_2).all(axis=1) == True].copy()
#     elif scorer==1:
#         labels = labels_1
#     elif scorer==2:
#         labels = labels_2
#
#     if just_artifact_labels==True:
#         labels.loc[(labels['1'] == 1) | (labels['2'] == 1) | (labels['3'] == 1), 'art'] = 1
#         labels.loc[(labels['1'] == 0) & (labels['2'] == 0) & (labels['3'] == 0), 'art'] = 0
#
#         labels = labels['art']
#     elif just_artifact_labels==False:
#         if artifact_to_stages==True:
#             labels.loc[labels["1"] == 1, 'w'] = 1
#             labels.loc[labels["2"] == 1, 'n'] = 1
#             labels.loc[labels["3"] == 1, 'r'] = 1
#
#             labels = labels.iloc[:, -3:]
#         elif artifact_to_stages==False:
#             labels.drop(labels.loc[(labels['1'] == 1) | (labels['2'] == 1) | (labels['3'] == 1)].index, inplace=True)
#             labels = labels.iloc[:, -3:]
#
#     return labels


# def load_recording_to_dataset_4(signal_path,
#                                 labels_path,
#                                 scorer,
#                                 just_artifact_labels,
#                                 artifact_to_stages,
#                                 balance_artifacts,
#                                 validation_split): # stft_size, stft_stride, fs, epoch_length,
#
#     # Is load_recording_to_dataset_4
#
#     raw_data = load_raw_recording(signal_path)
#     eeg_1 = preprocess_EEG(raw_data[0, :])
#     eeg_2 = preprocess_EEG(raw_data[1, :])
#     emg = preprocess_EMG(raw_data[2, :])
#     x = np.stack((eeg_1, eeg_2, emg))
#     x = windowing(x, window_size=32 * 5, window_stride=32)
#     x = np.transpose(x, (0, 3, 2, 1))
#
#     y = load_labels_2(labels_path,
#                       scorer=scorer,
#                       just_artifact_labels=just_artifact_labels,
#                       artifact_to_stages=artifact_to_stages)
#     x = x[y.index.to_numpy() - 2] # Select just the epochs in y
#     y = y.to_numpy()
#
#     if just_artifact_labels and balance_artifacts:
#         balanced_subset_idx = balanced_artifacts(y)
#         y = y[balanced_subset_idx]
#         x = x[balanced_subset_idx]
#
#     if validation_split is not None:
#         rdm_indexes = np.arange(x.shape[0])
#         np.random.shuffle(rdm_indexes)
#         train_indexes = rdm_indexes[:int(len(rdm_indexes)*(1-validation_split))]
#         val_indexes = rdm_indexes[int(len(rdm_indexes)*(1-validation_split)):]
#
#         x_train = x[train_indexes]
#         x_val = x[val_indexes]
#         labels_train = y[train_indexes]
#         labels_val = y[val_indexes]
#
#         input_dataset_train = tf.data.Dataset.from_tensor_slices(x_train)
#         labels_dataset_train = tf.data.Dataset.from_tensor_slices(labels_train)
#         input_dataset_val = tf.data.Dataset.from_tensor_slices(x_val)
#         labels_dataset_val = tf.data.Dataset.from_tensor_slices(labels_val)
#
#         train_dataset = tf.data.Dataset.zip((input_dataset_train, labels_dataset_train))
#         val_dataset = tf.data.Dataset.zip((input_dataset_val, labels_dataset_val))
#
#         return train_dataset, val_dataset
#     else:
#         input_dataset = tf.data.Dataset.from_tensor_slices(x)
#         if balance_artifacts == False:
#             labels_dataset = tf.data.Dataset.from_tensor_slices(y_1[2:-2])
#         else:
#             labels_dataset = tf.data.Dataset.from_tensor_slices(y_1)
#         dataset = tf.data.Dataset.zip((input_dataset, labels_dataset))
#
#         return dataset


# def load_recording_to_dataset_3(signal_path,
#                                 labels_path,
#                                 validation_split=None,
#                                 just_artifact_labels=False,
#                                 balance_artifacts=False):  # stft_size, stft_stride, fs, epoch_length,
#
#
#     # Like load_recording_to_dataset_2 but balancing the artifacts
#
#     raw_data = load_raw_recording(signal_path)
#     eeg_1 = preprocess_EEG(raw_data[0, :])
#     eeg_2 = preprocess_EEG(raw_data[1, :])
#     emg = preprocess_EMG(raw_data[2, :])
#     x = np.stack((eeg_1, eeg_2, emg))
#     x = windowing(x, window_size=32 * 5, window_stride=32)
#     x = np.transpose(x, (0, 3, 2, 1))
#
#     y_1, y_2 = load_labels(labels_path, just_artifact_labels=just_artifact_labels)
#
#     if just_artifact_labels and balance_artifacts:
#         balanced_subset_idx = balanced_artifacts(y_1[2:-2])
#         y_1 = y_1[2:-2][balanced_subset_idx]
#         x = x[balanced_subset_idx]
#
#     if validation_split is not None:
#         rdm_indexes = np.arange(x.shape[0])
#         np.random.shuffle(rdm_indexes)
#         train_indexes = rdm_indexes[:int(len(rdm_indexes) * (1 - validation_split))]
#         val_indexes = rdm_indexes[int(len(rdm_indexes) * (1 - validation_split)):]
#
#         x_train = x[train_indexes]
#         x_val = x[val_indexes]
#         if balance_artifacts == False:
#             labels_train = y_1[2:-2][train_indexes]
#             labels_val = y_1[2:-2][val_indexes]
#         else:
#             labels_train = y_1[train_indexes]
#             labels_val = y_1[val_indexes]
#
#         input_dataset_train = tf.data.Dataset.from_tensor_slices(x_train)
#         labels_dataset_train = tf.data.Dataset.from_tensor_slices(labels_train)
#         input_dataset_val = tf.data.Dataset.from_tensor_slices(x_val)
#         labels_dataset_val = tf.data.Dataset.from_tensor_slices(labels_val)
#         # labels_dataset = labels_dataset.batch(32)
#
#         train_dataset = tf.data.Dataset.zip((input_dataset_train, labels_dataset_train))
#         val_dataset = tf.data.Dataset.zip((input_dataset_val, labels_dataset_val))
#
#         return train_dataset, val_dataset
#     else:
#         input_dataset = tf.data.Dataset.from_tensor_slices(x)
#         if balance_artifacts == False:
#             labels_dataset = tf.data.Dataset.from_tensor_slices(y_1[2:-2])
#         else:
#             labels_dataset = tf.data.Dataset.from_tensor_slices(y_1)
#         dataset = tf.data.Dataset.zip((input_dataset, labels_dataset))
#
#         return dataset