from tensorflow.keras.layers import Input, MaxPool2D, Conv2D, Dense, Flatten, Dropout
from kornum_data.kornum_data_loading import SequenceDataset
from metrics import *
from tools import *

plt.ion()

save_path = r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SPINDLE\results\4 - trained on kornum data'
model_name = 'A_1'
data_path = r'C:\Users\javig\Documents\THESIS DATA\Raw kornum lab data\to_numpy\spectrograms'
csv_path = r"C:\Users\javig\Documents\THESIS DATA\Raw kornum lab data\to_numpy\labels_all.csv"

# -------------------------------------------------------------------------------------------------------------------------


train_sequence = SequenceDataset(data_path,
                                 csv_path,
                                 'train',
                                 100,
                                 False,
                                 True)

val_sequence = SequenceDataset(data_path,
                               csv_path,
                               'validation',
                               100,
                               False,
                               True)

# -------------------------------------------------------------------------------------------------------------------------


spindle_model = tf.keras.Sequential([
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

checkpoint_path = os.path.join(save_path, model_name, model_name + "_{epoch:02d}epochs" + ".h5")
if not os.path.exists(os.path.dirname(checkpoint_path)):
    os.makedirs(os.path.dirname(checkpoint_path))
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    monitor='val_loss',
    filepath=checkpoint_path,
    save_weights_only=True,
    save_freq='epoch',
    verbose=1)

spindle_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5 * 1e-5,
                                                         beta_1=0.9,
                                                         beta_2=0.999),
                      loss=BinaryWeightedCrossEntropy(),
                      # MulticlassWeightedCrossEntropy_2 tf.keras.losses.CategoricalCrossentropy()
                      metrics=[tf.keras.metrics.BinaryAccuracy(),
                               BinaryBalancedAccuracy(),
                               BinaryF1Score()])
# run_eagerly=True)

history1 = spindle_model.fit(
    x=train_sequence,
    validation_data=val_sequence,
    epochs=10,
    verbose=1,
    callbacks=[model_checkpoint_callback]
)

plot_history_cnn2(history1.history, model_name, save_path, epochs=10)

# spindle_model.save_weights(os.path.join(save_path, model_name, model_name + "_5epochs" + ".h5"))

# history2 = spindle_model.fit(x=train_dataset,
#                             validation_data=val_dataset,
#                             epochs=5,
#                             verbose=1)
#
# plot_history_cnn1(history2, model_name, save_path, epochs=5)
#
# spindle_model.save_weights(os.path.join(save_path, model_name, model_name + "_10epochs" + ".h5"))


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
#     # original
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


# def load_recording_to_dataset_2(signal_path, labels_path, validation_split=None, to_numpy=False): # stft_size, stft_stride, fs, epoch_length,
#     # like load_recording_to_dataset_1 but with training and validation subsets and shuffling
#
#     y_1, y_2 = load_labels(labels_path, artifact_to_stages=True)
#
#     # epochs_analysis(y_1, y_2)
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


# def plot_history(history, model_name, save_path):
#     fig, ax = plt.subplots(3, 1, figsize=(5, 12))
#
#     ax[0].plot(history.history['categorical_accuracy'])
#     ax[0].plot(history.history['val_categorical_accuracy'])
#     ax[0].plot(history.history['multiclass_balanced_accuracy'])
#     ax[0].plot(history.history['val_multiclass_balanced_accuracy'])
#     ax[0].legend(['Training standard', 'Validation standard', 'Training balanced', 'Validation balanced'])
#     ax[0].set_title('Accuracy')
#
#     # ax[1].plot(history.history['multiclass_balanced_accuracy'])
#     # ax[1].plot(history.history['val_multiclass_balanced_accuracy'])
#     # ax[1].legend(['Training', 'Validation'])
#     # ax[1].set_title('Balanced categorical accuracy')
#
#     ax[1].plot(history.history['loss'])
#     ax[1].plot(history.history['val_loss'])
#     ax[1].legend(['Training', 'Validation'])
#     ax[1].set_title('Weighted categorical cross entropy')
#
#     ax[2].plot(history.history['multiclass_F1_score'])
#     ax[2].plot(history.history['val_multiclass_F1_score'])
#     ax[2].legend(['Training', 'Validation'])
#     ax[2].set_title('Average F1 score')
#
#     fig.suptitle(model_name)
#
#     save_path = os.path.join(save_path, model_name, 'training_curve2.jpg')
#     if not os.path.exists(os.path.dirname(save_path)):
#         os.makedirs(os.path.dirname(save_path))
#     plt.savefig(save_path)
