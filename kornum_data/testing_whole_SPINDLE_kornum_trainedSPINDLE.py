from tensorflow.keras.layers import Input, MaxPool2D, Conv2D, Dense, Softmax, Flatten, Dropout
from metrics import *
from tools import *
from hmm import *
from kornum_data_loading import load_to_dataset, load_labels

data_path = r"C:\Users\javig\Documents\THESIS_DATA\Raw kornum lab data\Laura-EEGdata_cleaned\data-Kornum"
save_results_path = r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SPINDLE\results\5 - trained on SPINDLE (Final)/evaluation_kornum/whole_model'
weights_path_cnn1 = r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SPINDLE\results\3 - new round of results after meeting\A_1\A_1_5epochs.h5'
weights_path_cnn2 = r"C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SPINDLE\results\3 - new round of results after meeting\B_1\B_1_5epochs.h5"

plt.ion()

spindle_model_1 = tf.keras.Sequential([
    Input((160, 48, 3)),
    MaxPool2D(pool_size=(2, 3), strides=(2, 3)),
    Conv2D(filters=50, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
    MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    Dense(units=1000, activation='relu', kernel_initializer='glorot_uniform'),
    Dropout(0.5),
    Dense(units=1000, activation='relu', kernel_initializer='glorot_uniform'),
    Dropout(0.5),
    Dense(units=3, activation='softmax', kernel_initializer='glorot_uniform')
])

spindle_model_1.load_weights(weights_path_cnn1)

spindle_model_1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5 * 1e-5,
                                                         beta_1=0.9,
                                                         beta_2=0.999),
                      loss=MulticlassWeightedCrossEntropy_2(n_classes=3),
                      metrics=[tf.keras.metrics.CategoricalAccuracy(),
                               MulticlassF1Score(n_classes=3),
                               MulticlassBalancedAccuracy(n_classes=3)])

spindle_model_2 = tf.keras.Sequential([
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

spindle_model_2.load_weights(weights_path_cnn2)

spindle_model_2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5 * 1e-5,
                                                         beta_1=0.9,
                                                         beta_2=0.999),
                      loss=BinaryWeightedCrossEntropy,
                      # BinaryWeightedCrossEntropy tf.keras.losses.BinaryCrossentropy()
                      metrics=[tf.keras.metrics.BinaryAccuracy(),
                               BinaryBalancedAccuracy(),
                               BinaryF1Score()])

transition_matrix = np.load(r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SPINDLE\spindle_data\hmm_parameters\transition_matrix_SPINDLE.npy')
initial_probs = np.load(r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SPINDLE\spindle_data\hmm_parameters\inital_probs_SPINDLE.npy')

# -------------------------------------------------------------------------------------------------------------------------

labels_paths = [
    r"2DTUSERVER-Alexandra\tsv\M52-b2.tsv",
                r"2DTUSERVER-Alexandra\tsv\M58-b2.tsv",
                r"2DTUSERVER-CH\tsv\m13-b1.tsv",
                r"2DTUSERVER-CH\tsv\m15-b1.tsv",
                r"2DTUSERVER-CH\tsv\m6-b1.tsv",
                r"2DTUSERVER-LOUISE\tsv\M18-b2.tsv",
                r"2DTUSERVER-LOUISE\tsv\M313-b1.tsv",
                r"2DTUSERVER-Maria\tsv\m61-b2.tsv",
                r"2DTUSERVER-Maria\tsv\m86-b2.tsv",
                r"2DTUSERVER-Maria\tsv\m94-b1.tsv",
                r"2DTUSERVER-Maria\tsv\m94-b2.tsv"
                ]

signal_paths = [
    r"2DTUSERVER-Alexandra\EDF\M52-b2.edf",
                r"2DTUSERVER-Alexandra\EDF\M58-b2.edf",
                r"2DTUSERVER-CH\EDF\m13-b1.edf",
                r"2DTUSERVER-CH\EDF\m15-b1.edf",
                r"2DTUSERVER-CH\EDF\m6-b1.edf",
                r"2DTUSERVER-LOUISE\EDF\M18-b2.edf",
                r"2DTUSERVER-LOUISE\EDF\M313-b1.edf",
                r"2DTUSERVER-Maria\EDF\m61-b2.edf",
                r"2DTUSERVER-Maria\EDF\m86-b2.edf",
                r"2DTUSERVER-Maria\EDF\m94-b1.edf",
                r"2DTUSERVER-Maria\EDF\m94-b2.edf"
                ]

labels_paths = [os.path.join(data_path, p) for p in labels_paths]
signal_paths = [os.path.join(data_path, p) for p in signal_paths]

for i in range(len(signal_paths)):
    test_dataset = load_to_dataset(signal_path=signal_paths[i],
                                   labels_path=labels_paths[i],
                                   resample_rate=128,
                                   just_artifact_labels=False,
                                   just_stage_labels=True,
                                   validation_split=0)

    batch_size = 100
    test_dataset = test_dataset.batch(batch_size)

    for idx, batch in enumerate(test_dataset):
        if idx == 0:
            cnn1_probs = spindle_model_1(batch[0])
            cnn2_probs = spindle_model_2(batch[0])
        else:
            cnn1_probs = tf.concat([cnn1_probs, spindle_model_1(batch[0])], axis=0)
            cnn2_probs = tf.concat([cnn2_probs, spindle_model_2(batch[0])], axis=0)

    y_hmm, _, _ = viterbi(cnn1_probs, transition_matrix, initial_probs)
    y_cnn2 = (cnn2_probs > 0.5)[:, 0]
    y_whole = y_hmm
    y_whole[y_cnn2] = 3

    y_true = load_labels(labels_paths[i],
                        just_artifact_labels=False,
                        just_stage_labels=False)
    y_true = y_true.to_numpy()
    y_true = np.argmax(y_true, axis=1)

    if i==0:
        y_true_all = y_true
        y_whole_all = y_whole
    else:
        y_true_all = np.concatenate((y_true_all, y_true), axis=0)
        y_whole_all = np.concatenate((y_whole_all, y_whole), axis=0)


if not os.path.exists(save_results_path):
    os.makedirs(save_results_path)

compute_and_save_metrics_whole_model(y_true_all, y_whole_all, save_results_path)



