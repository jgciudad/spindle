from tensorflow.keras.layers import Input, MaxPool2D, Conv2D, Dense, Softmax, Flatten, Dropout
from metrics import *
from tools import *
from hmm import *
from kornum_data_loading import load_to_dataset, load_labels


# saved_model_folder = r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\SPINDLE_pycharm\results\3 - new round of results after meeting'
# save_path = r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\SPINDLE\results\3 - new round of results after meeting\A_1\Evaluation\Against scorers intersection\Excluding artifacts\Before HMM'
save_results_path = r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SPINDLE\results\5 - trained on SPINDLE (Final)/evaluation_kornum/CNN1'
model_name = 'A_1'
weights_path = r"C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SPINDLE\results\3 - new round of results after meeting\A_1\A_1_5epochs.h5"

plt.ion()

transition_matrix = np.load(r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SPINDLE\spindle_data\hmm_parameters\transition_matrix_SPINDLE.npy')
initial_probs = np.load(r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SPINDLE\spindle_data\hmm_parameters\inital_probs_SPINDLE.npy')

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
    Dense(units=3, activation='softmax', kernel_initializer='glorot_uniform')
])

spindle_model.load_weights(weights_path)

spindle_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5 * 1e-5,
                                                         beta_1=0.9,
                                                         beta_2=0.999),
                      loss=MulticlassWeightedCrossEntropy_2(n_classes=3),
                      metrics=[tf.keras.metrics.CategoricalAccuracy(),
                               MulticlassF1Score(n_classes=3),
                               MulticlassBalancedAccuracy(n_classes=3)])

# -------------------------------------------------------------------------------------------------------------------------

labels_paths = [r"C:\Users\javig\Documents\THESIS_DATA\Raw kornum lab data\Laura-EEGdata_cleaned\data-Kornum\2DTUSERVER-Alexandra\tsv\M52-b2.tsv",
                r"C:\Users\javig\Documents\THESIS_DATA\Raw kornum lab data\Laura-EEGdata_cleaned\data-Kornum\2DTUSERVER-Alexandra\tsv\M58-b2.tsv",
                r"C:\Users\javig\Documents\THESIS_DATA\Raw kornum lab data\Laura-EEGdata_cleaned\data-Kornum\2DTUSERVER-CH\tsv\m13-b1.tsv",
                r"C:\Users\javig\Documents\THESIS_DATA\Raw kornum lab data\Laura-EEGdata_cleaned\data-Kornum\2DTUSERVER-CH\tsv\m15-b1.tsv",
                r"C:\Users\javig\Documents\THESIS_DATA\Raw kornum lab data\Laura-EEGdata_cleaned\data-Kornum\2DTUSERVER-CH\tsv\m6-b1.tsv",
                r"C:\Users\javig\Documents\THESIS_DATA\Raw kornum lab data\Laura-EEGdata_cleaned\data-Kornum\2DTUSERVER-LOUISE\tsv\M18-b2.tsv",
                r"C:\Users\javig\Documents\THESIS_DATA\Raw kornum lab data\Laura-EEGdata_cleaned\data-Kornum\2DTUSERVER-LOUISE\tsv\M313-b1.tsv",
                r"C:\Users\javig\Documents\THESIS_DATA\Raw kornum lab data\Laura-EEGdata_cleaned\data-Kornum\2DTUSERVER-Maria\tsv\m61-b2.tsv",
                r"C:\Users\javig\Documents\THESIS_DATA\Raw kornum lab data\Laura-EEGdata_cleaned\data-Kornum\2DTUSERVER-Maria\tsv\m86-b2.tsv",
                r"C:\Users\javig\Documents\THESIS_DATA\Raw kornum lab data\Laura-EEGdata_cleaned\data-Kornum\2DTUSERVER-Maria\tsv\m94-b1.tsv",
                r"C:\Users\javig\Documents\THESIS_DATA\Raw kornum lab data\Laura-EEGdata_cleaned\data-Kornum\2DTUSERVER-Maria\tsv\m94-b2.tsv"
                ]


signal_paths = [r"C:\Users\javig\Documents\THESIS_DATA\Raw kornum lab data\Laura-EEGdata_cleaned\data-Kornum\2DTUSERVER-Alexandra\EDF\M52-b2.edf",
                r"C:\Users\javig\Documents\THESIS_DATA\Raw kornum lab data\Laura-EEGdata_cleaned\data-Kornum\2DTUSERVER-Alexandra\EDF\M58-b2.edf",
                r"C:\Users\javig\Documents\THESIS_DATA\Raw kornum lab data\Laura-EEGdata_cleaned\data-Kornum\2DTUSERVER-CH\EDF\m13-b1.edf",
                r"C:\Users\javig\Documents\THESIS_DATA\Raw kornum lab data\Laura-EEGdata_cleaned\data-Kornum\2DTUSERVER-CH\EDF\m15-b1.edf",
                r"C:\Users\javig\Documents\THESIS_DATA\Raw kornum lab data\Laura-EEGdata_cleaned\data-Kornum\2DTUSERVER-CH\EDF\m6-b1.edf",
                r"C:\Users\javig\Documents\THESIS_DATA\Raw kornum lab data\Laura-EEGdata_cleaned\data-Kornum\2DTUSERVER-LOUISE\EDF\M18-b2.edf",
                r"C:\Users\javig\Documents\THESIS_DATA\Raw kornum lab data\Laura-EEGdata_cleaned\data-Kornum\2DTUSERVER-LOUISE\EDF\M313-b1.edf",
                r"C:\Users\javig\Documents\THESIS_DATA\Raw kornum lab data\Laura-EEGdata_cleaned\data-Kornum\2DTUSERVER-Maria\EDF\m61-b2.edf",
                r"C:\Users\javig\Documents\THESIS_DATA\Raw kornum lab data\Laura-EEGdata_cleaned\data-Kornum\2DTUSERVER-Maria\EDF\m86-b2.edf",
                r"C:\Users\javig\Documents\THESIS_DATA\Raw kornum lab data\Laura-EEGdata_cleaned\data-Kornum\2DTUSERVER-Maria\EDF\m94-b1.edf",
                r"C:\Users\javig\Documents\THESIS_DATA\Raw kornum lab data\Laura-EEGdata_cleaned\data-Kornum\2DTUSERVER-Maria\EDF\m94-b2.edf"
                ]


for i in range(len(signal_paths)):
#     i=0
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
            cnn_probs = spindle_model(batch[0])
            y_true = batch[1]
        else:
            cnn_probs = tf.concat([cnn_probs, spindle_model(batch[0])], axis=0)
            y_true = tf.concat([y_true, batch[1]], axis=0)

    y_art = load_labels(labels_paths[i],
                        just_artifact_labels=True,
                        just_stage_labels=False)
    y_art = y_art.to_numpy()

    y_true = np.argmax(y_true, axis=1)
    y_true_filtered = y_true[y_art == 0]

    y_cnn = np.argmax(cnn_probs, axis=1)
    cnn_probs_filtered = cnn_probs[y_art == 0]
    y_cnn_filtered = np.argmax(cnn_probs_filtered, axis=1)

    y_hmm_withArts, _, _ = viterbi(cnn_probs, transition_matrix, initial_probs)
    y_hmm_filtered, _, _ = viterbi(cnn_probs_filtered, transition_matrix, initial_probs)
    y_hmm_withArts_filtered = y_hmm_withArts[y_art == 0]

    if i==0:
        y_true_filtered_all = y_true_filtered
        y_cnn_filtered_all = y_cnn_filtered
        y_true_all = y_true
        y_hmm_all = y_hmm_withArts
        y_hmm_filtered_all = y_hmm_filtered
        y_hmm_withArts_filtered_all = y_hmm_withArts_filtered
        y_hmm_withArts_all = y_hmm_withArts
        y_cnn_all = y_cnn
        y_art_all = y_art
    else:
        y_true_filtered_all = np.concatenate((y_true_filtered_all, y_true_filtered), axis=0)
        y_cnn_filtered_all = np.concatenate((y_cnn_filtered_all, y_cnn_filtered), axis=0)
        y_true_all = np.concatenate((y_true_all, y_true), axis=0)
        y_hmm_all = np.concatenate((y_hmm_all, y_hmm_withArts), axis=0)
        y_hmm_filtered_all = np.concatenate((y_hmm_filtered_all, y_hmm_filtered), axis=0)
        y_hmm_withArts_filtered_all = np.concatenate((y_hmm_withArts_filtered_all, y_hmm_withArts_filtered), axis=0)
        y_hmm_withArts_all = np.concatenate((y_hmm_withArts_all, y_hmm_withArts), axis=0)
        y_cnn_all = np.concatenate((y_cnn_all, y_cnn), axis=0)
        y_art_all = np.concatenate((y_art_all, y_art), axis=0)


# ------------------------------------------------ BEFORE HMM ----------------------------------------------------------

save_path = os.path.join(save_results_path, 'Before HMM')
if not os.path.exists(save_path):
    os.makedirs(save_path)

compute_and_save_metrics_cnn1(y_true_filtered_all, y_cnn_filtered_all, save_path, model_name)

# ------------------------------------------------ AFTER HMM (with artifacts) -----------------------------------------------------------

n_corrected_epochs, correction_matrix, cnn_transitions, hmm_transitions = evaluate_hmm_withArts_effect(y_true_all, y_art_all, y_cnn_all, y_hmm_withArts_all)

save_path = os.path.join(save_results_path, 'After HMM (with artifacts)')
if not os.path.exists(save_path):
    os.makedirs(save_path)

plot_and_save_hmm_effect(n_corrected_epochs,
                         correction_matrix,
                         cnn_transitions,
                         hmm_transitions,
                         save_path)

# CONCATENATE
compute_and_save_metrics_cnn1(y_true_filtered_all, y_hmm_withArts_filtered_all, save_path, model_name)

# ------------------------------------------------ AFTER HMM (artifacts filtered) -----------------------------------------------------------


n_corrected_epochs, correction_matrix, cnn_transitions, hmm_transitions = evaluate_hmm_effect(y_true_filtered_all, y_cnn_filtered_all, y_hmm_filtered_all)

save_path = os.path.join(save_results_path, 'After HMM (artifacts filtered)')
if not os.path.exists(save_path):
    os.makedirs(save_path)

plot_and_save_hmm_effect(n_corrected_epochs,
                         correction_matrix,
                         cnn_transitions,
                         hmm_transitions,
                         save_path)

# CONCATENATE
compute_and_save_metrics_cnn1(y_true_filtered_all, y_hmm_filtered_all, save_path, model_name)

