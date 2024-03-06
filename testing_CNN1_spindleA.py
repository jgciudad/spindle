from tensorflow.keras.layers import Input, MaxPool2D, Conv2D, Dense, Flatten, Dropout
from metrics import *
from tools import *
from hmm import *
from spindle_data.spindle_data_loading import load_to_dataset, load_labels

cohort = 'a'
scorer=1
model_name = 'A_1'

save_results_path = os.path.join('/Users/tlj258/Documents/HUMMUSS_paper/outputs/spindle', model_name, 'testing', 'spindle', 'cohort_' + cohort.upper(), 'scorer_' + str(scorer))

transition_matrix = np.load('/Users/tlj258/Documents/Code/EEG_scoring/SPINDLE/kornum_data/hmm_parameters/transition_matrix_kornum.npy')
initial_probs = np.load('/Users/tlj258/Documents/Code/EEG_scoring/SPINDLE/kornum_data/hmm_parameters/initial_probs_kornum.npy')

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

spindle_model.load_weights(os.path.join('/Users/tlj258/Documents/HUMMUSS_paper/outputs/spindle', model_name, 'best_model.h5'))

spindle_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5 * 1e-5,
                                                         beta_1=0.9,
                                                         beta_2=0.999),
                      loss=MulticlassWeightedCrossEntropy_2(n_classes=3),
                      metrics=[tf.keras.metrics.CategoricalAccuracy(),
                               MulticlassF1Score(n_classes=3),
                               MulticlassBalancedAccuracy(n_classes=3)])

# -------------------------------------------------------------------------------------------------------------------------


# labels_paths = [
#     '/Users/tlj258/Documents/THESIS_DATA/SPINDLE/data (original)/CohortA/scorings/A1.csv',
#     '/Users/tlj258/Documents/THESIS_DATA/SPINDLE/data (original)/CohortA/scorings/A2.csv',
#     '/Users/tlj258/Documents/THESIS_DATA/SPINDLE/data (original)/CohortA/scorings/A3.csv',
#     '/Users/tlj258/Documents/THESIS_DATA/SPINDLE/data (original)/CohortA/scorings/A4.csv'
#     ]

# signal_paths = [
    # '/Users/tlj258/Documents/THESIS_DATA/SPINDLE/data (original)/CohortA/recordings/A1.edf',
    # '/Users/tlj258/Documents/THESIS_DATA/SPINDLE/data (original)/CohortA/recordings/A2.edf',
    # '/Users/tlj258/Documents/THESIS_DATA/SPINDLE/data (original)/CohortA/recordings/A3.edf',
    # '/Users/tlj258/Documents/THESIS_DATA/SPINDLE/data (original)/CohortA/recordings/A4.edf'
    # ]

labels_paths = [
    # '/Users/tlj258/Documents/THESIS_DATA/SPINDLE/data (original)/CohortD/scorings/D1.csv',
    '/Users/tlj258/Documents/THESIS_DATA/SPINDLE/data (original)/CohortD/scorings/D2.csv',
    '/Users/tlj258/Documents/THESIS_DATA/SPINDLE/data (original)/CohortD/scorings/D3.csv',
    '/Users/tlj258/Documents/THESIS_DATA/SPINDLE/data (original)/CohortD/scorings/D4.csv',
    '/Users/tlj258/Documents/THESIS_DATA/SPINDLE/data (original)/CohortD/scorings/D5.csv',
    '/Users/tlj258/Documents/THESIS_DATA/SPINDLE/data (original)/CohortD/scorings/D6.csv'
    ]

signal_paths = [
    '/Users/tlj258/Documents/THESIS_DATA/SPINDLE/data (original)/CohortD/recordings/D1.edf',
    '/Users/tlj258/Documents/THESIS_DATA/SPINDLE/data (original)/CohortD/recordings/D2.edf',
    '/Users/tlj258/Documents/THESIS_DATA/SPINDLE/data (original)/CohortD/recordings/D3.edf',
    '/Users/tlj258/Documents/THESIS_DATA/SPINDLE/data (original)/CohortD/recordings/D4.edf',
    '/Users/tlj258/Documents/THESIS_DATA/SPINDLE/data (original)/CohortD/recordings/D5.edf',
    '/Users/tlj258/Documents/THESIS_DATA/SPINDLE/data (original)/CohortD/recordings/D6.edf'
    ]

for i in range(len(signal_paths)):
# i=0
    test_dataset = load_to_dataset(signal_paths=[signal_paths[i]],
                                   labels_paths=[labels_paths[i]],
                                   scorer=scorer,
                                   just_artifact_labels=False,
                                   artifact_to_stages=True,
                                   balance_artifacts=False,
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

    y_art = load_labels(labels_path=labels_paths[i],
                        scorer=scorer,
                        just_artifact_labels=True,
                        artifact_to_stages=False)
    y_art = y_art.to_numpy()

    y_true = np.argmax(y_true, axis=1)
    y_true_filtered = y_true[y_art == 0]

    y_cnn = np.argmax(cnn_probs, axis=1)
    cnn_probs_filtered = cnn_probs[y_art == 0]
    y_cnn_filtered = np.argmax(cnn_probs_filtered, axis=1)

    y_hmm_withArts, _, _ = viterbi(cnn_probs, transition_matrix, initial_probs)
    y_hmm_withArts_filtered = y_hmm_withArts[y_art == 0]
    y_hmm_noArts, _, _ = viterbi(cnn_probs_filtered, transition_matrix, initial_probs)

    if i==0:
        y_true_all = y_true
        y_true_filtered_all = y_true_filtered
        y_art_all = y_art

        y_cnn_all = y_cnn
        y_cnn_filtered_all = y_cnn_filtered

        y_hmm_all = y_hmm_withArts
        y_hmm_withArts_filtered_all = y_hmm_withArts_filtered
        y_hmm_noArts_all = y_hmm_noArts
        # y_hmm_withArts_all = y_hmm_withArts
    else:
        y_true_all = np.concatenate((y_true_all, y_true), axis=0)
        y_true_filtered_all = np.concatenate((y_true_filtered_all, y_true_filtered), axis=0)
        y_art_all = np.concatenate((y_art_all, y_art), axis=0)

        y_cnn_all = np.concatenate((y_cnn_all, y_cnn), axis=0)
        y_cnn_filtered_all = np.concatenate((y_cnn_filtered_all, y_cnn_filtered), axis=0)

        y_hmm_all = np.concatenate((y_hmm_all, y_hmm_withArts), axis=0)
        y_hmm_withArts_filtered_all = np.concatenate((y_hmm_withArts_filtered_all, y_hmm_withArts_filtered), axis=0)
        y_hmm_noArts_all = np.concatenate((y_hmm_noArts_all, y_hmm_noArts), axis=0)
        # y_hmm_withArts_all = np.concatenate((y_hmm_withArts_all, y_hmm_withArts), axis=0)

# ------------------------------------------------ BEFORE HMM ----------------------------------------------------------

save_path = os.path.join(os.path.join(save_results_path, 'Before HMM'))
if not os.path.exists(save_path):
    os.makedirs(save_path)

compute_and_save_metrics_cnn1(y_true_filtered_all, y_cnn_filtered_all, save_path, model_name)

# ------------------------------------------------ AFTER HMM (with artifacts) -----------------------------------------------------------

# n_corrected_epochs, correction_matrix, cnn_transitions, hmm_transitions = evaluate_hmm_withArts_effect(y_true_all, y_art_all, y_cnn_all, y_hmm_all)

save_path = os.path.join(os.path.join(save_results_path, 'After HMM (with artifacts)'))
if not os.path.exists(save_path):
    os.makedirs(save_path)

# plot_and_save_hmm_effect(n_corrected_epochs,
#                          correction_matrix,
#                          cnn_transitions,
#                          hmm_transitions,
#                          save_path)

# CONCATENATE
compute_and_save_metrics_cnn1(y_true_filtered_all, y_hmm_withArts_filtered_all, save_path, model_name)

# ------------------------------------------------ AFTER HMM (artifacts filtered) -----------------------------------------------------------


# n_corrected_epochs, correction_matrix, cnn_transitions, hmm_transitions = evaluate_hmm_effect(y_true_filtered_all, y_cnn_filtered_all, y_hmm_noArts_all)

save_path = os.path.join(os.path.join(save_results_path, 'After HMM (artifacts filtered)'))
if not os.path.exists(save_path):
    os.makedirs(save_path)

# plot_and_save_hmm_effect(n_corrected_epochs,
#                          correction_matrix,
#                          cnn_transitions,
#                          hmm_transitions,
#                          save_path)

# CONCATENATE
compute_and_save_metrics_cnn1(y_true_filtered_all, y_hmm_noArts_all, save_path, model_name)