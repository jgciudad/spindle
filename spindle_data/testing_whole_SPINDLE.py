from tensorflow.keras.layers import Input, MaxPool2D, Conv2D, Dense, Flatten, Dropout
from spindle_data.spindle_data_loading import load_to_dataset, load_labels
from metrics import *
from tools import *
from hmm import *

scorer = 2
save_results_path = r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SPINDLE\results\5 - trained on SPINDLE (Final)\evaluation_brownlab\scorer_' + str(scorer) + '\whole_model'
weights_path_cnn1 = r"C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SPINDLE\results\3 - new round of results after meeting\A_1\A_1_5epochs.h5"
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
    # r'C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/scorings/A1.csv',
    #             r'C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/scorings/A2.csv',
                r'C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/scorings/A3.csv',
                r'C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/scorings/A4.csv'
                ]

signal_paths = [
    # r"C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/recordings/A1.edf",
    #             r"C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/recordings/A2.edf",
                r"C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/recordings/A3.edf",
                r"C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/recordings/A4.edf"
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
            cnn1_probs = spindle_model_1(batch[0])
            cnn2_probs = spindle_model_2(batch[0])
            y_true_stages = batch[1]
        else:
            cnn1_probs = tf.concat([cnn1_probs, spindle_model_1(batch[0])], axis=0)
            cnn2_probs = tf.concat([cnn2_probs, spindle_model_2(batch[0])], axis=0)
            y_true_stages = tf.concat([y_true_stages, batch[1]], axis=0)

    y_true_art = load_labels(labels_path=labels_paths[i],
                             scorer=scorer,
                             just_artifact_labels=True,
                             artifact_to_stages=False)
    y_true_art = y_true_art.to_numpy()

    y_true_stages = np.argmax(y_true_stages, axis=1)
    # y_true_stages_filtered = y_true[y_art == 0]

    y_cnn1 = np.argmax(cnn1_probs, axis=1)
    y_cnn2 = cnn2_probs > 0.5
    y_cnn2 = y_cnn2.numpy()[:, 0]

    cnn1_probs_filtered = cnn1_probs[y_cnn2 == 0]
    # y_cnn_filtered = np.argmax(cnn_probs_filtered, axis=1)

    y_hmm_withArts, _, _ = viterbi(cnn1_probs, transition_matrix, initial_probs)
    y_hmm_filtered, _, _ = viterbi(cnn1_probs_filtered, transition_matrix, initial_probs)
    # y_hmm_withArts_filtered = y_hmm_withArts[y_art == 0]

    # y_hmm, _, _ = viterbi(cnn1_probs, transition_matrix, initial_probs)

    y_whole_withArts = y_hmm_withArts
    y_whole_withArts[y_cnn2] = 3
    y_whole_filtered = np.zeros(y_true_stages.shape)
    y_whole_filtered[y_cnn2] = 3
    y_whole_filtered[y_cnn2 == 0] = y_hmm_filtered
    y_true_whole = y_true_stages
    y_true_whole[y_true_art == 1] = 3

    if i==0:
        y_true_whole_all = y_true_whole
        y_whole_filtered_all = y_whole_filtered
        y_whole_withArts_all = y_whole_withArts
    else:
        y_true_whole_all = np.concatenate((y_true_whole_all, y_true_whole), axis=0)
        y_whole_filtered_all = np.concatenate((y_whole_filtered_all, y_whole_filtered), axis=0)
        y_whole_withArts_all = np.concatenate((y_whole_withArts_all, y_whole_withArts), axis=0)


# HMM WITH ARTS
save_path = os.path.join(save_results_path, 'hmm_arts')
if not os.path.exists(save_path):
    os.makedirs(save_path)

compute_and_save_metrics_whole_model(y_true_whole_all, y_whole_withArts_all, save_path)


# HMM WITHOUT ARTS
save_path = os.path.join(save_results_path, 'hmm_NO_arts')
if not os.path.exists(save_path):
    os.makedirs(save_path)

compute_and_save_metrics_whole_model(y_true_whole_all, y_whole_filtered_all, save_path)