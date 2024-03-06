from tensorflow.keras.layers import Input, MaxPool2D, Conv2D, Dense, Flatten, Dropout
from metrics import *
from tools import *
from hmm import *
from spindle_data.spindle_data_loading import load_to_dataset, load_labels


def evaluation_metrics_cnn1(y_true, y_pred, filename, metrics_previous=None):
    cm = confusion_matrix(y_true, y_pred)

    ppv = precision_score(y_true, y_pred, average=None)  # positive predictive value
    tpr = recall_score(y_true, y_pred,
                       average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    accuracy = accuracy_score(y_true, y_pred)

    metrics = pd.DataFrame([[filename, accuracy, ppv[0], tpr[0], f1[0], ppv[1], tpr[1], f1[1], ppv[2], tpr[2], f1[2]]], columns=['filename', 'acc', 'ppv_N', 'tpr_N', 'f1_N', 'ppv_R', 'tpr_R', 'f1_R', 'ppv_W', 'tpr_W', 'f1_W'])

    if metrics_previous is not None:
        metrics = pd.concat([metrics_previous, metrics], ignore_index=True)

    return metrics

def evaluation_metrics_whole_model(y_true, y_pred, filename, metrics_previous=None):
    ppv = precision_score(y_true, y_pred, average=None)  # positive predictive value
    tpr = recall_score(y_true, y_pred,
                       average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    accuracy = accuracy_score(y_true, y_pred)

    metrics = pd.DataFrame([[filename, accuracy, ppv[0], tpr[0], f1[0], ppv[1], tpr[1], f1[1], ppv[2], tpr[2], f1[2], ppv[3], tpr[3], f1[3]]], columns=['filename', 'acc', 'ppv_N', 'tpr_N', 'f1_N', 'ppv_R', 'tpr_R', 'f1_R', 'ppv_W', 'tpr_W', 'f1_W', 'ppv_A', 'tpr_A', 'f1_A'])

    if metrics_previous is not None:
        metrics = pd.concat([metrics_previous, metrics], ignore_index=True)

    return metrics

scorer = 1
# save_results_path = r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SPINDLE\results\5 - trained on SPINDLE (Final)\evaluation_brownlab\scorer_' + str(scorer) + '\whole_model'
weights_path_cnn1 = r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SPINDLE\results\4 - trained on kornum data\evaluation on kornum data\A_1\A_1_5e-6_FINAL_05epochs.h5'
weights_path_cnn2 = r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SPINDLE\results\4 - trained on kornum data\evaluation on kornum data\B_1\B_1_05epochs.h5'

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

transition_matrix = np.load(r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SPINDLE\kornum_data\hmm_parameters\transition_matrix_kornum.npy')
initial_probs = np.load(r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SPINDLE\kornum_data\hmm_parameters\initial_probs_kornum.npy')

# -------------------------------------------------------------------------------------------------------------------------

labels_paths = [
    r'C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/scorings/A1.csv',
                r'C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/scorings/A2.csv',
                r'C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/scorings/A3.csv',
                r'C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/scorings/A4.csv'
                ]

signal_paths = [
    r"C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/recordings/A1.edf",
                r"C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/recordings/A2.edf",
                r"C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/recordings/A3.edf",
                r"C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/recordings/A4.edf"
                ]



for i in range(len(signal_paths)):
#     i=0
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

    y_cnn2 = cnn2_probs > 0.5
    y_cnn2 = y_cnn2.numpy()[:, 0]


    y_hmm, _, _ = viterbi(cnn1_probs, transition_matrix, initial_probs)


    y_hmm_arts_filtered = y_hmm[y_true_art == 0]
    y_true_arts_filtered = y_true_stages[y_true_art == 0]

    y_whole = y_hmm
    y_whole[y_cnn2 == 1] = 3
    y_true_whole = y_true_stages
    y_true_whole[y_true_art == 1] = 3

    if i==0:
        metrics_hmm = evaluation_metrics_cnn1(y_true_arts_filtered, y_hmm_arts_filtered, signal_paths[i], metrics_previous=None)
        metrics_whole = evaluation_metrics_whole_model(y_true_whole, y_whole, signal_paths[i], metrics_previous=None)

    else:
        metrics_hmm = evaluation_metrics_cnn1(y_true_arts_filtered, y_hmm_arts_filtered, signal_paths[i], metrics_previous=metrics_hmm)
        metrics_whole = evaluation_metrics_whole_model(y_true_whole, y_whole, signal_paths[i], metrics_previous=metrics_whole)

# ------------------------------------------------ BEFORE HMM ----------------------------------------------------------

a=9
# metrics.to_csv(, index=False)
