from tensorflow.keras.layers import Input, MaxPool2D, Conv2D, Dense, Softmax, Flatten, Dropout
from metrics import *
from tools import *
from hmm import *
from kornum_data_loading import load_to_dataset, load_labels


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


# saved_model_folder = r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\SPINDLE_pycharm\results\3 - new round of results after meeting'
# save_path = r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\SPINDLE\results\3 - new round of results after meeting\A_1\Evaluation\Against scorers intersection\Excluding artifacts\Before HMM'
# save_results_path = r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SPINDLE\results\4 - trained on kornum data\A_1'
# model_name = 'A_1'
weights_path = r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SPINDLE\results\4 - trained on kornum data\evaluation on kornum data\A_1\A_1_5e-6_FINAL_05epochs.h5'

plt.ion()

transition_matrix = np.load(r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SPINDLE\kornum_data\hmm_parameters\transition_matrix_kornum.npy')
initial_probs = np.load(r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SPINDLE\kornum_data\hmm_parameters\initial_probs_kornum.npy')

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
    y_hmm_withArts_filtered = y_hmm_withArts[y_art == 0]

    if i==0:
        metrics = evaluation_metrics_cnn1(y_true_filtered, y_hmm_withArts_filtered, signal_paths[i], metrics_previous=None)
    else:
        metrics = evaluation_metrics_cnn1(y_true_filtered, y_hmm_withArts_filtered, signal_paths[i], metrics_previous=metrics)
# ------------------------------------------------ BEFORE HMM ----------------------------------------------------------

a=9
# metrics.to_csv(, index=False)
