from tensorflow.keras.layers import Input, MaxPool2D, Conv2D, Dense, Flatten, Dropout
from kornum_data_loading import load_to_dataset, load_labels
from metrics import *
from tools import *
plt.ion()

data_path = r"C:\Users\javig\Documents\THESIS_DATA\Raw kornum lab data\Laura-EEGdata_cleaned\data-Kornum"
save_results_path = r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SPINDLE\results\4 - trained on kornum data\B_1'
weights_path = r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SPINDLE\results\4 - trained on kornum data\B_1\B_1_05epochs.h5'
model_name = 'B_1'

plt.ion()

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

spindle_model.load_weights(weights_path)

spindle_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5 * 1e-5,
                                                         beta_1=0.9,
                                                         beta_2=0.999),
                      loss=BinaryWeightedCrossEntropy(),
                      # MulticlassWeightedCrossEntropy_2 tf.keras.losses.CategoricalCrossentropy()
                      metrics=[tf.keras.metrics.BinaryAccuracy(),
                               BinaryBalancedAccuracy(),
                               BinaryF1Score()])


# -------------------------------------------------------------------------------------------------------------------------

labels_paths = [r"2DTUSERVER-Alexandra\tsv\M52-b2.tsv",
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

signal_paths = [r"2DTUSERVER-Alexandra\EDF\M52-b2.edf",
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
#     i=0
    test_dataset = load_to_dataset(signal_path=signal_paths[i],
                                   labels_path=labels_paths[i],
                                   resample_rate=128,
                                   just_artifact_labels=True,
                                   just_stage_labels=False,
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

    if i==0:
        y_true_all = y_true
        cnn_probs_all = cnn_probs
    else:
        y_true_all = np.concatenate((y_true_all, y_true), axis=0)
        cnn_probs_all = np.concatenate((cnn_probs_all, cnn_probs), axis=0)


# ------------------------------------------------ THR=0.5 -------------------------------------------------------------

thr = 0.5

save_path = os.path.join(save_results_path, 'thr_05')
if not os.path.exists(save_path):
    os.makedirs(save_path)

compute_and_save_metrics_cnn2(y_true_all, cnn_probs_all, thr, save_path, model_name)

# ------------------------------------------------ THR=optimal----------------------------------------------------------

thr = np.load(r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SPINDLE\kornum_data\find_optimal_threshold\optimal_thr_kornum.npy')

save_path = os.path.join(save_results_path, 'opt_thr')
if not os.path.exists(save_path):
    os.makedirs(save_path)

compute_and_save_metrics_cnn2(y_true_all, cnn_probs_all, thr, save_path, model_name)


