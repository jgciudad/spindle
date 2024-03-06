from tensorflow.keras.layers import Input, MaxPool2D, Conv2D, Dense, Flatten, Dropout
from spindle_data.spindle_data_loading import load_to_dataset
from metrics import *
from tools import *

save_results_path = r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SPINDLE\results\4 - trained on kornum data\evaluation on spindle data\scorer 2\B_1'
weights_path = r"C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SPINDLE\results\4 - trained on kornum data\evaluation on kornum data\B_1\B_1_05epochs.h5"
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


labels_paths = [r'C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/scorings/A1.csv',
                r'C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/scorings/A2.csv',
                r'C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/scorings/A3.csv',
                r'C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/scorings/A4.csv'
                ]

signal_paths = [r"C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/recordings/A1.edf",
                r"C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/recordings/A2.edf",
                r"C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/recordings/A3.edf",
                r"C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/recordings/A4.edf"
                ]

for i in range(len(signal_paths)):
# i=0
    test_dataset = load_to_dataset(signal_paths=[signal_paths[i]],
                                   labels_paths=[labels_paths[i]],
                                   scorer=2,
                                   just_artifact_labels=True,
                                   artifact_to_stages=False,
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


