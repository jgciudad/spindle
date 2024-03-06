from tensorflow.keras.layers import Input, MaxPool2D, Conv2D, Dense, Softmax, Flatten, Dropout
from metrics import *
from tools import *
from hmm import *
from kornum_data.kornum_data_loading import load_to_dataset, load_labels
from sklearn import metrics

weights_path = r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SPINDLE\results\4 - trained on kornum data\evaluation on kornum data\B_1\B_1_05epochs.h5'

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

data_path = r"C:\Users\javig\Documents\THESIS_DATA\Raw kornum lab data\Laura-EEGdata_cleaned\data-Kornum"

training_labels = [r"2DTUSERVER-Alexandra\tsv\M23-b1.tsv",
                   r"2DTUSERVER-Alexandra\tsv\M23-b2.tsv",
                   r"2DTUSERVER-Alexandra\tsv\M23-b3.tsv",
                   r"2DTUSERVER-Alexandra\tsv\M29-b1.tsv",
                   r"2DTUSERVER-Alexandra\tsv\M29-b2.tsv",
                   r"2DTUSERVER-Alexandra\tsv\M29-b3.tsv",
                   r"2DTUSERVER-Alexandra\tsv\M48-b1.tsv",
                   r"2DTUSERVER-Alexandra\tsv\M48-b2.tsv",
                   r"2DTUSERVER-Alexandra\tsv\M48-b3.tsv",
                   r"2DTUSERVER-Alexandra\tsv\M52-b1.tsv",
                   r'2DTUSERVER-Alexandra\tsv\M52-b3.tsv',
                   r'2DTUSERVER-Alexandra\tsv\M58-b1.tsv',
                   r'2DTUSERVER-Alexandra\tsv\M58-b3.tsv',
                   r'2DTUSERVER-CH\tsv\m1-b1.tsv',
                   r'2DTUSERVER-CH\tsv\m11-b1.tsv',
                   r'2DTUSERVER-CH\tsv\m12-b1.tsv',
                   r'2DTUSERVER-CH\tsv\m14-b1.tsv',
                   r'2DTUSERVER-CH\tsv\m14-r3.tsv',
                   r'2DTUSERVER-CH\tsv\m15-r3.tsv',
                   r'2DTUSERVER-CH\tsv\m2-b1.tsv',
                   r'2DTUSERVER-CH\tsv\m3-b1.tsv',
                   r'2DTUSERVER-CH\tsv\m4-b1.tsv',
                   r'2DTUSERVER-CH\tsv\m5-b1.tsv',
                   r'2DTUSERVER-CH\tsv\m6-r3.tsv',
                   r'2DTUSERVER-CH\tsv\m7-b1.tsv',
                   r'2DTUSERVER-CH\tsv\m8-b1.tsv',
                   r'2DTUSERVER-CH\tsv\m8-r3.tsv',
                   r'2DTUSERVER-LOUISE\tsv\M16-b2.tsv',
                   r'2DTUSERVER-LOUISE\tsv\M16-b3.tsv',
                   r'2DTUSERVER-LOUISE\tsv\M18-b3.tsv',
                   r'2DTUSERVER-LOUISE\tsv\M20-b3.tsv',
                   r'2DTUSERVER-LOUISE\tsv\M309-b1.tsv',
                   r'2DTUSERVER-Maria\tsv\m121-b1.tsv',
                   r'2DTUSERVER-Maria\tsv\m121-b2.tsv',
                   r'2DTUSERVER-Maria\tsv\m61-b1.tsv',
                   r'2DTUSERVER-Maria\tsv\m63-b1.tsv',
                   r'2DTUSERVER-Maria\tsv\m63-b2.tsv',
                   r'2DTUSERVER-Maria\tsv\m86-b1.tsv',
                   r'2DTUSERVER-Maria\tsv\m88-b1.tsv',
                   r'2DTUSERVER-Maria\tsv\m88-b2.tsv',
                   r'2DTUSERVER-Maria\tsv\m96-b1.tsv',
                   r'2DTUSERVER-Maria\tsv\m96-b2.tsv',
                   r'2DTUSERVER-Marieke\tsv\m2-b1.tsv',
                   r'2DTUSERVER-Marieke\tsv\m21-b1.tsv']

training_signals = [r"2DTUSERVER-Alexandra\EDF\M23-b1.edf",
                    r"2DTUSERVER-Alexandra\EDF\M23-b2.edf",
                    r"2DTUSERVER-Alexandra\EDF\M23-b3.edf",
                    r"2DTUSERVER-Alexandra\EDF\M29-b1.edf",
                    r"2DTUSERVER-Alexandra\EDF\M29-b2.edf",
                    r"2DTUSERVER-Alexandra\EDF\M29-b3.edf",
                    r"2DTUSERVER-Alexandra\EDF\M48-b1.edf",
                    r"2DTUSERVER-Alexandra\EDF\M48-b2.edf",
                    r"2DTUSERVER-Alexandra\EDF\M48-b3.edf",
                    r"2DTUSERVER-Alexandra\EDF\M52-b1.edf",
                    r'2DTUSERVER-Alexandra\EDF\M52-b3.edf',
                    r'2DTUSERVER-Alexandra\EDF\M58-b1.edf',
                    r'2DTUSERVER-Alexandra\EDF\M58-b3.edf',
                    r'2DTUSERVER-CH\EDF\m1-b1.edf',
                    r'2DTUSERVER-CH\EDF\m11-b1.edf',
                    r'2DTUSERVER-CH\EDF\m12-b1.edf',
                    r'2DTUSERVER-CH\EDF\m14-b1.edf',
                    r'2DTUSERVER-CH\EDF\m14-r3.edf',
                    r'2DTUSERVER-CH\EDF\m15-r3.edf',
                    r'2DTUSERVER-CH\EDF\m2-b1.edf',
                    r'2DTUSERVER-CH\EDF\m3-b1.edf',
                    r'2DTUSERVER-CH\EDF\m4-b1.edf',
                    r'2DTUSERVER-CH\EDF\m5-b1.edf',
                    r'2DTUSERVER-CH\EDF\m6-r3.edf',
                    r'2DTUSERVER-CH\EDF\m7-b1.edf',
                    r'2DTUSERVER-CH\EDF\m8-b1.edf',
                    r'2DTUSERVER-CH\EDF\m8-r3.edf',
                    r'2DTUSERVER-LOUISE\EDF\M16-b2.edf',
                    r'2DTUSERVER-LOUISE\EDF\M16-b3.edf',
                    r'2DTUSERVER-LOUISE\EDF\M18-b3.edf',
                    r'2DTUSERVER-LOUISE\EDF\M20-b3.edf',
                    r'2DTUSERVER-LOUISE\EDF\M309-b1.edf',
                    r'2DTUSERVER-Maria\EDF\m121-b1.edf',
                    r'2DTUSERVER-Maria\EDF\m121-b2.edf',
                    r'2DTUSERVER-Maria\EDF\m61-b1.edf',
                    r'2DTUSERVER-Maria\EDF\m63-b1.edf',
                    r'2DTUSERVER-Maria\EDF\m63-b2.edf',
                    r'2DTUSERVER-Maria\EDF\m86-b1.edf',
                    r'2DTUSERVER-Maria\EDF\m88-b1.edf',
                    r'2DTUSERVER-Maria\EDF\m88-b2.edf',
                    r'2DTUSERVER-Maria\EDF\m96-b1.edf',
                    r'2DTUSERVER-Maria\EDF\m96-b2.edf',
                    r'2DTUSERVER-Marieke\EDF\m2-b1.edf',
                    r'2DTUSERVER-Marieke\EDF\m21-b1.edf']

training_labels = [os.path.join(data_path, p) for p in training_labels]
training_signals = [os.path.join(data_path, p) for p in training_signals]


for i in range(len(training_signals)):
#     i=29
    train_dataset = load_to_dataset(signal_path=training_signals[i],
                                   labels_path=training_labels[i],
                                   resample_rate=128,
                                   just_artifact_labels=True,
                                   just_stage_labels=False,
                                   validation_split=0)

    batch_size = 100
    train_dataset = train_dataset.batch(batch_size)

    for idx, batch in enumerate(train_dataset):
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

fpr, tpr, thresholds = metrics.roc_curve(y_true_all, cnn_probs_all)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr)
display.plot()
opt_thr_idx = optimal_threshold_idx(fpr, tpr)
opt_thr = thresholds[opt_thr_idx]
plt.plot(fpr[opt_thr_idx], tpr[opt_thr_idx], 'r.')
plt.title('Optimal threshold = %1.f' % opt_thr)

np.save('fpr_kornum_training_set.npy', fpr)
np.save('tpr_kornum_training_set.npy', tpr)
np.save('optimal_thr_kornum.npy', opt_thr)

plt.figure(figsize=(6.4*1.5, 4.8*1.5))
plt.plot(fpr, tpr, linewidth=3.5, color="royalblue")
plt.title('Minimum distance threshold = %1.2f' % opt_thr, fontsize=26)
plt.plot(fpr[opt_thr_idx], tpr[opt_thr_idx], '.', markersize=18, color="tomato")
plt.xlabel('True Positive Rate', fontsize=26)
plt.ylabel('False Positive Rate', fontsize=26)
plt.tick_params(axis='both', which='major', labelsize=26)
