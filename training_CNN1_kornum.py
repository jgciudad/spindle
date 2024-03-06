# import mne
# import scipy
# import scipy.signal
# import numpy as np
# import pandas as pd
import tensorflow as tf
import sklearn.metrics
# import random
from tensorflow.keras.layers import Input, MaxPool2D, Conv2D, Dense, Flatten, Dropout
from kornum_data.kornum_data_loading import SequenceDataset
from metrics import *
# from tools import *
import pickle

# plt.ion()

save_path = '/home/s202283/outputs/spindle'
model_name = 'A_3'

data_path = '/scratch/s202283/data/spindle_data/numpy'
csv_path = "/scratch/s202283/data/spindle_data/labels_all.csv"

print("Devices available: ", tf.config.list_physical_devices())

# -------------------------------------------------------------------------------------------------------------------------

train_sequence = SequenceDataset(data_path,
                                 csv_path,
                                 'train',
                                 100,
                                 True,
                                 False)

val_sequence = SequenceDataset(data_path,
                               csv_path,
                               'validation',
                               100,
                               True,
                               False)

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
    Dense(units=3, activation='softmax', kernel_initializer='glorot_uniform')
])

checkpoint_path = os.path.join(save_path, model_name)
if not os.path.exists(os.path.dirname(checkpoint_path)):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
checkpoint_callback = MyCustomCallback(validation_dataset=val_sequence,
                                       save_checkpoint_path=checkpoint_path,
                                       evaluation_rate=int(len(train_sequence)/10),
                                       improvement_threshold=0.001,
                                       early_stopping_thr=10)

spindle_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=5 * 1e-5,
                                                                beta_1=0.9,
                                                                beta_2=0.999),
                      loss=MulticlassWeightedCrossEntropy_2(n_classes=3),
                    #   loss = tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.CategoricalAccuracy(),
                               MulticlassF1Score(n_classes=3),
                               MulticlassBalancedAccuracy(n_classes=3)])
                    #   run_eagerly=True)

history1 = spindle_model.fit(
    x=train_sequence,
    epochs=10,
    verbose=1,
    callbacks=[checkpoint_callback])

print('End of training reached')

with open(os.path.join(save_path, model_name, 'training_history.pkl'), 'wb') as f:
    pickle.dump(history1.history, f)

print('History saving reached')

# plot_history_cnn1(history1.history, model_name, save_path, epochs=5)

# spindle_model.save_weights(os.path.join(save_path, model_name, model_name + "_5epochs" + ".h5"))

# history2 = spindle_model.fit(x=train_dataset,
#                             validation_data=val_dataset,
#                             epochs=5,
#                             verbose=1)
#
# plot_history_cnn1(history2, model_name, save_path, epochs=5)
#
# spindle_model.save_weights(os.path.join(save_path, model_name, model_name + "_10epochs" + ".h5"))