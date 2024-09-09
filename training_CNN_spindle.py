import sys
sys.path.append("..")
import tensorflow as tf
import sklearn.metrics
# import random
from tensorflow.keras.layers import Input, MaxPool2D, Conv2D, Dense, Flatten, Dropout
from spindle_data_loading_v2 import SequenceDataset2 
from labdata.metrics import *
# from tools import *
import pickle
import os
from base.config_loader import ConfigLoader
import argparse
import time
import keras as keras 
import numpy as np
import wandb

def customloss(y_true, y_pred,y_lab,n_classes,model):
 
    y_true = tf.one_hot(y_true, depth=n_classes)
    
    if model=="spindle_ss": 
        ce = tf.keras.metrics.categorical_crossentropy(y_true, y_pred)
        for i in range(n_classes):
            num_class_i = tf.shape(tf.where(y_true[:, i] == 1))[0]

            if num_class_i > 0:
                w = tf.shape(y_true)[0] / (n_classes * num_class_i)
                w = tf.cast(w, dtype=tf.float32)
                
                ce = tf.tensor_scatter_nd_update(ce, tf.where(y_true[:, i] == 1), w * ce[y_true[:, i] == 1])
    return tf.reduce_mean(ce)

def parse():
    """define and parse arguments for the script"""
    parser = argparse.ArgumentParser(description='training script')
    parser.add_argument('--experiment', '-e', required=True,
                        help='name of experiment to run')
    parser.add_argument('--test_lab', '-t', required=True,
                        help="lab to test the model on: 'Antoine', 'Kornum', 'Alessandro', 'Sebastian' or 'Maiken'")
    parser.add_argument('--save_dir', '-s', required=True,
                        help="path to save model'")

    return parser.parse_args()

args       = parse()
config     = ConfigLoader(save_dir=os.path.join(args.save_dir, args.test_lab), experiment=args.experiment)  # load config from experiment
save_path  = args.save_dir
model_name = config.MODEL_NAME 
data_path  = config.DATA_DIR + config.DATA_FILE
# ------------------------------------------------------ WEIGHTS AND BIASES -------------------------------------------------------------------

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="SPINDLE-ON-SPINDLE",

    # track hyperparameters and run metadata
    config={
    "learning_rate": config.LEARNING_RATE,
    "architecture": "SPINDLE",
    "dataset": "SPINDLE",
    "epochs": config.EPOCHS,
    }
)

# ------------------------------------------------------ MODEL SPECIFICATIONS -------------------------------------------------------------------
dl_train = SequenceDataset2(data_folder=config.DATA_FILE,
                                  set='train',
                                  config=config,
                                  test_lab = args.test_lab)

NCLASSES_MODEL_SS  = 3
metrics_list_SS    = [tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
                     MulticlassF1Score(n_classes=NCLASSES_MODEL_SS,name='f1_score'),
                     MulticlassBalancedAccuracy(n_classes=NCLASSES_MODEL_SS,name='balance_accuracy')]
last_activation_SS = 'softmax' 

    
if config.LOSS_TYPE == 'weighted_ce':
    loss_fn_SS = customloss
    loss_fn_AA = customloss
elif config.LOSS_TYPE == 'normal_ce':
    loss_fn_SS = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    loss_fn_AA = BinaryWeightedCrossEntropy()


print("Devices available: ", tf.config.list_physical_devices())

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
    Dense(units=NCLASSES_MODEL_SS, activation=last_activation_SS, kernel_initializer='glorot_uniform')
])


checkpoint_path = os.path.join(save_path, model_name)
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path, exist_ok=True)

# Initialize optimizer
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=config.LEARNING_RATE, beta_1=0.9, beta_2=0.999)

spindle_model.compile(optimizer=optimizer, loss=loss_fn_SS, metrics=metrics_list_SS)
train_acc_metric = tf.keras.metrics.CategoricalAccuracy() 
val_acc_metric = tf.keras.metrics.CategoricalAccuracy() 

# ------------------------------------------------------ TRAINING LOOP -------------------------------------------------------------------

epochs                 = config.EPOCHS
best_val_f1            = -np.inf
early_stopping_counter = 0
patience               = 10
f1_metric              = MulticlassF1Score(n_classes=NCLASSES_MODEL_SS, name='f1_score')
acc_all                = []
categorical_accuracy = tf.keras.metrics.CategoricalAccuracy()

for epoch in range(epochs):
    print(f"\nStart of epoch {epoch}")
    dl_train.on_epoch_end()

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train, y_batch_labs) in enumerate(dl_train.train_dataloader):
        with tf.GradientTape() as tape:

            # SPINDLE MODEL 
            logits_SS  = spindle_model(x_batch_train, training=True)  
            mask       = y_batch_train != 3    
            loss_value = loss_fn_SS(y_batch_train[mask], logits_SS[mask],y_batch_labs[mask],3,"spindle_ss")  # Compute the loss value for this minibatch
            grads      = tape.gradient(loss_value, spindle_model.trainable_weights)
            optimizer.apply_gradients(zip(grads, spindle_model.trainable_weights))
            train_acc_metric.update_state(y_batch_train[mask], logits_SS[mask])

            for metric in metrics_list_SS:
                metric.update_state(tf.one_hot(y_batch_train[mask], depth=3), tf.one_hot(np.argmax(logits_SS[mask],axis=1),depth=3))
            
            if step % 20 == 0: 
                metrics_log = {metric.name: metric.result().numpy() for metric in metrics_list_SS}
                metrics_log["loss"] = loss_value.numpy()
                wandb.log(metrics_log)
    
    # Evaluate on the validation dataset at the end of each epoch
    f1_metric.reset_states()
    categorical_accuracy.reset_states()

    for x_batch_val, y_batch_val, y_batch_labs in dl_train.val_dataloader:
        val_logits = spindle_model(x_batch_val, training=False)
        mask = y_batch_val != 3    
        f1_metric.update_state(tf.one_hot(y_batch_val[mask], depth=NCLASSES_MODEL_SS), tf.one_hot(np.argmax(val_logits[mask],axis=1),depth=3))
        categorical_accuracy.update_state(tf.one_hot(y_batch_val[mask], depth=3), tf.one_hot(np.argmax(val_logits[mask],axis=1),depth=3))

    val_f1 = f1_metric.result().numpy()
    val_acc = categorical_accuracy.result().numpy()
    acc_all.append(val_acc)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        early_stopping_counter = 0
        # Save the best model checkpoint
        spindle_model.save_weights(save_path+"epoch"+str(epoch)+".h5")
    else:
        early_stopping_counter += 1
        print(f"Early stopping counter: {early_stopping_counter} out of {patience}")

    if early_stopping_counter >= patience:
        print("Early stopping triggered")

    # Log metrics at the end of the epoch
    metrics_log = {metric.name: metric.result().numpy() for metric in metrics_list_SS}
    metrics_log["epoch"] = epoch
    metrics_log["val_f1"] = val_f1
    wandb.log(metrics_log)
    print(f"Epoch {epoch} metrics: {metrics_log}")

np.save(save_path+'array.npy', np.array(acc_all))


# epochs       = config.EPOCHS
# best_val_f1 = -np.inf
# early_stopping_counter = 0
# patience = 2
# f1_metric = MulticlassF1Score(n_classes=NCLASSES_MODEL_SS, name='f1_score')

# for epoch in range(epochs):
#     print(f"\nStart of epoch {epoch}")

#     # Reset metrics at the start of each epoch
#     for metric in metrics_list_SS:
#         metric.reset_states()

#     # Iterate over the batches of the dataset.
#     for step, (x_batch_train, y_batch_train, y_batch_labs) in enumerate(dl_train.train_dataloader):
#         with tf.GradientTape() as tape:

#             # SPINDLE MODEL 
#             logits_SS  = spindle_model(x_batch_train, training=True)  
#             mask = y_batch_train != 3    
#             loss_value = loss_fn_SS(y_batch_train[mask], logits_SS[mask],y_batch_labs[mask],3,"spindle_ss")  # Compute the loss value for this minibatch
#             grads = tape.gradient(loss_value, spindle_model.trainable_weights)
#             optimizer.apply_gradients(zip(grads, spindle_model.trainable_weights))
#             train_acc_metric.update_state(y_batch_train[mask], logits_SS[mask])

#             for metric in metrics_list_SS:
#                 metric.update_state(tf.one_hot(y_batch_train[mask], depth=3), tf.one_hot(np.argmax(logits_SS[mask],axis=1),depth=3))
            
#             if step % 20 == 0: 
#                 metrics_log = {metric.name: metric.result().numpy() for metric in metrics_list_SS}
#                 metrics_log["loss"] = loss_value.numpy()
#                 wandb.log(metrics_log)
    
#     # Evaluate on the validation dataset at the end of each epoch
#     f1_metric.reset_states()

#     for x_batch_val, y_batch_val, y_batch_labs in dl_train.val_dataloader:
#         val_logits = spindle_model(x_batch_val, training=False)
#         mask = y_batch_val != 3    
#         f1_metric.update_state(tf.one_hot(y_batch_val[mask], depth=NCLASSES_MODEL_SS), tf.one_hot(np.argmax(val_logits[mask],axis=1),depth=3))

#     val_f1 = f1_metric.result().numpy()

#     if val_f1 > best_val_f1:
#         best_val_f1 = val_f1
#         early_stopping_counter = 0
#         # Save the best model checkpoint
#         spindle_model.save_weights(checkpoint_path+"f1"+str(val_f1)+".h5")
#     else:
#         early_stopping_counter += 1
#         print(f"Early stopping counter: {early_stopping_counter} out of {patience}")

#     if early_stopping_counter >= patience:
#         print("Early stopping triggered")
#         break

#     # Log metrics at the end of the epoch
#     metrics_log = {metric.name: metric.result().numpy() for metric in metrics_list_SS}
#     metrics_log["epoch"] = epoch
#     metrics_log["val_f1"] = val_f1
#     wandb.log(metrics_log)
#     print(f"Epoch {epoch} metrics: {metrics_log}")

# -------------------------------------------------------------------------------------------------------------------------


# if config.ARTEFACT_DETECTION==False:
#     JUST_ARTIFACT_LABELS = False 
#     last_activation = 'softmax'
#     if config.Just_not_art_epochs==False:
#         NCLASSES_MODEL = 4
#         raise Exception('Testing for JUST_NOT_ART_EPOCHS==False not implemented. compute_and_save_metrics_cnn1() needs to be adapted.')
#     else:
#         NCLASSES_MODEL = 3

#     metrics_list = [tf.keras.metrics.CategoricalAccuracy(),
#                    MulticlassF1Score(n_classes=NCLASSES_MODEL),
#                    MulticlassBalancedAccuracy(n_classes=NCLASSES_MODEL)]
    
#     if config.LOSS_TYPE=='weighted_ce':
#         loss=MulticlassWeightedCrossEntropy_2(n_classes=NCLASSES_MODEL)
#     elif config.LOSS_TYPEE=='normal_ce':
#         loss=tf.keras.losses.CategoricalCrossentropy()

# else:
#     if config.Just_not_art_epochs==True: raise Exception('If ARTIFACT_DETECTION=True, JUST_NOT_ART_EPOCHS must be False')
#     JUST_ARTIFACT_LABELS = True
#     last_activation = 'sigmoid' 
#     NCLASSES_MODEL = 1

#     metrics_list=[tf.keras.metrics.BinaryAccuracy(),
#                 BinaryBalancedAccuracy(),
#                 BinaryF1Score()]
    
#     if config.LOSS_TYPE=='weighted_ce':
#         loss=BinaryWeightedCrossEntropy()
#     elif config.LOSS_TYPE=='normal_ce':
#         raise Exception("Not implemented")
# print("Devices available: ", tf.config.list_physical_devices())


# spindle_model = tf.keras.Sequential([
#     Input((160, 48, 3)),
#     MaxPool2D(pool_size=(2, 3), strides=(2, 3)),
#     Conv2D(filters=50, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
#     MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
#     Flatten(),
#     Dense(units=1000, activation='relu', kernel_initializer='glorot_uniform'),
#     Dropout(0.5),
#     Dense(units=1000, activation='relu', kernel_initializer='glorot_uniform'),
#     Dropout(0.5),
#     Dense(units=NCLASSES_MODEL, activation=last_activation, kernel_initializer='glorot_uniform')
# ])

# checkpoint_path = os.path.join(save_path, model_name)
# if not os.path.exists(checkpoint_path):
#     os.makedirs(checkpoint_path, exist_ok=True)

# # weights and bias 

# checkpoint_callback = MyCustomCallback(validation_dataset=val_sequence,
#                                        save_checkpoint_path=checkpoint_path,
#                                        evaluation_rate=int(len(train_sequence)/10),
#                                        improvement_threshold=0.001,
#                                        early_stopping_thr=10,
#                                        artifact_detection=config.ARTEFACT_DETECTION)

# spindle_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=config.LEARNING_RATE,
#                                                                 beta_1=0.9,
#                                                                 beta_2=0.999),
#                       loss=loss,
#                       metrics=metrics_list)
#                     #   run_eagerly=True)

# history = spindle_model.fit(
#     x=train_sequence,
#     epochs=config.EPOCHS,
#     verbose=1,
#     callbacks=[checkpoint_callback])

# print('End of training reached')

# with open(os.path.join(save_path, model_name, 'training_history.pkl'), 'wb') as f:
#     pickle.dump(history.history, f)

# print('History saving reached')