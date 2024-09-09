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
def customloss(y_true, y_pred,y_lab,n_classes):

    y_true = tf.one_hot(y_true, depth=n_classes)
    ce = tf.keras.metrics.categorical_crossentropy(y_true, y_pred)
    ce = ce*y_lab

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
print("done loading")

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

epochs       = config.EPOCHS
best_val_f1 = -np.inf
early_stopping_counter = 0
patience  = 6
f1_metric = MulticlassF1Score(n_classes=NCLASSES_MODEL_SS, name='f1_score')
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
            assert len(np.unique(y_batch_train))<=3
            mask = y_batch_train != 3    
            loss_value = loss_fn_SS(y_batch_train[mask], logits_SS[mask],y_batch_labs[mask],3)  # Compute the loss value for this minibatch
            grads = tape.gradient(loss_value, spindle_model.trainable_weights)
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
    spindle_model.save_weights(save_path+"epoch"+str(epoch)+".h5")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        early_stopping_counter = 0
        # Save the best model checkpoint
        spindle_model.save_weights(save_path+"epoch"+str(epoch)+".h5")

    # Log metrics at the end of the epoch
    metrics_log = {metric.name: metric.result().numpy() for metric in metrics_list_SS}
    metrics_log["epoch"]  = epoch
    metrics_log["val_f1"] = val_f1
    wandb.log(metrics_log)
    print(f"Epoch {epoch} metrics: {metrics_log}")

np.save(save_path+'array.npy', np.array(acc_all))
