import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, f1_score, precision_score, accuracy_score, cohen_kappa_score


def plot_history_cnn1(history, model_name, save_path, epochs=''):
    # The histories must be the dictionary inside the Keras History objects (e.g. History.history)

    fig, ax = plt.subplots(3, 1, figsize=(5, 12))

    ax[0].plot(history['categorical_accuracy'])
    ax[0].plot(history['val_categorical_accuracy'])
    ax[0].plot(history['multiclass_balanced_accuracy'])
    ax[0].plot(history['val_multiclass_balanced_accuracy'])
    ax[0].legend(['Training standard', 'Validation standard', 'Training balanced', 'Validation balanced'])
    ax[0].set_title('Accuracy')

    # ax[1].plot(history.history['multiclass_balanced_accuracy'])
    # ax[1].plot(history.history['val_multiclass_balanced_accuracy'])
    # ax[1].legend(['Training', 'Validation'])
    # ax[1].set_title('Balanced categorical accuracy')

    ax[1].plot(history['loss'])
    ax[1].plot(history['val_loss'])
    ax[1].legend(['Training', 'Validation'])
    ax[1].set_title('Weighted categorical cross entropy')

    ax[2].plot(history['multiclass_F1_score'])
    ax[2].plot(history['val_multiclass_F1_score'])
    ax[2].legend(['Training', 'Validation'])
    ax[2].set_title('Average F1 score')

    fig.suptitle(model_name)

    save_path = os.path.join(save_path, model_name, 'training_curve_' + str(epochs) + '_epochs.jpg')
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path)


def plot_history_cnn2(history, model_name, save_path, epochs=''):
    # The histories must be the dictionary inside the Keras History objects (e.g. History.history)

    fig, ax = plt.subplots(3, 1, figsize=(5, 12))

    ax[0].plot(history['binary_accuracy'])
    ax[0].plot(history['val_binary_accuracy'])
    ax[0].plot(history['binary_balanced_accuracy'])
    ax[0].plot(history['val_binary_balanced_accuracy'])
    ax[0].legend(['Training standard', 'Validation standard', 'Training balanced', 'Validation balanced'])
    ax[0].set_title('Accuracy')

    # ax[1].plot(history.history['multiclass_balanced_accuracy'])
    # ax[1].plot(history.history['val_multiclass_balanced_accuracy'])
    # ax[1].legend(['Training', 'Validation'])
    # ax[1].set_title('Balanced categorical accuracy')

    ax[1].plot(history['loss'])
    ax[1].plot(history['val_loss'])
    ax[1].legend(['Training', 'Validation'])
    ax[1].set_title('Weighted categorical cross entropy')

    ax[2].plot(history['F1_score'])
    ax[2].plot(history['val_F1_score'])
    ax[2].legend(['Training', 'Validation'])
    ax[2].set_title('F1 score')

    fig.suptitle(model_name)

    save_path = os.path.join(save_path, model_name, 'training_curve_' + str(epochs) + '_epochs.jpg')
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path)


def concatenate_histories(history1, history2):
    # The histories must be the dictionary inside the Keras History objects (e.g. History.history)

    history = {}
    for k in history1.keys():
        history[k] = history1[k] + history2[k]

    return history


def matrix_to_excel(cm, save_path, filename):
    ## convert your array into a dataframe
    df = pd.DataFrame(cm)

    df.to_excel(os.path.join(save_path, filename), index=False)


def compute_and_save_metrics_cnn1(y_true, y_pred, save_path, model_name):
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NREM', 'REM', 'WAKE'])
    cm_plot = disp.plot(cmap=plt.cm.Blues)
    plt.savefig(os.path.join(save_path, 'cm_' + model_name + '.png'))

    matrix_to_excel(cm, save_path, 'confusion_matrix.xlsx')

    # ppv = [cm[0,0]/np.sum(cm[:,0]), cm[1,1]/np.sum(cm[:,1]), cm[2,2]/np.sum(cm[:,2]) ] # positive predictive value # it works properly but I prefer to use the sklearn implementation
    kappa = cohen_kappa_score(y1=y_true, y2=y_pred, labels=np.arange(1, 4))
    kappa = np.array([kappa, 2])
    matrix_to_excel(kappa, save_path, 'kappa.xlsx')

    ppv = precision_score(y_true, y_pred, average=None)  # positive predictive value
    npv = [np.sum(cm[1:, 1:]) / np.sum(cm[:, 1:]),
           (np.sum(cm[:, 0]) + np.sum(cm[:, 2]) - cm[1, 0] - cm[1, 2]) / (np.sum(cm[:, 0]) + np.sum(cm[:, 2])),
           np.sum(cm[:2, :2]) / np.sum(cm[:, :2])]  # negative predictive value
    # tpr = [cm[0,0]/np.sum(cm[0, :]), cm[1,1]/np.sum(cm[1, :]), cm[2,2]/np.sum(cm[2, :]) ] # true positive rate, recall, sensitivity
    tpr = recall_score(y_true, y_pred,
                       average=None)  # true positive rate # it works properly but I prefer to use the sklearn implementation
    tnr = [np.sum(cm[1:, 1:]) / np.sum(cm[1:, :]),
           (np.sum(cm[0, :]) + np.sum(cm[2, :]) - cm[0, 1] - cm[2, 1]) / (np.sum(cm[0, :]) + np.sum(cm[2, :])),
           np.sum(cm[:2, :2]) / np.sum(cm[:2, :])]  # true negative rate
    f1 = f1_score(y_true, y_pred, average=None)
    accuracy = accuracy_score(y_true, y_pred)

    metrics = pd.DataFrame(np.vstack([ppv, npv, tpr, tnr, f1]), columns=['NREM', 'REM', 'WAKE'])
    metrics.to_csv(os.path.join(save_path, 'metrics_' + model_name + '.csv'), index=False)

    fig, ax = plt.subplots(2, 1)
    ax[0].axis('off')
    ax[0].table(cellText=metrics.round(3).values, colLabels=metrics.keys(),
                rowLabels=['PPV', 'NPV', 'TPR', 'TNR', 'f1'],
                loc='center')
    ax[1].axis('off')
    ax[1].table(cellText=np.round(np.array(accuracy).reshape((1, 1)), 3), colLabels=['Accuracy'], loc='center')
    plt.show()
    plt.savefig(os.path.join(save_path, 'metrics_' + model_name + '.png'))


def compute_and_save_metrics_cnn2(y_true, y_pred, threshold, save_path, model_name):

    y_pred = y_pred > threshold

    if threshold==0.5:
        str_thr = 'normal'
    else:
        str_thr = 'opt'

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['not art.', 'art.'])
    cm_plot = disp.plot(cmap=plt.cm.Blues)
    plt.savefig(os.path.join(save_path, 'cm_' + str_thr + '_' + model_name + '.png'))

    ppv = precision_score(y_true, y_pred)  # positive predictive value
    npv = cm[0, 0] / np.sum(cm[:, 0])  # negative predictive value
    tpr = recall_score(y_true, y_pred)
    tnr = cm[0, 0] / np.sum(cm[0, :])  # true negative rate
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    agreement_score = cm[1, 1] / (cm[1, 1] + cm[1, 0] + cm[0, 1])

    metrics = pd.DataFrame(np.vstack([accuracy, ppv, npv, tpr, tnr, f1, agreement_score]), columns=['art.'])
    metrics.to_csv(os.path.join(save_path, 'metrics_' + str_thr + '_' + model_name + '.csv'), index=False)

    fig, ax = plt.subplots(1, 1)
    ax.axis('off')
    ax.table(cellText=metrics.round(3).values, colLabels=metrics.keys(),
                rowLabels=['Acc.', 'PPV', 'NPV', 'TPR', 'TNR', 'f1', 'agr_score'],
                loc='center')
    plt.show()
    plt.savefig(os.path.join(save_path, 'metrics_' + str_thr + '_' + model_name + '.png'))


def compute_and_save_metrics_whole_model(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NREM', 'REM', 'WAKE', 'Art'])
    cm_plot = disp.plot(cmap=plt.cm.Blues)
    plt.savefig(os.path.join(save_path, 'cm_whole_model' + '.png'))

    # ppv = [cm[0,0]/np.sum(cm[:,0]), cm[1,1]/np.sum(cm[:,1]), cm[2,2]/np.sum(cm[:,2]) ] # positive predictive value # it works properly but I prefer to use the sklearn implementation
    ppv = precision_score(y_true, y_pred, average=None)  # positive predictive value
    npv = [np.sum(cm[1:, 1:]) / np.sum(cm[:, 1:]),
           (np.sum(cm) - np.sum(cm[1, :]) - np.sum(cm[:, 1]) + cm[1, 1]) / (np.sum(cm) - np.sum(cm[:, 1])),
           (np.sum(cm) - np.sum(cm[2, :]) - np.sum(cm[:, 2]) + cm[2, 2]) / (np.sum(cm) - np.sum(cm[:, 2])),
           np.sum(cm[:2, :2]) / np.sum(cm[:, :2])]  # negative predictive value
    # tpr = [cm[0,0]/np.sum(cm[0, :]), cm[1,1]/np.sum(cm[1, :]), cm[2,2]/np.sum(cm[2, :]) ] # true positive rate, recall, sensitivity
    tpr = recall_score(y_true, y_pred,
                       average=None)  # true positive rate # it works properly but I prefer to use the sklearn implementation
    tnr = [np.sum(cm[1:, 1:]) / np.sum(cm[1:, :]),
           (np.sum(cm) - np.sum(cm[1, :]) - np.sum(cm[:, 1]) + cm[1, 1]) / (np.sum(cm) - np.sum(cm[1, :])),
           (np.sum(cm) - np.sum(cm[2, :]) - np.sum(cm[:, 2]) + cm[2, 2]) / (np.sum(cm) - np.sum(cm[2, :])),
           np.sum(cm[:2, :2]) / np.sum(cm[:2, :])]  # true negative rate
    f1 = f1_score(y_true, y_pred, average=None)
    accuracy = accuracy_score(y_true, y_pred)

    metrics = pd.DataFrame(np.vstack([ppv, npv, tpr, tnr, f1]), columns=['NREM', 'REM', 'WAKE', 'Art'])
    metrics.to_csv(os.path.join(save_path, 'metrics_whole_model' + '.csv'), index=False)

    fig, ax = plt.subplots(2, 1)
    ax[0].axis('off')
    ax[0].table(cellText=metrics.round(3).values, colLabels=metrics.keys(),
                rowLabels=['PPV', 'NPV', 'TPR', 'TNR', 'f1'],
                loc='center')
    ax[1].axis('off')
    ax[1].table(cellText=np.round(np.array(accuracy).reshape((1, 1)), 3), colLabels=['Accuracy'], loc='center')
    plt.show()
    plt.savefig(os.path.join(save_path, 'metrics_whole_model' + '.png'))


def optimal_threshold_idx(fpr, tpr):
    distances = np.sqrt(np.square(fpr) + np.square(1 - tpr))

    min_distance_idx = np.argmin(distances)

    return min_distance_idx
