import tensorflow as tf
import numpy as np
# from spindle_data.spindle_data_loading import load_labels
# from kornum_data.kornum_data_loading import load_labels
from matplotlib import pyplot as plt
import os


def hmm_prediction(cnn_output_probs, transition_matrix, prior_probs):
    # Transition matrix rows are S_(t-1), columns are S_t
    cnn_prediction = tf.argmax(cnn_output_probs, axis=1)

    posterior_probs = np.zeros((cnn_output_probs.shape[0], 3))
    for i in range(cnn_output_probs.shape[0]):
        # for s in range(3):
        if i == 0:
            posterior_probs[i, :] = cnn_output_probs[i, :] * initial_probs
        else:
            posterior_probs[i, :] = cnn_output_probs[i, :] * transition_matrix[cnn_prediction[i - 1], :] / prior_probs

    return np.argmax(posterior_probs, axis=1)


def get_transition_matrix(labels, cancel_forbidden_transitions):
    """
    :param labels: list of .csv labels (the original SPINDLE label files)
    :return: 3x3 transition matrix, where rows are S(t-1) and columns S(t), and the order of the classes is NREM, REM, W
    """

    def calculate_transition_matrix(lab):
        transition_matrix = np.zeros((3, 3))

        lab = lab.to_numpy()  # transform to numpy

        for i in range(1, len(lab)):
            class_last = np.argmax(lab[i - 1])
            class_current = np.argmax(lab[i])

            transition_matrix[class_last, class_current] += 1

        return transition_matrix

    transition_matrix = np.zeros((3, 3))
    for l in labels:
        labels1 = load_labels(l,
                              scorer=1,
                              just_artifact_labels=False,
                              artifact_to_stages=True)
        labels2 = load_labels(l,
                              scorer=2,
                              just_artifact_labels=False,
                              artifact_to_stages=True)
        transition_matrix = transition_matrix + calculate_transition_matrix(labels1)
        transition_matrix = transition_matrix + calculate_transition_matrix(labels2)

    transition_matrix = transition_matrix / np.sum(transition_matrix, axis=1, keepdims=True)
    if cancel_forbidden_transitions:
        transition_matrix[1, 0] = 0
        transition_matrix[2, 1] = 0

    return transition_matrix


def get_transition_matrix_kornum(labels, cancel_forbidden_transitions):
    """
    :param labels: list of .tsv labels (the original kornum label files)
    :return: 3x3 transition matrix, where rows are S(t-1) and columns S(t), and the order of the classes is NREM, REM, W
    """

    def calculate_transition_matrix(lab):
        transition_matrix = np.zeros((3, 3))

        lab = lab.to_numpy()  # transform to numpy

        for i in range(1, len(lab)):
            class_last = np.argmax(lab[i - 1])
            class_current = np.argmax(lab[i])

            if class_last != 3 and class_current != 3:
                transition_matrix[class_last, class_current] += 1
            else:
                a=9

        return transition_matrix

    transition_matrix = np.zeros((3, 3))
    for l in labels:
        labels_i = load_labels(l,
                               just_artifact_labels=False,
                               just_stage_labels=False)
        transition_matrix = transition_matrix + calculate_transition_matrix(labels_i)

    transition_matrix = transition_matrix / np.sum(transition_matrix, axis=1, keepdims=True)
    if cancel_forbidden_transitions:
        transition_matrix[1, 0] = 0
        transition_matrix[2, 1] = 0

    return transition_matrix


def get_priors(labels):
    def calculate_priors(lab):
        priors = np.zeros(3)

        lab = lab.to_numpy()  # transform to numpy

        for i in range(len(lab)):
            stage = np.argmax(lab[i])

            priors[stage] += 1

        return priors

    priors = np.zeros(3)
    for l in labels:
        labels1 = load_labels(l,
                              scorer=1,
                              just_artifact_labels=False,
                              artifact_to_stages=True)
        labels2 = load_labels(l,
                              scorer=2,
                              just_artifact_labels=False,
                              artifact_to_stages=True)
        priors = priors + calculate_priors(labels1)
        priors = priors + calculate_priors(labels2)

    priors = priors / np.sum(priors, keepdims=True)

    return priors


def get_priors_kornum(labels):
    def calculate_priors(lab):
        priors = np.zeros(3)

        lab = lab.to_numpy()  # transform to numpy

        for i in range(len(lab)):
            stage = np.argmax(lab[i])

            if stage!=3:
                priors[stage] += 1

        return priors

    priors = np.zeros(3)
    for l in labels:
        labels_i = load_labels(l,
                               just_artifact_labels=False,
                               just_stage_labels=False)
        priors = priors + calculate_priors(labels_i)

    priors = priors / np.sum(priors, keepdims=True)

    return priors


def viterbi(y, A, Pi):
    """
    From https://stackoverflow.com/questions/9729968/python-implementation-of-viterbi-algorithm
    (Inspired on wikipedia)

    Returns the MAP estimate of state trajectory of Hidden Markov Model.

    Parameters
    ----------
    y : array (T,)
        Observation state sequence. int dtype.
    A : array (K, K)
        State transition matrix. See HiddenMarkovModel.state_transition  for
        details.
        Must be:

         S(t)->  N |  R |  W |
        ------|----|----|----|
          N   |    |    |    |
        ------|----|----|----|
          R   |    |    |    |
        ------|----|----|----|
          W   |    |    |    |
          ^   |----------------
        S(t-1)|


    B : array (K, M)
        Emission matrix. See HiddenMarkovModel.emission for details.
    Pi: optional, (K,)
        Initial state probabilities: Pi[i] is the probability x[0] == i. If
        None, uniform initial distribution is assumed (Pi[:] == 1/K).

    Returns
    -------
    x : array (T,)
        Maximum a posteriori probability estimate of hidden state trajectory,
        conditioned on observation sequence y under the model parameters A, B,
        Pi.
    T1: array (K, T)
        the probability of the most likely path so far
    T2: array (K, T)
        the x_j-1 of the most likely path so far
    """
    # Cardinality of the state space
    K = A.shape[0]

    T = len(y)
    T1 = np.zeros((K, T))  # , 'd')
    T2 = np.zeros((K, T))  # , 'B')

    # Initialize the tracking tables from first observation
    T1[:, 0] = np.log(Pi) + np.log(y[0, :])
    T2[:, 0] = 0

    # Iterate throught the observations updating the tracking tables
    for j in range(1, T):
        for i in range(K):

            if y[j, i] == 0:
                print('WARNING: CNN_OUTPUT=0 DETECTED AT j=', j)

            T1[i, j] = np.max(T1[:, j - 1] + np.log(A[:, i]) + np.log(y[j, i]))
            T2[i, j] = np.argmax(T1[:, j - 1] + np.log(A[:, i]) + np.log(y[j, i]))

        # T1[:, j] = np.max(T1[:, j - 1] * A * tf.transpose(y[j,:]), 0)
        # T2[:, j] = np.argmax(T1[:, j - 1] * A.T, 0)

    # Build the output, optimal model trajectory
    # x = np.empty(T, 'B')
    x = np.zeros(T, dtype=int)
    x[-1] = np.argmax(T1[:, T - 1])
    for i in reversed(range(1, T)):
        x[i - 1] = T2[x[i], i]

    return x, T1, T2


def evaluate_hmm_effect(y_true, y_cnn, y_hmm):
    def calculate_transition_matrix(labels):
        transition_matrix = np.zeros((3, 3))

        for i in range(1, len(labels)):
            class_last = labels[i - 1]
            class_current = labels[i]

            transition_matrix[class_last, class_current] += 1

        return transition_matrix

    corrected_epochs_cnn = y_cnn[y_cnn != y_hmm]
    corrected_epochs_hmm = y_hmm[y_cnn != y_hmm]
    corrected_epochs_true = y_true[y_cnn != y_hmm]

    n_corrected_epochs = np.zeros((1, 4))
    correction_matrix = np.zeros((3, 3, 3))
    for s in range(3):
        corrected_epochs_cnn_s = corrected_epochs_cnn[corrected_epochs_true == s]
        corrected_epochs_hmm_s = corrected_epochs_hmm[corrected_epochs_true == s]

        n_corrected_epochs[0, s] = len(corrected_epochs_cnn_s)

        for i in range(len(corrected_epochs_cnn_s)):
            correction_matrix[s, corrected_epochs_cnn_s[i], corrected_epochs_hmm_s[i]] += 1

    n_corrected_epochs[0, 3] = np.sum(n_corrected_epochs[0, :3])

    cnn_transitions = calculate_transition_matrix(y_cnn)
    hmm_transitions = calculate_transition_matrix(y_hmm)

    return n_corrected_epochs, correction_matrix, cnn_transitions, hmm_transitions


def evaluate_hmm_withArts_effect(y_true, y_true_art, y_cnn, y_hmm):
    def calculate_transition_matrix(labels):
        transition_matrix = np.zeros((3, 3))

        for i in range(1, len(labels)):
            if y_true_art[i - 1] != 1 and y_true_art[i] != 1:
                class_last = labels[i - 1]
                class_current = labels[i]

                transition_matrix[class_last, class_current] += 1

        return transition_matrix

    y_cnn_filtered = y_cnn[y_true_art == 0]
    y_hmm_filtered = y_hmm[y_true_art == 0]
    y_true_filtered = y_true[y_true_art == 0]

    corrected_epochs_cnn = y_cnn_filtered[y_cnn_filtered != y_hmm_filtered]
    corrected_epochs_hmm = y_hmm_filtered[y_cnn_filtered != y_hmm_filtered]
    corrected_epochs_true = y_true_filtered[y_cnn_filtered != y_hmm_filtered]

    n_corrected_epochs = np.zeros((1, 4))
    correction_matrix = np.zeros((3, 3, 3))
    for s in range(3):
        corrected_epochs_cnn_s = corrected_epochs_cnn[corrected_epochs_true == s]
        corrected_epochs_hmm_s = corrected_epochs_hmm[corrected_epochs_true == s]

        n_corrected_epochs[0, s] = len(corrected_epochs_cnn_s)

        for i in range(len(corrected_epochs_cnn_s)):
            correction_matrix[s, corrected_epochs_cnn_s[i], corrected_epochs_hmm_s[i]] += 1

    n_corrected_epochs[0, 3] = np.sum(n_corrected_epochs[0, :3])

    cnn_transitions = calculate_transition_matrix(y_cnn)
    hmm_transitions = calculate_transition_matrix(y_hmm)

    return n_corrected_epochs, correction_matrix, cnn_transitions, hmm_transitions


def plot_and_save_hmm_effect(n_corrected_epochs,
                             correction_matrix,
                             cnn_transitions,
                             hmm_transitions,
                             save_path):
    correction_matrix = (correction_matrix / np.reshape(n_corrected_epochs[0, :3], (3, 1, 1)))

    fig, ax = plt.subplots(6, 1, figsize=(6.4, 8.5))
    ax[0].axis('off')
    ax[0].table(cellText=np.round(n_corrected_epochs.reshape((1, 4)), 3), colLabels=['NREM', 'REM', 'WAKE', 'Total'],
                loc='center')
    ax[0].set_title('Number of corrected epochs')
    ax[1].axis('off')
    ax[1].table(cellText=np.round(correction_matrix[0], 3),
                colLabels=['hmm_N', 'hmm_R', 'hmm_W'],
                rowLabels=['cnn_N', 'cnn_R', 'cnn_W'],
                loc='center')
    ax[1].set_title('True NREM')
    ax[2].axis('off')
    ax[2].table(cellText=np.round(correction_matrix[1], 3),
                colLabels=['hmm_N', 'hmm_R', 'hmm_W'],
                rowLabels=['cnn_N', 'cnn_R', 'cnn_W'],
                loc='center')
    ax[2].set_title('True REM')
    ax[3].axis('off')
    ax[3].table(cellText=np.round(correction_matrix[2], 3),
                colLabels=['hmm_N', 'hmm_R', 'hmm_W'],
                rowLabels=['cnn_N', 'cnn_R', 'cnn_W'],
                loc='center')
    ax[3].set_title('True WAKE')
    ax[4].axis('off')
    ax[4].table(cellText=np.round(cnn_transitions, 3),
                colLabels=['i=N', 'i=R', 'i=W'],
                rowLabels=['(i-1)=N', '(i-1)=R', '(i-1)=W'],
                loc='center')
    ax[4].set_title('CNN predictions transitions')
    ax[5].axis('off')
    ax[5].table(cellText=np.round(hmm_transitions, 3),
                colLabels=['i=N', 'i=R', 'i=W'],
                rowLabels=['(i-1)=N', '(i-1)=R', '(i-1)=W'],
                loc='center')
    ax[5].set_title('HMM predictions transitions')

    plt.show()
    plt.savefig(os.path.join(save_path, 'hmm_metrics' + '.png'))
