from hmm import get_transition_matrix_kornum, get_priors_kornum
import os
import numpy as np

data_path = r"C:\Users\javig\Documents\THESIS DATA\Raw kornum lab data\Laura-EEGdata_cleaned\data-Kornum"

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

transition_matrix_cancelling = get_transition_matrix_kornum(training_labels,
                                                            cancel_forbidden_transitions=True)

# transition_matrix_not_cancelling = get_transition_matrix_kornum(training_labels,
#                                                             cancel_forbidden_transitions=False)

initial_probs = get_priors_kornum(training_labels)

np.save('transition_matrix_kornum.npy', transition_matrix_cancelling)
np.save('initial_probs_kornum.npy', initial_probs)

