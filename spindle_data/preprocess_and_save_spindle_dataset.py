from spindle_data_loading import load_recording
import os
import string
import random
import pandas as pd
import numpy as np

dataset_folder = r'C:\Users\javig\Desktop\SPINDLE dataset\SPINDLE dataset\data (original)\CohortA'
destination_folder = r'C:\Users\javig\Desktop\SPINDLE dataset\SPINDLE dataset\preprocessed_saved_as_kornum_2'

training_validation_labels = [r'scorings\A1.csv',
                              r'scorings\A2.csv']

training_validation_signals = [r'recordings\A1.edf',
                               r'recordings\A2.edf']

test_labels = [r'scorings\A3.csv',
                r'scorings\A4.csv']

test_signals = [r'recordings\A3.edf',
               r'recordings\A4.edf']


def save_to_numpy(data, labels, path, df_all, set):
    filenames = []

    for idx, r in enumerate(data):
        characters = string.ascii_lowercase + string.digits
        filename = ''.join(random.choice(characters) for i in range(16))
        while os.path.exists(os.path.join(path, filename) + '.npy'):
            filename = ''.join(random.choice(characters) for i in range(16))

        np.save(os.path.join(path, filename), r)
        filenames.append(filename)

    df = pd.DataFrame(labels, columns=['NREM', 'REM', 'WAKE'])
    filenames = [f + '.npy' for f in filenames]
    df.insert(0, 'File', filenames)

    if set == 'train':
        df['train'] = 1
        df['validation'] = 0
        df['test'] = 0
    elif set == 'validation':
        df['train'] = 0
        df['validation'] = 1
        df['test'] = 0
    elif set == 'test':
        df['train'] = 0
        df['validation'] = 0
        df['test'] = 1

    if df_all is not None:
        df_all = pd.concat([df_all, df])
    else:
        df_all = df

    return df_all


number_of_files = len(training_validation_signals) + len(test_signals)
file_counter = 1

for i in range(len(test_signals)):
    print('Processing file ', file_counter)
    print('Remaning files: ', number_of_files - file_counter)

    x, y = load_recording([dataset_folder + os.sep + test_signals[i]],
                          [dataset_folder + os.sep + test_labels[i]],
                          scorer=1,
                          just_artifact_labels=False,
                          artifact_to_stages=True,
                          balance_artifacts=False,
                          validation_split=0)

    if i == 0:
        df_all = None

    df_all = save_to_numpy(x, y, destination_folder, df_all, 'test')

    file_counter += 1

for i in range(len(training_validation_signals)):
    print('Processing file ', file_counter)
    print('Remaining files: ', number_of_files - file_counter)

    x_train, x_val, labels_train, labels_val = load_recording([dataset_folder + os.sep + training_validation_signals[i]],
                                                              [dataset_folder + os.sep + training_validation_labels[i]],
                                                              scorer=1,
                                                              just_artifact_labels=False,
                                                              artifact_to_stages=True,
                                                              balance_artifacts=False,
                                                              validation_split=0.15)

    # if i==0:
    #     df_all = None

    df_all = save_to_numpy(x_train, labels_train, destination_folder, df_all, 'train')
    df_all = save_to_numpy(x_val, labels_val, destination_folder, df_all, 'validation')

    file_counter += 1

df_all.to_csv(os.path.dirname(destination_folder) + '/labels_all.csv', index=False)
