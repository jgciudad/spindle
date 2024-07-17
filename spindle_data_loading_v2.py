import mne
import numpy as np
import tensorflow as tf
from labdata.final_preprocessing import preprocess_EEG, preprocess_EMG
import string
import os
import scipy
import random
from scipy.io import loadmat
import matplotlib.pyplot as plt
import h5py 
import tables 
from base.data.data_table import COLUMN_MOUSE_ID, COLUMN_LABEL, COLUMN_LAB


class SequenceDataset2(tf.keras.utils.Sequence):
    
    def __init__(self, data_folder, set ,config,test_lab):

        self.config = config
        self.set = set
        self.BATCH_SIZE = config.BATCH_SIZE
        self.data_fraction = config.DATA_FRACTION
        self.data = None
        self.max_idx = 0
        self.test_lab = test_lab
        self.data_folder = data_folder
        self.file = tables.open_file(data_folder)
        self.labs_and_stages = self.get_lab_and_stage_data()
        self.fixed_n = config.FIXED_N
        if self.set == 'train':
            self.train_validation_split()
            self.train_indices, self.train_dist = self.get_indices(self.labs_and_stages_train,self.fixed_n,True)
            self.val_indices, self.val_dist     = self.get_indices(self.labs_and_stages_val,self.fixed_n,False)
            self.loss_weights = self.get_loss_weights()
            self.train_dataloader = TuebingenDataLoaderSet(indices=self.train_indices, config=config, max_idx=self.max_idx, batch_size=self.BATCH_SIZE, loss_weigths=self.loss_weights)
            self.val_dataloader   = TuebingenDataLoaderSet(indices=self.val_indices, config=config, max_idx=self.max_idx,batch_size=self.BATCH_SIZE, loss_weigths=self.loss_weights)
        else:
            self.indices, _ = self.get_indices(self.labs_and_stages)

        self.file.close()
    
    def __len__(self): # specifies the length of the total number of batches 
        return int(np.floor(len(self.indices) / self.BATCH_SIZE))

    def __getitem__(self, index): 
        if self.data is None:  # open in thread
            self.file = tables.open_file(self.config.DATA_FILE)
            self.data = self.file.root['multiple_labs']

        # Calculate the start and end index for the batch
        start_idx = index * self.BATCH_SIZE
        end_idx = min((index + 1) * self.BATCH_SIZE, len(self.indices))

        batch_features    = []
        batch_labs        = []
        batch_labels      = []

        for idx in range(start_idx, end_idx):
            internal_index = self.indices[idx]
            feature = self.data[internal_index][3]
            label = self.config.STAGES.index(str(self.data[internal_index][COLUMN_LABEL], 'utf-8'))
            lab   = self.config.LABS.index(str(self.data[internal_index][COLUMN_LAB], 'utf-8'))
            w     = self.loss_weights[str(self.data[internal_index][COLUMN_LAB], 'utf-8')][str(self.data[internal_index][COLUMN_LABEL], 'utf-8')]
            batch_features.append(feature)
            batch_labels.append(label)
            batch_labs.append(w)

        # Convert lists to numpy arrays
        batch_features = np.array(batch_features)
        batch_labels = np.array(batch_labels)
        batch_labs = np.array(batch_labs)

        return batch_features, batch_labels, batch_labs


    def get_lab_and_stage_data(self):
            """ load indices of samples in the pytables table for each lab

            if data_fraction is set, load only a random fraction of the indices

            Returns:
                list: list with entries for each lab containing lists with indices of samples in that lab
            """
            lab_and_stage_data = {}
            table = self.file.root['multiple_labs']

            if self.set != 'test':
                for lab in self.config.LABS:
                    if lab != self.test_lab:
                        lab_and_stage_data[lab] = {}
                        for stage in self.config.STAGES:
                            lab_and_stage_data[lab][stage] = table.get_where_list('({}=="{}") & ({}=="{}")'.format(COLUMN_LAB, lab, COLUMN_LABEL, stage))
                            if lab_and_stage_data[lab][stage].size > 0:
                                if max(lab_and_stage_data[lab][stage]) > self.max_idx:
                                    self.max_idx = max(lab_and_stage_data[lab][stage])
            else:
                lab_and_stage_data[self.test_lab] = {}
                for stage in self.config.STAGES:
                    lab_and_stage_data[self.test_lab][stage] = table.get_where_list('({}=="{}") & ({}=="{}")'.format(COLUMN_LAB, self.test_lab, COLUMN_LABEL, stage))
                    if lab_and_stage_data[self.test_lab][stage].size > 0:
                        if max(lab_and_stage_data[self.test_lab][stage]) > self.max_idx:
                            self.max_idx = max(lab_and_stage_data[self.test_lab][stage])

            return lab_and_stage_data

    def format_dictionary(dictionary: dict):
        """short method to format dicts into a semi-tabular structure"""
        formatted_str = ''
        for x in dictionary:
            formatted_str += '\t{:20s}: {}\n'.format(x, dictionary[x])
        return formatted_str[:-1]


    def get_indices(self, labs_and_stages_set,fixed_n,train):
            """ loads indices of samples in the pytables table the dataloader returns

            if flag `balanced` is set, rebalancing is done here by randomly drawing samples from all samples in a stage
            until nitems * BALANCING_WEIGHTS[stage] is reached

            drawing of the samples is done with replacement, so samples can occur more than once in the dataloader """
            indices  = np.empty(0)
            indices_ = np.empty(0)
            random_samples = np.empty(0)
            # spindle self.train_n 56928 self.val_n 14228
            if fixed_n==True: 
                if train==True: 
                    n  = np.round(56928/len(self.labs_and_stages))
                    print("number of labs:")
                    print(len(self.labs_and_stages))
                    print(self.train_n)
                else: 
                    n  = np.round(14228/len(self.labs_and_stages))
                    print(self.val_n)

                for lab in labs_and_stages_set:
                    indices_ = np.empty(0)
                    for stage in labs_and_stages_set[lab]:
                        indices_ = np.r_[indices_, labs_and_stages_set[lab][stage]].astype('int')
                    random_samples = np.r_[random_samples, np.random.choice(indices_, int(n), replace=False)].astype('int')
                indices = np.sort(random_samples)  # the samples are sorted by index for the creation of a transformation matrix

                data_dist = {}
                for lab in labs_and_stages_set:
                    data_dist[lab] = {}
                    for stage in labs_and_stages_set[lab]:
                        count = np.intersect1d(indices, labs_and_stages_set[lab][stage]).size
                        data_dist[lab][stage] = count

            else:
                data_dist = {}
                for lab in labs_and_stages_set:
                    data_dist[lab] = {}
                    for stage in labs_and_stages_set[lab]:
                        data_dist[lab][stage] = labs_and_stages_set[lab][stage].size
            
                for lab in labs_and_stages_set:
                    for stage in labs_and_stages_set[lab]:
                        indices = np.r_[indices, labs_and_stages_set[lab][stage]].astype('int')
                indices = np.sort(indices)  # the samples are sorted by index for the creation of a transformation matrix
            print(len(indices))
            return indices, data_dist

    def train_validation_split(self):
            self.labs_and_stages_train = {}
            self.labs_and_stages_val = {}
            self.val_n   = 0
            self.train_n = 0 
            if self.config.DATA_FRACTION == True:
                num_samples_per_lab_train = int(self.config.ORIGINAL_DATASET_SIZE / len(self.labs_and_stages))
                num_samples_per_lab_val = int(num_samples_per_lab_train * self.config.VALIDATION_SPLIT)

                for lab in self.labs_and_stages:
                    self.labs_and_stages_train[lab] = {}
                    self.labs_and_stages_val[lab] = {}
                    l_size = sum([s.size for s in self.labs_and_stages[lab].values()])

                    for stage in list(self.labs_and_stages[lab])[:-1]:
                        lab_stage_size = self.labs_and_stages[lab][stage].size
                        stage_ratio = lab_stage_size / l_size
                        
                        shuffled_indexes = np.random.permutation(self.labs_and_stages[lab][stage])
                        self.labs_and_stages_train[lab][stage] = np.sort(shuffled_indexes[:-int(num_samples_per_lab_val*stage_ratio)])
                        self.labs_and_stages_val[lab][stage] = np.sort(shuffled_indexes[-int(num_samples_per_lab_val*stage_ratio):])
                        self.train_n += len(self.labs_and_stages_train[lab][stage])
                        self.val_n   += len(self.labs_and_stages_val[lab][stage])
            else:

                for lab in self.labs_and_stages:
                    self.labs_and_stages_train[lab] = {}
                    self.labs_and_stages_val[lab] = {}

                    for stage in list(self.labs_and_stages[lab])[-1]:
                        lab_stage_size = self.labs_and_stages[lab][stage].size
                        
                        shuffled_indexes = np.random.permutation(self.labs_and_stages[lab][stage])
                        self.labs_and_stages_train[lab][stage] = np.sort(shuffled_indexes[:-int(lab_stage_size * self.config.VALIDATION_SPLIT)])
                        self.labs_and_stages_val[lab][stage] = np.sort(shuffled_indexes[-int(lab_stage_size * self.config.VALIDATION_SPLIT):])
                        self.train_n += len(self.labs_and_stages_train[lab][stage])
                        self.val_n   += len(self.labs_and_stages_train[lab][stage])

    def get_loss_weights(self):
            loss_weights = {}
            
            lab_sizes = [sum(self.train_dist[lab].values()) for lab in self.train_dist]
            print("N_training data")
            print(sum(lab_sizes))
        
            for lab in self.train_dist:
                loss_weights[lab] = {}

                for stage in self.config.STAGES[:-1]:
                    # c = len(self.train_dist)
                    # s = 3
                    loss_weights[lab][stage] = sum(lab_sizes) / len(self.train_dist) / 3 / self.train_dist[lab][stage]
                    print(lab)
                    print(stage)
                    print( self.train_dist[lab][stage])
                    print("####")
            print(loss_weights)                        
            return loss_weights
    
class TuebingenDataLoaderSet(SequenceDataset2):
    def __init__(self, indices, config, max_idx, batch_size, loss_weigths=None):
        self.indices = np.random.permutation(indices)
        self.config = config
        self.BATCH_SIZE = batch_size
        self.file = tables.open_file(self.config.DATA_FILE)
        self.file.close()
        self.data = None
        self.max_idx = max_idx
        self.loss_weights = loss_weigths



class SequenceDataset2_test(tf.keras.utils.Sequence):
    
    def __init__(self, data_folder, set ,config,test_lab):
        self.config = config
        self.set = set
        self.BATCH_SIZE = config.BATCH_SIZE
        self.data_fraction = config.DATA_FRACTION
        self.data = None
        self.max_idx = 0
        self.test_lab = test_lab
        self.data_folder = data_folder
        self.file = tables.open_file(data_folder)
        self.labs_and_stages = self.get_lab_and_stage_data()
        self.indices, _ = self.get_indices(self.labs_and_stages)
        self.file.close()
    
    def __len__(self): # specifies the length of the total number of batches 
        return int(np.floor(len(self.indices) / self.BATCH_SIZE))

    def __getitem__(self, index): 
        if self.data is None:  # open in thread
            self.file = tables.open_file(self.config.DATA_FILE)
            self.data = self.file.root['multiple_labs']

        # Calculate the start and end index for the batch
        start_idx = index * self.BATCH_SIZE
        end_idx = min((index + 1) * self.BATCH_SIZE, len(self.indices))

        batch_features    = []
        batch_labs        = []
        batch_labels      = []

        for idx in range(start_idx, end_idx):
            internal_index = self.indices[idx]
            feature = self.data[internal_index][3]
            
            label = self.config.STAGES.index(str(self.data[internal_index][COLUMN_LABEL], 'utf-8'))
            lab = self.config.LABS.index(str(self.data[internal_index][COLUMN_LAB], 'utf-8'))

            batch_features.append(feature)
            batch_labels.append(label)
            batch_labs.append(lab)

        # Convert lists to numpy arrays
        batch_features = np.array(batch_features)
        batch_labels = np.array(batch_labels)
        batch_labs = np.array(batch_labs)

        return batch_features, batch_labels, batch_labs


    def get_lab_and_stage_data(self):
            """ load indices of samples in the pytables table for each lab

            if data_fraction is set, load only a random fraction of the indices

            Returns:
                list: list with entries for each lab containing lists with indices of samples in that lab
            """
            lab_and_stage_data = {}
            table = self.file.root['multiple_labs']

            if self.set != 'test':
                for lab in self.config.LABS:
                    if lab != self.test_lab:
                        lab_and_stage_data[lab] = {}
                        for stage in self.config.STAGES:
                            lab_and_stage_data[lab][stage] = table.get_where_list('({}=="{}") & ({}=="{}")'.format(COLUMN_LAB, lab, COLUMN_LABEL, stage))
                            if lab_and_stage_data[lab][stage].size > 0:
                                if max(lab_and_stage_data[lab][stage]) > self.max_idx:
                                    self.max_idx = max(lab_and_stage_data[lab][stage])
            else:
                lab_and_stage_data[self.test_lab] = {}
                for stage in list(self.config.STAGES)[:-1]:
                    lab_and_stage_data[self.test_lab][stage] = table.get_where_list('({}=="{}") & ({}=="{}")'.format(COLUMN_LAB, self.test_lab, COLUMN_LABEL, stage))
                    if lab_and_stage_data[self.test_lab][stage].size > 0:
                        if max(lab_and_stage_data[self.test_lab][stage]) > self.max_idx:
                            self.max_idx = max(lab_and_stage_data[self.test_lab][stage])

            return lab_and_stage_data

    def format_dictionary(dictionary: dict):
        """short method to format dicts into a semi-tabular structure"""
        formatted_str = ''
        for x in dictionary:
            formatted_str += '\t{:20s}: {}\n'.format(x, dictionary[x])
        return formatted_str[:-1]


    def get_indices(self, labs_and_stages_set):
            """ loads indices of samples in the pytables table the dataloader returns

            if flag `balanced` is set, rebalancing is done here by randomly drawing samples from all samples in a stage
            until nitems * BALANCING_WEIGHTS[stage] is reached

            drawing of the samples is done with replacement, so samples can occur more than once in the dataloader """
            indices = np.empty(0)
            
            data_dist = {}
            for lab in labs_and_stages_set:
                data_dist[lab] = {}
                for stage in labs_and_stages_set[lab]:
                    data_dist[lab][stage] = labs_and_stages_set[lab][stage].size
        
            for lab in labs_and_stages_set:
                for stage in labs_and_stages_set[lab]:
                    indices = np.r_[indices, labs_and_stages_set[lab][stage]].astype('int')
            indices = np.sort(indices)  # the samples are sorted by index for the creation of a transformation matrix

        
            return indices, data_dist

    



