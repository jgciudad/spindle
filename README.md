# Project for testing the automatic sleep stage classification model SPINDLE in three different ways 
1) Retraining the original model
2) Training the model on a diverse dataset with fixed n (same N as in original project)
3) Training the model on a large and diverse dataset 

To run an experiment 
1) Create a new config file (see ex.  standard_config_spindle.yml in the config folder)
2) In the utils/data/config_loader.py change the name of the config file two different places <-- change this and use args.. 
3) Run the transform_labs.py script to transform the data from time series EEG and EMG data to images (see spindle paper for pipeline) the data is stored in a h5file
4) Run the training_CNN_fixed_n.py (can be used for all three experiments just change config file) script to train the model
5) run the testing_CNN_spindle.py to test the model. It will output confusion matrices normalized columnwised (precision) and row-wise (recall) and a csv file for each stage the recall, precision, f1-score, accuracy and balance accuracy is calculated
