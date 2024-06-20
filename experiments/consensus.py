import numpy as np 
import pandas as pd 
import scipy.io as sio
import os 
import glob 
import h5py
import matplotlib.pyplot as plt
################################## Read Alessandro data and save as a dataframe ##################################
mat_contents = sio.loadmat("/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/Alessandro/Scoring_BO.mat")

# scorer 1 
flattened_arrays = []
for j in range(mat_contents["Scoring_BO"].shape[1]):
    assert len(mat_contents["Scoring_BO"][0, j].flatten())==2700
    flattened_arrays.append(mat_contents["Scoring_BO"][0, j].flatten())

df_lab1_sc1 = pd.DataFrame(flattened_arrays).T

# scorer 2 
flattened_arrays = []
for j in range(mat_contents["Scoring_BO"].shape[1]):
    assert len(mat_contents["Scoring_BO"][0, j].flatten())==2700

    flattened_arrays.append(mat_contents["Scoring_BO"][1, j].flatten())

df_lab1_sc2 = pd.DataFrame(flattened_arrays).T

df_lab1_sc1.to_csv("/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/preprocessed/df_lab1_sc1.csv",index=False)
df_lab1_sc2.to_csv("/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/preprocessed/df_lab1_sc2.csv",index=False)

################################## Read Antoines data and save as a dataframe ##################################

# scorer 1 
base = "/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/Antoine/Hypnogram_Scored/MB/*"
I_   = sorted(glob.glob(base))
flattened_arrays = []
for j in range(len(I_)): 
    assert len(np.concatenate(sio.loadmat(I_[j])["Hypnogram"]))==2700
    flattened_arrays.append(np.concatenate(sio.loadmat(I_[j])["Hypnogram"]))

df_lab2_sc1 = pd.DataFrame(flattened_arrays).T


# scorer 2 
base = "/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/Antoine/Hypnogram_Scored/YY/*"
I_   = sorted(glob.glob(base))
flattened_arrays = []
for j in range(len(I_)): 
    assert len(np.concatenate(sio.loadmat(I_[j])["Hypnogram"]))==2700
    flattened_arrays.append(np.concatenate(sio.loadmat(I_[j])["Hypnogram"]))

df_lab2_sc2 = pd.DataFrame(flattened_arrays).T


df_lab2_sc1.to_csv("/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/preprocessed/df_lab2_sc1.csv",index=False)
df_lab2_sc2.to_csv("/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/preprocessed/df_lab2_sc2.csv",index=False)
################################## Read Kornum data and save as a dataframe ##################################

# scorer 1 
base = "/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/Kornum/BK/*"
I_   = sorted(glob.glob(base))
flattened_arrays = []
for j in range(len(I_)): 
    t_ = pd.read_csv(I_[j], skiprows=10, engine='python', sep='\t', index_col=False).iloc[:, 4]
    assert len(t_)==2700
    flattened_arrays.append(t_)

df_lab3_sc1 = pd.DataFrame(flattened_arrays).T

# scorer 2 
base = "/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/Kornum/CH/*"
I_   = sorted(glob.glob(base))
flattened_arrays = []
for j in range(len(I_)): 
    t_ = pd.read_csv(I_[j], skiprows=10, engine='python', sep='\t', index_col=False).iloc[:, 4]
    assert len(t_)==2700
    flattened_arrays.append(t_)

df_lab3_sc2 = pd.DataFrame(flattened_arrays).T


df_lab3_sc1.to_csv("/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/preprocessed/df_lab3_sc1.csv",index=False)
df_lab3_sc2.to_csv("/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/preprocessed/df_lab3_sc2.csv",index=False)
################################## Read Maiken data and save as a dataframe ##################################
# scorer 1 
base = "/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/Maiken/AT/*"
I_   = sorted(glob.glob(base))
flattened_arrays = []
for j in range(len(I_)): 
    t_ = pd.read_csv(I_[j], skiprows=10, engine='python', sep='\t', index_col=False).iloc[:, 4]
    assert len(t_)==2700
    flattened_arrays.append(t_)

df_lab4_sc1 = pd.DataFrame(flattened_arrays).T

# scorer 2 
base = "/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/Maiken/MA/*"
I_   = sorted(glob.glob(base))
flattened_arrays = []
for j in range(len(I_)): 
    t_ = pd.read_csv(I_[j], skiprows=10, engine='python', sep='\t', index_col=False).iloc[:, 4]
    assert len(t_)==2700
    flattened_arrays.append(t_)

df_lab4_sc2 = pd.DataFrame(flattened_arrays).T


df_lab4_sc1.to_csv("/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/preprocessed/df_lab4_sc1.csv",index=False)
df_lab4_sc2.to_csv("/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/preprocessed/df_lab4_sc2.csv",index=False)

for k in range(df_lab3_sc1.shape[1]):
    fig, axs = plt.subplots(4, 1, figsize=(12, 8))
    axs[0].plot(df_lab3_sc1.iloc[:,k])
    axs[0].set_ylim(0, 4)
    axs[1].plot(df_lab3_sc2.iloc[:,k])
    axs[1].set_ylim(0, 4)
    axs[2].plot(df_lab4_sc1.iloc[:,k])
    axs[2].set_ylim(0, 4)
    axs[3].plot(df_lab4_sc2.iloc[:,k])
    axs[3].set_ylim(0, 4)

    plt.savefig("/zhome/dd/4/109414/Validationstudy/spindle/results/consensus/traces"+str(k)+".png")



################################## Read Sebastien data and save as a dataframe ##################################
# scorer 1 
base = "/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/Sebastien/Sebastien/*"
I_   = sorted(glob.glob(base))
flattened_arrays = []
for j in range(len(I_)): 
    filename = I_[j]
    with h5py.File(filename, 'r') as f:
        # List all groups        
        assert len(np.concatenate(f["hypno"][:]))==10800
        flattened_arrays.append(np.concatenate(f["hypno"][:]))

df_lab5_sc1 = pd.DataFrame(flattened_arrays).T


# scorer 2 
base = "/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/Sebastien/Renato/*"
I_   = sorted(glob.glob(base))
flattened_arrays = []
for j in range(len(I_)): 
    filename = I_[j]
    with h5py.File(filename, 'r') as f:
        # List all groups        
        assert len(np.concatenate(f["hypno"][:]))==10800
        flattened_arrays.append(np.concatenate(f["hypno"][:]))

df_lab5_sc2 = pd.DataFrame(flattened_arrays).T


df_lab5_sc1.to_csv("/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/preprocessed/df_lab5_sc1.csv",index=False)
df_lab5_sc2.to_csv("/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/preprocessed/df_lab5_sc2.csv",index=False)