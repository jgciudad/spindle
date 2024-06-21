import numpy as np 
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt 
import seaborn as sns

##################################################   Lab 1   ##################################################
#[1]: wakefulness [2]: non-REM sleep [3]: REM sleep
df1_sc1 = pd.read_csv('/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/preprocessed/df_lab1_sc1.csv')
df1_sc2 = pd.read_csv('/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/preprocessed/df_lab1_sc2.csv')

pr_1     = []
pr_1_all = []
for j in range(df1_sc1.shape[1]):
    pr_n = []
    for k in range(1,4):
        pr_n.append(cohen_kappa_score(df1_sc1.iloc[:,j]==k, df1_sc2.iloc[:,j]==k))        
    pr_1.append(np.mean(pr_n))
    pr_1_all.append(pr_n)
##################################################   Lab 2   ##################################################
# 1 = WAKE / 2 = NREM / 4 = REM / 5 = Artifacts
    
df2_sc1 = pd.read_csv('/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/preprocessed/df_lab2_sc1.csv')
df2_sc2 = pd.read_csv('/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/preprocessed/df_lab2_sc2.csv')
pr_2 = []
pr_2_all = []

for j in range(df1_sc1.shape[1]):
    pr_n = []
    for k in np.array([1,2,4]):
        pr_n.append(cohen_kappa_score(df2_sc1.iloc[:,j]==k, df2_sc2.iloc[:,j]==k))        
    pr_2.append(np.mean(pr_n))
    pr_2_all.append(pr_n)


##################################################   Lab 3   ##################################################
# 1 = WAKE / 2 = NREM / 3 = REM 

df3_sc1 = pd.read_csv('/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/preprocessed/df_lab3_sc1.csv')
df3_sc2 = pd.read_csv('/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/preprocessed/df_lab3_sc2.csv')

pr_3 = []
pr_3_all = []

for j in range(df1_sc1.shape[1]):
    pr_n = []
    for k in np.array([1,2,3]):
        pr_n.append(cohen_kappa_score(df3_sc1.iloc[:,j]==k, df3_sc2.iloc[:,j]==k))        
    pr_3.append(np.mean(pr_n))
    pr_3_all.append(pr_n)

##################################################   Lab 4   ##################################################
# 1 = WAKE / 2 = NREM / 3 = REM 

df4_sc1 = pd.read_csv('/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/preprocessed/df_lab4_sc1.csv')
df4_sc2 = pd.read_csv('/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/preprocessed/df_lab4_sc2.csv')

pr_4 = []
pr_4_all = []

for j in range(df1_sc1.shape[1]):
    pr_n = []
    for k in np.array([1,2,3]):
        pr_n.append(cohen_kappa_score(df4_sc1.iloc[:,j]==k, df4_sc2.iloc[:,j]==k))        
    pr_4.append(np.mean(pr_n))
    pr_4_all.append(pr_n)

##################################################   Lab 5   ##################################################
#1 = Wake 2 = SWS (NON REM) 4 = PS (REM) 5 = artefact # scored in 5 seconds but labels for 1s 
    
df5_sc1 = pd.read_csv('/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/preprocessed/df_lab5_sc1.csv')
df5_sc2 = pd.read_csv('/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/preprocessed/df_lab5_sc2.csv')

pr_5 = []
pr_5_all = []

for j in range(df1_sc1.shape[1]):
    pr_n = []
    for k in np.array([1,2,4]):
        pr_n.append(cohen_kappa_score(df5_sc1.iloc[:,j]==k, df5_sc2.iloc[:,j]==k))        
    pr_5.append(np.mean(pr_n))
    pr_5_all.append(pr_n)


##################################################   Plot   ##################################################

df = pd.DataFrame([pr_1,pr_2,pr_3,pr_4,pr_5])
df.columns = ["M1_3am6am","M1_3pm6pm","M3_3am6am","M3_3pm6pm", "M4_3am6am", "M4_3pm6pm", "M5_3pm6pm", "M8_3am6am", "M8_3pm6pm"]

plt.figure(figsize=(10, 6))
df_transposed = df.T
df_long = df_transposed.reset_index().melt(id_vars='index', var_name='Variable', value_name='Value')
df_long.columns = ['Time Period', 'Lab', 'Value']
df_long['Phase'] = df_long['Time Period'].apply(lambda x: 'Dark' if '3am6am' in x else 'Light')

# Plot the box plot using Seaborn
plt.figure(figsize=(12, 6))
sns.boxplot(x='Lab', y='Value', hue='Phase', data=df_long, whis=[0, 100])
sns.stripplot(x='Lab', y='Value', hue='Phase', data=df_long, dodge=True, jitter=True, marker='o', alpha=0.5, color='black')
plt.title('Within lab Agreement')
plt.xlabel('Lab')
plt.ylabel('Cohens kappa')
plt.ylim((0.4, 1))
plt.savefig("/zhome/dd/4/109414/Validationstudy/spindle/results/consensus/within.png",dpi=300)

##################################################   Plot   ##################################################

df = pd.DataFrame(np.concatenate([pr_1_all,pr_2_all,pr_3_all,pr_4_all,pr_5_all]))
df.columns = ["Wake","NREM","REM"]
df["Animal_ID"] = 5*["M1_3am6am","M1_3pm6pm","M3_3am6am","M3_3pm6pm", "M4_3am6am", "M4_3pm6pm", "M5_3pm6pm", "M8_3am6am", "M8_3pm6pm"]
df["Lab"]       = np.concatenate([9*["Lab1"],9*["Lab2"],9*["Lab3"],9*["Lab4"],9*["Lab5"]])

df_melted = pd.melt(df, id_vars=['Animal_ID', 'Lab'], value_vars=['Wake', 'NREM', 'REM'],
                      var_name='Sleep_Stage', value_name='Value')

# Plot the box plot using Seaborn
plt.figure(figsize=(12, 6))
sns.boxplot(x='Lab', y='Value', hue = "Sleep_Stage",data=df_melted, whis=[0, 100])
sns.stripplot(x='Lab', y='Value', hue = "Sleep_Stage",  data=df_melted, dodge=True, jitter=True, marker='o', alpha=0.5, color='black')
plt.title('Within lab Agreement')
plt.xlabel('Lab')
plt.ylabel('Cohens kappa')
plt.ylim((0.4, 1))
plt.savefig("/zhome/dd/4/109414/Validationstudy/spindle/results/consensus/within_ss.png",dpi=300)