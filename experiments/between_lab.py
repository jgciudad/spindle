import numpy as np 
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt 
import seaborn as sns

##################################################   Lab 1   ##################################################
#[1]: wakefulness [2]: non-REM sleep [3]: REM sleep


df1_sc1 = pd.read_csv('/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/preprocessed/df_lab1_sc1.csv')
df1_sc2 = pd.read_csv('/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/preprocessed/df_lab1_sc2.csv')

df1_sc1 = df1_sc1.replace({1: 0, 2: 1, 3: 2,0: 4})
df1_sc2 = df1_sc2.replace({1: 0, 2: 1, 3: 2,0: 4})
df1_sc1 = df1_sc1.applymap(lambda x: 4 if x not in [0, 1, 2] else x)
df1_sc2 = df1_sc2.applymap(lambda x: 4 if x not in [0, 1, 2] else x)


df2_sc1 = pd.read_csv('/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/preprocessed/df_lab2_sc1.csv')
df2_sc2 = pd.read_csv('/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/preprocessed/df_lab2_sc2.csv')

df2_sc1 = df2_sc1.replace({1: 0, 2: 1, 4: 2,0: 4})
df2_sc2 = df2_sc2.replace({1: 0, 2: 1, 4: 2,0: 4})
df2_sc1 = df2_sc1.applymap(lambda x: 4 if x not in [0, 1, 2] else x)
df2_sc2 = df2_sc2.applymap(lambda x: 4 if x not in [0, 1, 2] else x)

df3_sc1 = pd.read_csv('/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/preprocessed/df_lab3_sc1.csv')
df3_sc2 = pd.read_csv('/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/preprocessed/df_lab3_sc2.csv')

df3_sc1 = df3_sc1.replace({1: 0, 2: 1, 3: 2,0: 4})
df3_sc2 = df3_sc2.replace({1: 0, 2: 1, 3: 2,0: 4})
df3_sc1 = df3_sc1.applymap(lambda x: 4 if x not in [0, 1, 2] else x)

df4_sc1 = pd.read_csv('/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/preprocessed/df_lab4_sc1.csv')
df4_sc2 = pd.read_csv('/zhome/dd/4/109414/Validationstudy/spindle/consensus_data/preprocessed/df_lab4_sc2.csv')

df4_sc1 = df4_sc1.replace({1: 0, 2: 1, 3: 2,0: 4})
df4_sc2 = df4_sc2.replace({1: 0, 2: 1, 3: 2,0: 4})
df4_sc1 = df4_sc1.applymap(lambda x: 4 if x not in [0, 1, 2] else x)

##################################################   Lab 1 vs. rest  ##################################################
lab1_vsrest = []

for j in range(df1_sc1.shape[1]): # loops across mice 
    s1_n1 = [] 
    s1_n2 = []
    s1_n3 = [] 
    s1_n4 = []
    s1_n5 = [] 
    s1_n6 = []

    s2_n1 = [] 
    s2_n2 = []
    s2_n3 = [] 
    s2_n4 = []
    s2_n5 = [] 
    s2_n6 = []

    for k in range(3):            # loops across stages 
        s1_n1.append(cohen_kappa_score(df1_sc1.iloc[:,j]==k, df2_sc1.iloc[:,j]==k))     
        s1_n2.append(cohen_kappa_score(df1_sc1.iloc[:,j]==k, df2_sc2.iloc[:,j]==k))     
        s1_n3.append(cohen_kappa_score(df1_sc1.iloc[:,j]==k, df3_sc1.iloc[:,j]==k))     
        s1_n4.append(cohen_kappa_score(df1_sc1.iloc[:,j]==k, df3_sc2.iloc[:,j]==k))     
        s1_n5.append(cohen_kappa_score(df1_sc1.iloc[:,j]==k, df4_sc1.iloc[:,j]==k))    
        s1_n6.append(cohen_kappa_score(df1_sc1.iloc[:,j]==k, df4_sc2.iloc[:,j]==k)) 

        s2_n1.append(cohen_kappa_score(df1_sc2.iloc[:,j]==k, df2_sc1.iloc[:,j]==k))     
        s2_n2.append(cohen_kappa_score(df1_sc2.iloc[:,j]==k, df2_sc2.iloc[:,j]==k))     
        s2_n3.append(cohen_kappa_score(df1_sc2.iloc[:,j]==k, df3_sc1.iloc[:,j]==k))     
        s2_n4.append(cohen_kappa_score(df1_sc2.iloc[:,j]==k, df3_sc2.iloc[:,j]==k))     
        s2_n5.append(cohen_kappa_score(df1_sc2.iloc[:,j]==k, df4_sc1.iloc[:,j]==k))    
        s2_n6.append(cohen_kappa_score(df1_sc2.iloc[:,j]==k, df4_sc2.iloc[:,j]==k)) 
 
    vec = np.array([np.mean(s1_n1),np.mean(s1_n2),np.mean(s1_n3),np.mean(s1_n4),np.mean(s1_n5),np.mean(s1_n6),np.mean(s2_n1),np.mean(s2_n2),np.mean(s2_n3),np.mean(s2_n4),np.mean(s2_n5),np.mean(s2_n6)])
    lab1_vsrest.append(vec)



##################################################   Lab 2 vs. rest  ##################################################
lab2_vsrest = []

for j in range(df1_sc1.shape[1]): # loops across mice 
    s1_n1 = [] 
    s1_n2 = []
    s1_n3 = [] 
    s1_n4 = []
    s1_n5 = [] 
    s1_n6 = []

    s2_n1 = [] 
    s2_n2 = []
    s2_n3 = [] 
    s2_n4 = []
    s2_n5 = [] 
    s2_n6 = []

    for k in range(3):            # loops across stages 
        s1_n1.append(cohen_kappa_score(df2_sc1.iloc[:,j]==k, df1_sc1.iloc[:,j]==k))     
        s1_n2.append(cohen_kappa_score(df2_sc1.iloc[:,j]==k, df1_sc2.iloc[:,j]==k))     
        s1_n3.append(cohen_kappa_score(df2_sc1.iloc[:,j]==k, df3_sc1.iloc[:,j]==k))     
        s1_n4.append(cohen_kappa_score(df2_sc1.iloc[:,j]==k, df3_sc2.iloc[:,j]==k))     
        s1_n5.append(cohen_kappa_score(df2_sc1.iloc[:,j]==k, df4_sc1.iloc[:,j]==k))    
        s1_n6.append(cohen_kappa_score(df2_sc1.iloc[:,j]==k, df4_sc2.iloc[:,j]==k)) 

        s2_n1.append(cohen_kappa_score(df2_sc2.iloc[:,j]==k, df1_sc1.iloc[:,j]==k))     
        s2_n2.append(cohen_kappa_score(df2_sc2.iloc[:,j]==k, df1_sc2.iloc[:,j]==k))     
        s2_n3.append(cohen_kappa_score(df2_sc2.iloc[:,j]==k, df3_sc1.iloc[:,j]==k))     
        s2_n4.append(cohen_kappa_score(df2_sc2.iloc[:,j]==k, df3_sc2.iloc[:,j]==k))     
        s2_n5.append(cohen_kappa_score(df2_sc2.iloc[:,j]==k, df4_sc1.iloc[:,j]==k))    
        s2_n6.append(cohen_kappa_score(df2_sc2.iloc[:,j]==k, df4_sc2.iloc[:,j]==k)) 
 
    vec = np.array([np.mean(s1_n1),np.mean(s1_n2),np.mean(s1_n3),np.mean(s1_n4),np.mean(s1_n5),np.mean(s1_n6),np.mean(s2_n1),np.mean(s2_n2),np.mean(s2_n3),np.mean(s2_n4),np.mean(s2_n5),np.mean(s2_n6)])
    lab2_vsrest.append(vec)



##################################################   Lab 3 vs. rest  ##################################################
lab3_vsrest = []

for j in range(df1_sc1.shape[1]): # loops across mice 
    s1_n1 = [] 
    s1_n2 = []
    s1_n3 = [] 
    s1_n4 = []
    s1_n5 = [] 
    s1_n6 = []

    s2_n1 = [] 
    s2_n2 = []
    s2_n3 = [] 
    s2_n4 = []
    s2_n5 = [] 
    s2_n6 = []

    for k in range(3):            # loops across stages 
        s1_n1.append(cohen_kappa_score(df3_sc1.iloc[:,j]==k, df1_sc1.iloc[:,j]==k))     
        s1_n2.append(cohen_kappa_score(df3_sc1.iloc[:,j]==k, df1_sc2.iloc[:,j]==k))     
        s1_n3.append(cohen_kappa_score(df3_sc1.iloc[:,j]==k, df2_sc1.iloc[:,j]==k))     
        s1_n4.append(cohen_kappa_score(df3_sc1.iloc[:,j]==k, df2_sc2.iloc[:,j]==k))     
        s1_n5.append(cohen_kappa_score(df3_sc1.iloc[:,j]==k, df4_sc1.iloc[:,j]==k))    
        s1_n6.append(cohen_kappa_score(df3_sc1.iloc[:,j]==k, df4_sc2.iloc[:,j]==k)) 

        s2_n1.append(cohen_kappa_score(df3_sc2.iloc[:,j]==k, df1_sc1.iloc[:,j]==k))     
        s2_n2.append(cohen_kappa_score(df3_sc2.iloc[:,j]==k, df1_sc2.iloc[:,j]==k))     
        s2_n3.append(cohen_kappa_score(df3_sc2.iloc[:,j]==k, df2_sc1.iloc[:,j]==k))     
        s2_n4.append(cohen_kappa_score(df3_sc2.iloc[:,j]==k, df2_sc2.iloc[:,j]==k))     
        s2_n5.append(cohen_kappa_score(df3_sc2.iloc[:,j]==k, df4_sc1.iloc[:,j]==k))    
        s2_n6.append(cohen_kappa_score(df3_sc2.iloc[:,j]==k, df4_sc2.iloc[:,j]==k)) 
 
    vec = np.array([np.mean(s1_n1),np.mean(s1_n2),np.mean(s1_n3),np.mean(s1_n4),np.mean(s1_n5),np.mean(s1_n6),np.mean(s2_n1),np.mean(s2_n2),np.mean(s2_n3),np.mean(s2_n4),np.mean(s2_n5),np.mean(s2_n6)])
    lab3_vsrest.append(vec)

##################################################   Lab 4 vs. rest  ##################################################
lab4_vsrest = []

for j in range(df1_sc1.shape[1]): # loops across mice 
    s1_n1 = [] 
    s1_n2 = []
    s1_n3 = [] 
    s1_n4 = []
    s1_n5 = [] 
    s1_n6 = []

    s2_n1 = [] 
    s2_n2 = []
    s2_n3 = [] 
    s2_n4 = []
    s2_n5 = [] 
    s2_n6 = []

    for k in range(3):            # loops across stages 
        s1_n1.append(cohen_kappa_score(df4_sc1.iloc[:,j]==k, df1_sc1.iloc[:,j]==k))     
        s1_n2.append(cohen_kappa_score(df4_sc1.iloc[:,j]==k, df1_sc2.iloc[:,j]==k))     
        s1_n3.append(cohen_kappa_score(df4_sc1.iloc[:,j]==k, df2_sc1.iloc[:,j]==k))     
        s1_n4.append(cohen_kappa_score(df4_sc1.iloc[:,j]==k, df2_sc2.iloc[:,j]==k))     
        s1_n5.append(cohen_kappa_score(df4_sc1.iloc[:,j]==k, df3_sc1.iloc[:,j]==k))    
        s1_n6.append(cohen_kappa_score(df4_sc1.iloc[:,j]==k, df3_sc2.iloc[:,j]==k)) 

        s2_n1.append(cohen_kappa_score(df4_sc2.iloc[:,j]==k, df1_sc1.iloc[:,j]==k))     
        s2_n2.append(cohen_kappa_score(df4_sc2.iloc[:,j]==k, df1_sc2.iloc[:,j]==k))     
        s2_n3.append(cohen_kappa_score(df4_sc2.iloc[:,j]==k, df2_sc1.iloc[:,j]==k))     
        s2_n4.append(cohen_kappa_score(df4_sc2.iloc[:,j]==k, df2_sc2.iloc[:,j]==k))     
        s2_n5.append(cohen_kappa_score(df4_sc2.iloc[:,j]==k, df3_sc1.iloc[:,j]==k))    
        s2_n6.append(cohen_kappa_score(df4_sc2.iloc[:,j]==k, df3_sc2.iloc[:,j]==k)) 
 
    vec = np.array([np.mean(s1_n1),np.mean(s1_n2),np.mean(s1_n3),np.mean(s1_n4),np.mean(s1_n5),np.mean(s1_n6),np.mean(s2_n1),np.mean(s2_n2),np.mean(s2_n3),np.mean(s2_n4),np.mean(s2_n5),np.mean(s2_n6)])
    lab4_vsrest.append(vec)


print("done")

print(np.mean(np.array([np.mean(lab1_vsrest),np.mean(lab2_vsrest),np.mean(lab3_vsrest),np.mean(lab4_vsrest)])))

df1 = pd.DataFrame(lab1_vsrest).T
df2 = pd.DataFrame(lab2_vsrest).T
df3 = pd.DataFrame(lab3_vsrest).T
df4 = pd.DataFrame(lab4_vsrest).T

stacked_df = pd.concat([df1,df2,df3,df4], axis=0).reset_index(drop=True)
plot_df = pd.DataFrame({"Value": stacked_df.mean()})
plot_df["animals"] = ["M1_3am6am","M1_3pm6pm","M3_3am6am","M3_3pm6pm", "M4_3am6am", "M4_3pm6pm", "M5_3pm6pm", "M8_3am6am", "M8_3pm6pm"]
plot_df["Phase"]   = plot_df['animals'].apply(lambda x: 'Dark' if '3am6am' in x else 'Light')
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.figure(figsize=(6, 6))
sns.boxplot(x='Phase', y='Value',data=plot_df, whis=[0, 100], color=default_colors[0])
sns.stripplot(x='Phase', y='Value', data=plot_df, dodge=True, jitter=True, marker='o', alpha=0.5, color="black")
plt.title('Between lab Agreement')
plt.xlabel('Lab')
plt.ylabel('Cohens kappa')
plt.ylim((0.7, 1))
plt.savefig("/zhome/dd/4/109414/Validationstudy/spindle/results/consensus/between_lab.png",dpi=300)


## 1) lab1 vs. rest få dette godkendt osv. 
## 2) lav opløselighed på w, n, r 
## 3) lav opløselighed på lights on / lights off
## 4) kig på power plots ud fra label scores (for sjov)

#######################################################################################################################

##################################################   Lab 1 vs. rest  ##################################################
lab1_vsrest = []

for j in range(df1_sc1.shape[1]): # loops across mice 
    s1_n1 = [] 
    s1_n2 = []
    s1_n3 = [] 
    s1_n4 = []
    s1_n5 = [] 
    s1_n6 = []

    s2_n1 = [] 
    s2_n2 = []
    s2_n3 = [] 
    s2_n4 = []
    s2_n5 = [] 
    s2_n6 = []

    for k in range(3):            # loops across stages 
        s1_n1.append(cohen_kappa_score(df1_sc1.iloc[:,j]==k, df2_sc1.iloc[:,j]==k))     
        s1_n2.append(cohen_kappa_score(df1_sc1.iloc[:,j]==k, df2_sc2.iloc[:,j]==k))     
        s1_n3.append(cohen_kappa_score(df1_sc1.iloc[:,j]==k, df3_sc1.iloc[:,j]==k))     
        s1_n4.append(cohen_kappa_score(df1_sc1.iloc[:,j]==k, df3_sc2.iloc[:,j]==k))     
        s1_n5.append(cohen_kappa_score(df1_sc1.iloc[:,j]==k, df4_sc1.iloc[:,j]==k))    
        s1_n6.append(cohen_kappa_score(df1_sc1.iloc[:,j]==k, df4_sc2.iloc[:,j]==k)) 

        s2_n1.append(cohen_kappa_score(df1_sc2.iloc[:,j]==k, df2_sc1.iloc[:,j]==k))     
        s2_n2.append(cohen_kappa_score(df1_sc2.iloc[:,j]==k, df2_sc2.iloc[:,j]==k))     
        s2_n3.append(cohen_kappa_score(df1_sc2.iloc[:,j]==k, df3_sc1.iloc[:,j]==k))     
        s2_n4.append(cohen_kappa_score(df1_sc2.iloc[:,j]==k, df3_sc2.iloc[:,j]==k))     
        s2_n5.append(cohen_kappa_score(df1_sc2.iloc[:,j]==k, df4_sc1.iloc[:,j]==k))    
        s2_n6.append(cohen_kappa_score(df1_sc2.iloc[:,j]==k, df4_sc2.iloc[:,j]==k)) 
 
    vec = np.array([s1_n1,s1_n2,s1_n3,s1_n4,s1_n5,s1_n6,s2_n1,s2_n2,s2_n3,s2_n4,s2_n5,s2_n6])
    lab1_vsrest.append(vec)



##################################################   Lab 2 vs. rest  ##################################################
lab2_vsrest = []

for j in range(df1_sc1.shape[1]): # loops across mice 
    s1_n1 = [] 
    s1_n2 = []
    s1_n3 = [] 
    s1_n4 = []
    s1_n5 = [] 
    s1_n6 = []

    s2_n1 = [] 
    s2_n2 = []
    s2_n3 = [] 
    s2_n4 = []
    s2_n5 = [] 
    s2_n6 = []

    for k in range(3):            # loops across stages 
        s1_n1.append(cohen_kappa_score(df2_sc1.iloc[:,j]==k, df1_sc1.iloc[:,j]==k))     
        s1_n2.append(cohen_kappa_score(df2_sc1.iloc[:,j]==k, df1_sc2.iloc[:,j]==k))     
        s1_n3.append(cohen_kappa_score(df2_sc1.iloc[:,j]==k, df3_sc1.iloc[:,j]==k))     
        s1_n4.append(cohen_kappa_score(df2_sc1.iloc[:,j]==k, df3_sc2.iloc[:,j]==k))     
        s1_n5.append(cohen_kappa_score(df2_sc1.iloc[:,j]==k, df4_sc1.iloc[:,j]==k))    
        s1_n6.append(cohen_kappa_score(df2_sc1.iloc[:,j]==k, df4_sc2.iloc[:,j]==k)) 

        s2_n1.append(cohen_kappa_score(df2_sc2.iloc[:,j]==k, df1_sc1.iloc[:,j]==k))     
        s2_n2.append(cohen_kappa_score(df2_sc2.iloc[:,j]==k, df1_sc2.iloc[:,j]==k))     
        s2_n3.append(cohen_kappa_score(df2_sc2.iloc[:,j]==k, df3_sc1.iloc[:,j]==k))     
        s2_n4.append(cohen_kappa_score(df2_sc2.iloc[:,j]==k, df3_sc2.iloc[:,j]==k))     
        s2_n5.append(cohen_kappa_score(df2_sc2.iloc[:,j]==k, df4_sc1.iloc[:,j]==k))    
        s2_n6.append(cohen_kappa_score(df2_sc2.iloc[:,j]==k, df4_sc2.iloc[:,j]==k)) 
 
    vec = np.array([s1_n1,s1_n2,s1_n3,s1_n4,s1_n5,s1_n6,s2_n1,s2_n2,s2_n3,s2_n4,s2_n5,s2_n6])
    lab2_vsrest.append(vec)



##################################################   Lab 3 vs. rest  ##################################################
lab3_vsrest = []

for j in range(df1_sc1.shape[1]): # loops across mice 
    s1_n1 = [] 
    s1_n2 = []
    s1_n3 = [] 
    s1_n4 = []
    s1_n5 = [] 
    s1_n6 = []

    s2_n1 = [] 
    s2_n2 = []
    s2_n3 = [] 
    s2_n4 = []
    s2_n5 = [] 
    s2_n6 = []

    for k in range(3):            # loops across stages 
        s1_n1.append(cohen_kappa_score(df3_sc1.iloc[:,j]==k, df1_sc1.iloc[:,j]==k))     
        s1_n2.append(cohen_kappa_score(df3_sc1.iloc[:,j]==k, df1_sc2.iloc[:,j]==k))     
        s1_n3.append(cohen_kappa_score(df3_sc1.iloc[:,j]==k, df2_sc1.iloc[:,j]==k))     
        s1_n4.append(cohen_kappa_score(df3_sc1.iloc[:,j]==k, df2_sc2.iloc[:,j]==k))     
        s1_n5.append(cohen_kappa_score(df3_sc1.iloc[:,j]==k, df4_sc1.iloc[:,j]==k))    
        s1_n6.append(cohen_kappa_score(df3_sc1.iloc[:,j]==k, df4_sc2.iloc[:,j]==k)) 

        s2_n1.append(cohen_kappa_score(df3_sc2.iloc[:,j]==k, df1_sc1.iloc[:,j]==k))     
        s2_n2.append(cohen_kappa_score(df3_sc2.iloc[:,j]==k, df1_sc2.iloc[:,j]==k))     
        s2_n3.append(cohen_kappa_score(df3_sc2.iloc[:,j]==k, df2_sc1.iloc[:,j]==k))     
        s2_n4.append(cohen_kappa_score(df3_sc2.iloc[:,j]==k, df2_sc2.iloc[:,j]==k))     
        s2_n5.append(cohen_kappa_score(df3_sc2.iloc[:,j]==k, df4_sc1.iloc[:,j]==k))    
        s2_n6.append(cohen_kappa_score(df3_sc2.iloc[:,j]==k, df4_sc2.iloc[:,j]==k)) 
 
    vec = np.array([s1_n1,s1_n2,s1_n3,s1_n4,s1_n5,s1_n6,s2_n1,s2_n2,s2_n3,s2_n4,s2_n5,s2_n6])
    lab3_vsrest.append(vec)

##################################################   Lab 4 vs. rest  ##################################################
lab4_vsrest = []

for j in range(df1_sc1.shape[1]): # loops across mice 
    s1_n1 = [] 
    s1_n2 = []
    s1_n3 = [] 
    s1_n4 = []
    s1_n5 = [] 
    s1_n6 = []

    s2_n1 = [] 
    s2_n2 = []
    s2_n3 = [] 
    s2_n4 = []
    s2_n5 = [] 
    s2_n6 = []

    for k in range(3):            # loops across stages 
        s1_n1.append(cohen_kappa_score(df4_sc1.iloc[:,j]==k, df1_sc1.iloc[:,j]==k))     
        s1_n2.append(cohen_kappa_score(df4_sc1.iloc[:,j]==k, df1_sc2.iloc[:,j]==k))     
        s1_n3.append(cohen_kappa_score(df4_sc1.iloc[:,j]==k, df2_sc1.iloc[:,j]==k))     
        s1_n4.append(cohen_kappa_score(df4_sc1.iloc[:,j]==k, df2_sc2.iloc[:,j]==k))     
        s1_n5.append(cohen_kappa_score(df4_sc1.iloc[:,j]==k, df3_sc1.iloc[:,j]==k))    
        s1_n6.append(cohen_kappa_score(df4_sc1.iloc[:,j]==k, df3_sc2.iloc[:,j]==k)) 

        s2_n1.append(cohen_kappa_score(df4_sc2.iloc[:,j]==k, df1_sc1.iloc[:,j]==k))     
        s2_n2.append(cohen_kappa_score(df4_sc2.iloc[:,j]==k, df1_sc2.iloc[:,j]==k))     
        s2_n3.append(cohen_kappa_score(df4_sc2.iloc[:,j]==k, df2_sc1.iloc[:,j]==k))     
        s2_n4.append(cohen_kappa_score(df4_sc2.iloc[:,j]==k, df2_sc2.iloc[:,j]==k))     
        s2_n5.append(cohen_kappa_score(df4_sc2.iloc[:,j]==k, df3_sc1.iloc[:,j]==k))    
        s2_n6.append(cohen_kappa_score(df4_sc2.iloc[:,j]==k, df3_sc2.iloc[:,j]==k)) 
 
    vec = np.array([s1_n1,s1_n2,s1_n3,s1_n4,s1_n5,s1_n6,s2_n1,s2_n2,s2_n3,s2_n4,s2_n5,s2_n6])
    lab4_vsrest.append(vec)



##################################################  plot  ##################################################


df1 = pd.DataFrame(lab1_vsrest).T
df2 = pd.DataFrame(lab2_vsrest).T
df3 = pd.DataFrame(lab3_vsrest).T
df4 = pd.DataFrame(lab4_vsrest).T

stacked_df = pd.concat([df1,df2,df3,df4], axis=0).reset_index(drop=True)
plot_df = pd.DataFrame({"Value": stacked_df.mean()})
plot_df["animals"] = ["M1_3am6am","M1_3pm6pm","M3_3am6am","M3_3pm6pm", "M4_3am6am", "M4_3pm6pm", "M5_3pm6pm", "M8_3am6am", "M8_3pm6pm"]
plot_df["Phase"]   = plot_df['animals'].apply(lambda x: 'Dark' if '3am6am' in x else 'Light')
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.figure(figsize=(6, 6))
sns.boxplot(x='Phase', y='Value',data=plot_df, whis=[0, 100], color=default_colors[0])
sns.stripplot(x='Phase', y='Value', data=plot_df, dodge=True, jitter=True, marker='o', alpha=0.5, color="black")
plt.title('Between lab Agreement')
plt.xlabel('Lab')
plt.ylabel('Cohens kappa')
plt.ylim((0.7, 1))
plt.savefig("/zhome/dd/4/109414/Validationstudy/spindle/results/consensus/between_lab.png",dpi=300)
