from keras.layers import Dense
from keras.models import Sequential, load_model
from keras import optimizers, regularizers
import numpy as np
import pandas as pd

'''
9 bus systme
'''

'''
Feed-forward Neural Network
'''
path = '/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Hyperparameter_tuning3/'
par_runs = 100
nr_models = 1000
models_pr_par_run = nr_models/par_runs

A = []
for i in range(1, par_runs+1):
    A.append(np.loadtxt(path+'RandomSearch'+str(i)+'.csv', delimiter=","))

A = np.vstack(A)
indices_acc = (-A[:, -1]).argsort()[0:n]
best_model_acc = A[indices_acc[0], :]

batch_size, lr, lr_decay, l2, nl, n1, n2, n3, n4, score, val_acc = best_model_acc.tolist()

print('Best model in terms of accuracy:')
print('Batch size: '+str(batch_size))
print('Learning rate: '+str(lr))
print('Learning rate decay: '+str(lr_decay))
print('Regularization strength: '+str(l2))
print('Number of layers '+str(nl))
print('Layer 1 size '+str(n1))
print('Layer 2 size '+str(n2))
print('Layer 3 size '+str(n3))
print('Layer 4 size '+str(n4))
print('Validation accuracy '+str(val_acc))


A_df = pd.DataFrame(A[indices_acc[0:5], :], columns=['batch_size', 'alpha', 'decay', 'l2', 'nl', 'n1', 'n2', 'n3', 'n4', 'score', 'acc'])
pd.set_option('display.max_columns', None)
print(A_df)



'''
9-bus MC Dropout
'''
path = '/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Hyperparameter_tuning_BNN2/'
par_runs = 100
nr_models = 1000
models_pr_par_run = nr_models/par_runs

A = []
for i in range(1, par_runs+1):
    A.append(np.loadtxt(path+'RandomSearch'+str(i)+'.csv', delimiter=","))

A = np.vstack(A)
indices_acc = (-A[:, -2]).argsort()[0:n]
best_model_acc = A[indices_acc[0:5], :]



batch_size, lr, lr_decay, l2, nl, n1, n2, n3, n4, score, val_acc = best_model_acc.tolist()

print('Best model in terms of accuracy:')
print('Batch size: '+str(batch_size))
print('Learning rate: '+str(lr))
print('Learning rate decay: '+str(lr_decay))
print('Regularization strength: '+str(l2))
print('Number of layers '+str(nl))
print('Layer 1 size '+str(n1))
print('Layer 2 size '+str(n2))
print('Layer 3 size '+str(n3))
print('Layer 4 size '+str(n4))
print('Validation accuracy '+str(val_acc))


A_df = pd.DataFrame(A[indices_acc[0:5], :], columns=['batch_size', 'alpha', 'decay', 'nl', 'n1', 'n2', 'n3', 'n4', 'do_p1', 'do_p2', 'do_p3', 'do_p4', 'acc', 'ECE'])
pd.set_option('display.max_columns', None)
print(A_df)




'''
118 bus system
'''

'''
Feed-forward Neural network
'''
path1 = '/Users/Eigil/Dropbox/DTU/Speciale/Analysis/HypTun_Big1/Run1/'
path2 = '/Users/Eigil/Dropbox/DTU/Speciale/Analysis/HypTun_Big1/Run1_more/'
path3 = '/Users/Eigil/Dropbox/DTU/Speciale/Analysis/HypTun_Big1/Run1_more2/'
par_runs = 100
nr_models = 1000
models_pr_par_run = nr_models/par_runs

#Getting hyperparameters
A_NN = []
for i in range(1, par_runs+1):
    A_NN.append(np.loadtxt(path1+'RandomSearch'+str(i)+'.csv', delimiter=","))

A_NN = np.vstack(A_NN)

#Concatenating histories
pathdict = {0 : path1, 1: path2, 2 : path3}
histories = []
for j in range(3):
    histories_run = []
    for i in range(1, 100+1):
        history_path = pathdict[j]+'history'+str(i)+'.csv'
        history_df = pd.read_csv(history_path)
        history_df = history_df.drop(columns='Unnamed: 0')
        
        history_df.columns = [str(col) + '_' + str(i) for col in history_df.columns]
        #Add to list
        histories_run.append(history_df)
        
    histories_run_df = pd.concat(histories_run, axis=1)
    histories.append(histories_run_df)

histories_df = pd.concat(histories, axis=0)
histories_df = histories_df.reset_index()



n = 50

val_accuracy_col_list = [col for col in histories_df.columns if 'val_accuracy' in col]
histories_df_val = histories_df[val_accuracy_col_list]

val_score_col_list = [col for col in histories_df.columns if 'val_loss' in col]
histories_df_val_score = histories_df[val_score_col_list]


train_accuracy_col_list = ["accuracy_"+str(x) for x in range(1, 100+1)]
histories_df_train = histories_df[train_accuracy_col_list]

n_highest_val_runs = list(histories_df_val.iloc[-1].nlargest(n=n).index)
runs = [s.strip('val_accuracy_') for s in n_highest_val_runs]

n_highest_val = histories_df_val[n_highest_val_runs]


window1 = 30
window2 = 50
highest_val = 0
#Plotting validation accuracy
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10, 4))
for run in runs:
    val_accuracy = histories_df['val_accuracy_'+run]
    train_accuracy = histories_df['accuracy_'+run]
    val_loss = histories_df['val_loss_'+run]
    
    val_index = list(val_accuracy.index)
    train_index = list(train_accuracy.index)
    
    val_accuracy = np.convolve(val_accuracy, np.ones((window1,))/window1, mode='valid')
    train_accuracy = np.convolve(train_accuracy, np.ones((window2,))/window2, mode='valid')
    
    val_indices = np.convolve(val_index, np.ones((window1,))/window1, mode='valid')
    train_indices = np.convolve(train_index, np.ones((window2,))/window2, mode='valid')
    
    axes[0].plot(val_indices, val_accuracy, alpha=0.35, linewidth=0.5)
    axes[1].plot(train_indices, train_accuracy, alpha=0.35, linewidth=1)

axes[0].set_ylim([0.9, 0.985])
axes[1].set_ylim([0.9, 0.985])
axes[0].set_xlabel('Number of epochs')
axes[1].set_xlabel('Number of epochs')
axes[0].set_ylabel('Validation accuracy')
axes[1].set_ylabel('Training accuracy')

fig.suptitle('Smoothed accuracy learning curves for the 50 best 118-bus NN models')


#Finding the best-performing hyperparameters
best_10_indices = [int(s)-1 for s in runs[0:10]]

A_NN = []
for i in range(1, par_runs+1):
    load = np.loadtxt(path1+'RandomSearch'+str(i)+'.csv', delimiter=",")
    best_index = np.argmax(load[:, -1])
    A_NN.append(load[best_index, :])
A_NN = np.vstack(A_NN)

A_NN_df = pd.DataFrame(A_NN[best_10_indices, :], columns=['batch_size', 'alpha', 'decay', 'l2', 'nl', 'n1', 'n2', 'n3', 'n4', 'n5', 'score', 'acc'])
A_NN_df['acc'] = histories_df_val.iloc[-1, best_10_indices].values
A_NN_df['score'] = histories_df_val_score.iloc[-1, best_10_indices].values



#Losses
#for run in runs:
#    val_loss = histories_df['val_loss_'+run]
#    val_index = list(val_loss.index)
#    plt.plot(val_index, val_loss, alpha=0.35)
#plt.ylim([0, 1])
#for run in runs:
#    train_loss = histories_df['loss_'+run]
#    train_index = list(train_loss.index)
#    plt.plot(train_index, train_loss, alpha=0.35)
#plt.ylim([0, 0.4])
#

'''
BNN
'''
path1 = '/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Bay118bus/Run_1_and_2/'
path2 = '/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Bay118bus/Run3_notdone/'
path3 = '/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Bay118bus/Run4/'


par_runs = 1000
i_list = list(range(1, par_runs+1))

A_NN = []
for i in range(1, par_runs+1):
    try:
        A_NN.append(np.loadtxt(path3+'RandomSearch'+str(i)+'.csv', delimiter=","))
    except OSError:
        i_list.remove(i)
        print('Number '+str(i)+' has not converged')
A_NN = np.vstack(A_NN)
A_NN = np.concatenate((np.array(i_list).reshape(-1, 1), A_NN), axis=1)
n = 1000
indices_acc = (-A_NN[:, -3]).argsort()[0:n]


A_NN[indices_acc[0], :] #Best run is 287


A_NN[indices_acc[0:10], :] #Best run is 287
np.sort(A_NN[indices_acc[0:20], 0]).astype(int)




'''
MC-Dropout
'''
path1 = '/Users/Eigil/Dropbox/DTU/Speciale/Analysis/HypTun_Big_BNN1/Run1/'
path2 = '/Users/Eigil/Dropbox/DTU/Speciale/Analysis/HypTun_Big_BNN1/Run1_more/'
path3 = '/Users/Eigil/Dropbox/DTU/Speciale/Analysis/HypTun_Big_BNN1/Run1_more2/'
par_runs = 100
nr_models = 1000
models_pr_par_run = nr_models/par_runs



#Getting hyperparameters
A_NN = []
for i in range(1, par_runs+1):
    A_NN.append(np.loadtxt(path1+'RandomSearch'+str(i)+'.csv', delimiter=","))

A_NN = np.vstack(A_NN)

#Concatenating histories
pathdict = {0 : path1, 1: path2, 2: path3}
histories = []
for j in range(3):
    histories_run = []
    for i in range(1, 100+1):
        history_path = pathdict[j]+'history'+str(i)+'.csv'
        history_df = pd.read_csv(history_path)
        history_df = history_df.drop(columns='Unnamed: 0')
        
        history_df.columns = [str(col) + '_' + str(i) for col in history_df.columns]
        #Add to list
        histories_run.append(history_df)
        
    histories_run_df = pd.concat(histories_run, axis=1)
    histories.append(histories_run_df)

histories_df = pd.concat(histories, axis=0)
histories_df = histories_df.reset_index()



n = 50

val_accuracy_col_list = [col for col in histories_df.columns if 'val_accuracy' in col]
histories_df_val = histories_df[val_accuracy_col_list]

train_accuracy_col_list = ["accuracy_"+str(x) for x in range(1, 100+1)]
histories_df_train = histories_df[train_accuracy_col_list]

n_highest_val_runs = list(histories_df_val.iloc[-1].nlargest(n=n).index)
runs = [s.strip('val_accuracy_') for s in n_highest_val_runs]

n_highest_val = histories_df_val[n_highest_val_runs]


window1 = 30
window2 = 30
highest_val = 0
#Plotting validation accuracy
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10, 4))
for run in runs:
    val_accuracy = histories_df['val_accuracy_'+run]
    train_accuracy = histories_df['accuracy_'+run]
    val_loss = histories_df['val_loss_'+run]
    
    val_index = list(val_accuracy.index)
    train_index = list(train_accuracy.index)
    
    val_accuracy = np.convolve(val_accuracy, np.ones((window1,))/window1, mode='valid')
    train_accuracy = np.convolve(train_accuracy, np.ones((window2,))/window2, mode='valid')
    
    val_indices = np.convolve(val_index, np.ones((window1,))/window1, mode='valid')
    train_indices = np.convolve(train_index, np.ones((window2,))/window2, mode='valid')
    
    axes[0].plot(val_indices, val_accuracy, alpha=0.35, linewidth=0.5)
    axes[1].plot(train_indices, train_accuracy, alpha=0.35, linewidth=1)

axes[0].set_ylim([0.9, 0.985])
axes[1].set_ylim([0.9, 0.985])
axes[0].set_xlabel('Number of epochs')
axes[1].set_xlabel('Number of epochs')
axes[0].set_ylabel('Validation accuracy')
axes[1].set_ylabel('Training accuracy')


plt.tight_layout()
st = fig.suptitle('Smoothed accuracy learning curves for the 50 best 118-bus MC-Dropout models')
fig.tight_layout()
st.set_y(0.95)
fig.subplots_adjust(top=0.85)
plt.show()

#Best run:
print('Best run number: '+str(histories_df_val.iloc[-1].idxmax()))
print('Validation accuracy: '+str(histories_df_val.iloc[-1].max()))




best_10_indices = [int(s)-1 for s in runs[0:10]]

A_NN = []
for i in range(1, par_runs+1):
    load = np.loadtxt(path1+'RandomSearch'+str(i)+'.csv', delimiter=",")
    best_index = np.argmax(load[:, -1])
    A_NN.append(load[best_index, :])
A_NN = np.vstack(A_NN)

A_NN_df = pd.DataFrame(A_NN[best_10_indices, :], columns=['batch_size', 'alpha', 'decay', 'nl', 'n1', 'n2', 'n3', 'n4', 'n5', 'do_p1', 'do_p2', 'do_p3', 'do_p4', 'do_p5', 'acc', 'ECE'])
A_NN_df['acc'] = histories_df_val.iloc[-2, best_10_indices].values


A_NN_df = A_NN_df.iloc[0:5, :]



'''
Ensembles
'''
path1 = '/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Ensembles_118/Run2_history/'
par_runs = 20
nr_models = 20
models_pr_par_run = nr_models/par_runs

#Concatenating histories
pathdict = {0 : path1}
histories = []
for j in range(1):
    histories_run = []
    for i in range(1, par_runs+1):
        history_path = pathdict[j]+'history'+str(i)+'.csv'
        history_df = pd.read_csv(history_path)
        history_df = history_df.drop(columns='Unnamed: 0')
        
        history_df.columns = [str(col) + '_' + str(i) for col in history_df.columns]
        #Add to list
        histories_run.append(history_df)
        
    histories_run_df = pd.concat(histories_run, axis=1)
    histories.append(histories_run_df)

histories_df = pd.concat(histories, axis=0)
histories_df = histories_df.reset_index()





n = 20

val_accuracy_col_list = [col for col in histories_df.columns if 'val_accuracy' in col]
histories_df_val = histories_df[val_accuracy_col_list]
train_accuracy_col_list = ["accuracy_"+str(x) for x in range(1, 20+1)]
histories_df_train = histories_df[train_accuracy_col_list]
n_highest_val_runs = list(histories_df_val.iloc[-1].nlargest(n=n).index)
runs = [s.strip('val_accuracy_') for s in n_highest_val_runs]
n_highest_val = histories_df_val[n_highest_val_runs]
window1 = 30
window2 = 30
highest_val = 0
#Plotting validation accuracy
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10, 4))
for run in runs:
    val_accuracy = histories_df['val_accuracy_'+run]
    train_accuracy = histories_df['accuracy_'+run]
    val_loss = histories_df['val_loss_'+run]
    
    val_index = list(val_accuracy.index)
    train_index = list(train_accuracy.index)
    
    val_accuracy = np.convolve(val_accuracy, np.ones((window1,))/window1, mode='valid')
    train_accuracy = np.convolve(train_accuracy, np.ones((window2,))/window2, mode='valid')
    
    val_indices = np.convolve(val_index, np.ones((window1,))/window1, mode='valid')
    train_indices = np.convolve(train_index, np.ones((window2,))/window2, mode='valid')
    
    axes[0].plot(val_indices, val_accuracy, alpha=0.35, linewidth=0.5)
    axes[1].plot(train_indices, train_accuracy, alpha=0.35, linewidth=1)

axes[0].set_ylim([0.9, 0.985])
axes[1].set_ylim([0.9, 0.985])
axes[0].set_xlabel('Number of epochs')
axes[1].set_xlabel('Number of epochs')
axes[0].set_ylabel('Validation accuracy')
axes[1].set_ylabel('Training accuracy')


plt.tight_layout()
st = fig.suptitle('Smoothed accuracy learning curves for the 20 ensemble 118-bus models')
fig.tight_layout()
st.set_y(0.95)
fig.subplots_adjust(top=0.85)
plt.show()