par_number = 1
data_dir = '/zhome/13/e/97883/Documents/Speciale_code/Data/BigCombined1.mat'

past_models_dir = '/zhome/13/e/97883/Documents/Speciale_code/Results/HypTun_Big_BNN1/Run1_more/'
past_models_dir2 = '/zhome/13/e/97883/Documents/Speciale_code/Results/HypTun_Big_BNN1/Run1/'
Result_dir = '/zhome/13/e/97883/Documents/Speciale_code/Results/HypTun_Big_BNN1/Run1_more2/'

number_of_epochs = 10000

import scipy.io as sio
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential, load_model
from keras import optimizers, regularizers
from sklearn.model_selection import train_test_split


def bin_acc(y_prob, y_true, M, ECE_calc = False):
    #y_prob: Array of probabilities for predictions
    #y_true: Array of actual labels
    #M: Number of bins
    #ECE_calc: Boolean, whether the expected calibration error should be calculated
    
    if len(y_prob) != len(y_true):
        print('Error: Dimensions dont match')
    
    n = len(y_prob)
    
    #Sorting the predictions into bins
    sorted_indices = np.argsort(y_prob)
    y_prob = y_prob[sorted_indices]
    y_true = y_true[sorted_indices]
    
    
    conf_bins = np.linspace(0, 1, M+1)
    acc = [] #List of accuracies for bins
    Bn = []  #List of number of observations in each bin
    conf = [] #Average confidence in each bin
    
    for i in range(1, M+1):
        index = (conf_bins[i] >= y_prob) & (conf_bins[i-1] <= y_prob)
        #Check if there are any observations in the range
        if np.any(index):
            Bn.append(np.sum(index))
            conf.append(np.mean(y_prob[index]))
            acc.append(np.mean(y_true[index]))
        else:
            Bn.append(0)
            conf.append(0)
            acc.append(0)
        
    if ECE_calc:
        ECE = 0
        for i in range(len(acc)):
            ECE += Bn[i]*np.abs(acc[i] - conf[i])
        ECE = ECE/n
        return(ECE)
    else:
        return(acc)
   
    ECE = bin_acc(y_pred_prob, y_val, M=20, ECE_calc=True)




mat_content = sio.loadmat(data_dir)

X = mat_content['X']
y = mat_content['y'].ravel()
y = np.reshape(y, (-1, 1))
##Training/Testing set (stratify)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.14, stratify = y, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify = y_test, random_state=0)

#Normalising the data
X_train_mean = np.mean(X_train, axis=0)
X_train = X_train - X_train_mean
X_test = X_test - X_train_mean #Differences up to 1e-2
X_val = X_val - X_train_mean

model = keras.models.load_model(past_models_dir+'best_acc'+str(par_number))
A = np.loadtxt(past_models_dir2+'RandomSearch'+str(par_number)+'.csv', delimiter=",")

n = A.shape[0]
indices_acc = (-A[:, -2]).argsort()[0:n]
batch_size, alpha, decay, nl, n1, n2, n3, n4, n5, dropout_p1, dropout_p2, dropout_p3, dropout_p4, dropout_p5, accuracies, ECEs = A[indices_acc[0], :].tolist()
batch_size = int(batch_size)


history = model.fit(X_train, y_train, epochs=number_of_epochs, batch_size=batch_size, verbose=0, validation_data=(X_val, y_val))

no_forward_passes = 1000

Y_pred_MC = np.zeros((len(y_val), no_forward_passes))
Y_pred_prob_MC = np.zeros((len(y_val), no_forward_passes))

for j in range(no_forward_passes):
	y_pred_prob = model.predict(X_val)
	Y_pred_prob_MC[:, j] = y_pred_prob.squeeze()
	y_pred = (y_pred_prob > 0.5).astype(int)
	Y_pred_MC[:, j] = y_pred.squeeze()
	
y_pred_prob = np.mean(Y_pred_prob_MC, axis=1)
y_pred = (y_pred_prob > 0.5).astype(int)

acc = np.mean(np.mean(y_pred.squeeze() == y_val.squeeze()))
ECE = bin_acc(y_pred_prob, y_val, M=15, ECE_calc=True)






model.save(filepath=Result_dir+'best_acc'+str(par_number))
hist_df = pd.DataFrame(history.history)
with open(Result_dir+'history'+str(par_number)+'.csv', mode='w') as f:
    hist_df.to_csv(f)


hyperparameters = np.zeros((1, 16))
hyperparameters[0, :] = np.array([batch_size, alpha, decay, nl, n1, n2, n3, n4, n5, dropout_p1, dropout_p2, dropout_p3, dropout_p4, dropout_p5, acc, ECE])


np.savetxt(Result_dir+'RandomSearch'+str(par_number)+'.csv', hyperparameters, delimiter=",", header="batch_size,alpha,decay,l2,nl,n1,n2,n3,n4,n5,score,accuracies")
