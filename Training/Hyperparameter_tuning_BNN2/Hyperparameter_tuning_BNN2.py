par_number = 1
Result_dir = '/zhome/13/e/97883/Documents/Speciale_code/Results/Hyperparameter_tuning_BNN2/'
data_dir = '/zhome/13/e/97883/Documents/Speciale_code/Data/9bus_30150.npz'

'''
Loading packages
'''

import scipy.io as sio
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers, regularizers
import keras
from sklearn.model_selection import train_test_split



'''
Defining functions (ECE)
'''

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
    



'''
Loading in the data
'''

npzfile = np.load(data_dir)
X = npzfile['X']
y = npzfile['y']
y = np.reshape(y, (-1, 1))

##Training/Testing set (stratify)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.14, stratify = y, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify = y_test, random_state=0)

#Normalising the data
X_train_mean = np.mean(X_train, axis=0)
X_train = X_train - X_train_mean
X_test = X_test - X_train_mean #Differences up to 1e-2
X_val = X_val - X_train_mean



no_samples = 10
batch_size_array = [32, 64, 128, 256, 512]
n1_range = [4, 100]
n2_range = [4, 100]
n3_range = [2, 50]
n4_range = [2, 50]
dropout_range = [0, 0.5]



alpha_range = [-4, -1] 
decay_range = [-8, -4] #learning rate decay
#l2_range = [-5, -1] #Regularization parameter

number_of_layers = [2, 3, 4]
number_of_epochs = 2000

#Sequential implementation
accuracies = np.empty((no_samples, 1))
scores = np.empty((no_samples, 1))
ECEs = np.empty((no_samples, 1))
accuracies[:] = np.NaN
scores[:] = np.NaN
ECEs[:] = np.NaN
hyperparameters = np.empty((no_samples, 12))
#batch_size, alpha, decay, nl, n1, n2, n3, n4, n5, dropout_p1, dropout_p2, dropout_p3, dropout_p4, dropout_p5

hyperparameters[:] = np.NaN




def reject_sample(std, cutoff):
    #This is a simple way to enforce something like Inverse Transform Sampling
    reject = True
    while reject:
        candidate = np.random.normal(loc=0, scale=std)
        if (candidate >= 0) and (candidate <= cutoff):
            reject = False
    
    return(candidate)

#Showing, that it works    
#sample_list = [reject_sample(0.4246609, 0.5) for i in range(100000)]
#plt.hist(sample_list, bins=50)



best_acc = 0

best_acc_i = 0
#Can be parallelized
for i in range(no_samples):
    
    print('Trying hyperparameter configuration number: '+str(i+1))
    
    #Draw learning rate
    r = alpha_range[0] + np.random.rand()*(alpha_range[1] - alpha_range[0])
    alpha = 10 ** r
    
    #Draw number of layers
    index = np.random.randint(0, len(number_of_layers))
    nl = number_of_layers[index]
    
    #Draw number of neurons
    n1 = np.random.randint(n1_range[0], n1_range[1])
    n2 = np.random.randint(n2_range[0], n2_range[1])
    n3 = np.NaN
    n4 = np.NaN
    if nl > 2:
        n3 = np.random.randint(n3_range[0], n3_range[1])
        
    if nl > 3:
        n4 = np.random.randint(n4_range[0], n4_range[1])
    
    #Draw dropout rates
    #Having a std of 0.425 ensures, that pdf(0.5) = 1/2 *pdf(0)
    #Having a std of 0.322 ensures that pdf(0.5) = 0.3*pdf(0)
    
    dropout_p1 = reject_sample(std=0.3222157, cutoff=0.5)
    dropout_p2 = reject_sample(std=0.3222157, cutoff=0.5)
    dropout_p3 = np.NaN
    dropout_p4 = np.NaN
    if nl > 2:
        dropout_p3 = reject_sample(std=0.3222157, cutoff=0.5)
    
    if nl > 3:
        dropout_p4 = reject_sample(std=0.3222157, cutoff=0.5)
    
    
    
    
    #Draw batch size
    index = np.random.randint(0, len(batch_size_array))
    batch_size = batch_size_array[index]
    
    #Learning rate decay
    r = decay_range[0] + np.random.rand()*(decay_range[1] - decay_range[0])
    decay = 10**r
        
    
    #Build model
    inputs = keras.Input(shape=(8,))
    x = Dense(n1, activation='relu')(inputs)
    x = Dropout(dropout_p1)(x, training=True)
    x = Dense(n2, activation='relu')(x)
    x = Dropout(dropout_p2)(x, training=True)
    if nl > 2:
        x = Dense(n3, activation='relu')(x)
        x = Dropout(dropout_p3)(x, training=True)
    if nl > 3:
        x = Dense(n4, activation='relu')(x)
        x = Dropout(dropout_p4)(x, training=True)
    x = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=inputs, outputs=x)
    
    
    adam = optimizers.Adam(learning_rate = alpha, decay=decay)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, epochs=number_of_epochs, batch_size=batch_size, verbose=0)
    
    # Scores (including uncertainty)
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
    ECE = bin_acc(y_pred_prob, y_val, M=20, ECE_calc=True)
    
        
    if acc > best_acc:
        model.save(filepath=Result_dir+'best_acc'+str(par_number)+'.h5')
        best_acc = acc
        best_acc_i = i
        hist_df = pd.DataFrame(history.history)
        with open(Result_dir+'history'+str(par_number)+'.csv', mode='w') as f:
            hist_df.to_csv(f)
        
    #Sequential implementation
    accuracies[i, 0] = acc
    ECEs[i, 0] = ECE
    hyperparameters[i, :] = np.array([batch_size, alpha, decay, nl, n1, n2, n3, n4, dropout_p1, dropout_p2, dropout_p3, dropout_p4])

#np.savez('RandomSearch.npz', HypPar=hyperparameters, scores=scores, accs = accuracies)
print('Best accuracy index: '+str(best_acc_i))

CSV = np.concatenate((hyperparameters, accuracies, ECEs), axis=1)
#np.savetxt('test.csv', CSV, delimiter=",", header="batch_size,alpha,decay,l2, nl,n1,n2,n3,n4,score,accuracies")
np.savetxt(Result_dir+'RandomSearch'+str(par_number)+'.csv', CSV, delimiter=",", header="batch_size,alpha,decay,nl,n1,n2,n3,n4,do_p1, do_p2, do_p3, do_p4, accuracies, ECEs")

