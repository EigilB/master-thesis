par_number = 1
Result_dir = '/zhome/13/e/97883/Documents/Speciale_code/Results/Hyperparameter_tuning3/'
data_dir = '/zhome/13/e/97883/Documents/Speciale_code/Data/9bus_30150.npz'


##
#CHANGES FROM HYPERPARAMETER_TUNING2:
# - Loading of the data is the same every time (not discarded at randomly)
# - Only 10 models trained per job
# - Training history is saved

'''
Loading in the data
'''

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers, regularizers
from sklearn.model_selection import train_test_split
import pandas as pd

#First with limited data (Classification)
#npzfile = np.load('/Users/Eigil/Dropbox/DTU/Speciale/Data/9bus_30150.npz')
npzfile = np.load(data_dir)
X = npzfile['X']
y = npzfile['y']
#mat_content = sio.loadmat(data_dir)

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


alpha_range = [-4, -1]
decay_range = [-8, -4] #learning rate decay
l2_range = [-5, -1] #Regularization parameter

number_of_layers = [2, 3, 4]
number_of_epochs = 2000

#Sequential implementation
accuracies = np.empty((no_samples, 1))
scores = np.empty((no_samples, 1))
accuracies[:] = np.NaN
scores[:] = np.NaN
hyperparameters = np.empty((no_samples, 9))
#Hyperparameters: Batch size, alpha, decay, number of layers, n1, n2, n3, n4
hyperparameters[:] = np.NaN


best_acc = 0
best_score = np.inf
best_acc_i = 0
best_score_i = 0
#Can be parallelized
for i in range(no_samples):
    
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
            
    
    #Draw batch size
    index = np.random.randint(0, len(batch_size_array))
    batch_size = batch_size_array[index]
    
    #Learning rate decay
    r = decay_range[0] + np.random.rand()*(decay_range[1] - decay_range[0])
    decay = 10**r
    
    #Regularization
    l2 = l2_range[0] + np.random.rand()*(l2_range[1] - l2_range[0])
    l2 = 10**l2
    #Each bin 0.1, 0.01, 0.001 have the same probability. 0 should have the same
    if np.random.binomial(1, p=1/(l2_range[1] - l2_range[0] + 2)) == 1:
        l2 = 0
        
    
    #Build model
    model = Sequential()
    model.add(Dense(n1, input_dim=8, activation='relu', kernel_regularizer=regularizers.l2(l2)))
    model.add(Dense(n2, activation='relu'))
    if nl > 2:
        model.add(Dense(n3, activation='relu', kernel_regularizer=regularizers.l2(l2)))
    if nl > 3:
        model.add(Dense(n4, activation = 'relu', kernel_regularizer=regularizers.l2(l2)))
    model.add(Dense(1, activation='sigmoid'))
    
    
    adam = optimizers.Adam(learning_rate = alpha, decay=decay)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, epochs=number_of_epochs, batch_size=batch_size, verbose=0)
    score, acc = model.evaluate(X_val, y_val, verbose=0)
    
    #Save best model wrt. score


    if acc > best_acc:
        #Save model
        model.save(filepath=Result_dir+'best_acc'+str(par_number)+'.h5')
        
        #Save training history
        hist_df = pd.DataFrame(history.history)
        with open(Result_dir+'history'+str(par_number)+'.csv', mode='w') as f:
            hist_df.to_csv(f)
        
        #Update best accuracy
        best_acc = acc
        best_acc_i = i
    
    
    #Sequential implementation
    scores[i, 0] = score
    accuracies[i, 0] = acc
    hyperparameters[i, :] = np.array([batch_size, alpha, decay, l2, nl, n1, n2, n3, n4])
    print('Trying hyperparameter configuration number: '+str(i+1))

#np.savez('RandomSearch.npz', HypPar=hyperparameters, scores=scores, accs = accuracies)
print('Best score index: '+str(best_score_i))
print('Best accuracy index: '+str(best_acc_i))

CSV = np.concatenate((hyperparameters, scores, accuracies), axis=1)
np.savetxt(Result_dir+'RandomSearch'+str(par_number)+'.csv', CSV, delimiter=",", header="batch_size,alpha,decay,l2,nl,n1,n2,n3,n4,score,accuracies")
