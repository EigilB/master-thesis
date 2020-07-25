par_number = 1
Result_dir = '/zhome/13/e/97883/Documents/Speciale_code/Results/Hyperparameter_tuning2/'
data_dir = '/zhome/13/e/97883/Documents/Speciale_code/Data/Classification4_50000.mat'


##
#CHANGES FROM HYPERPARAMETER_TUNING1:
# - Bigger data set (approx 35000 observations)
# - Stratified sampling
# - Balanced classes
# - Possibly bigger networks
# - Batch-sizes restricted to be between 32 and 512
# - L2 regularization on all layers are implemented
# - Saves the best model


#GOALS WITH THIS HYPERPARAMETER TUNING:
# - Reduce avoidable bias from approximately 0.7% to 0% (allowing for bigger models,
#longer training)
#- Reduce variance error from approximately 0.8% to 0% (more data, L2 regularization)

#FUTURE HYPERPARAMETER TUNINGS:
# - Implementing early stopping (?)

#Result_direct = '/zhome/13/e/97883/Documents/Speciale_code/Results/Hyperparameter_tuning2/'


'''
Loading in the data
'''

import scipy.io as sio
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers, regularizers
from sklearn.model_selection import train_test_split

#First with limited data (Classification)
#mat_content = sio.loadmat('/Users/Eigil/Dropbox/DTU/Speciale/Data Generation/Classification4_50000.mat')
mat_content = sio.loadmat(data_dir)

X = mat_content['X']
y = mat_content['y'].ravel()

#Computing number of observations for smallest class
counts = np.unique(y, return_counts=True)
n = np.min(counts[1])

mask = np.hstack([np.random.choice(np.where(y == l)[0], n, replace=False)
                      for l in np.unique(y)])

    
print('Number of discarded observations: '+str(len(y) - len(mask)))
X = X[mask]
y = y[mask]
y = np.reshape(y, (-1, 1))
##Training/Testing set (stratify)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.14, stratify = y, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify = y_test, random_state=0)

#Normalising the data
X_train_mean = np.mean(X_train, axis=0)
X_train = X_train - X_train_mean
X_test = X_test - X_train_mean #Differences up to 1e-2
X_val = X_val - X_train_mean



no_samples = 100
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
    if score < best_score:
        model.save(filepath='best_score'+str(par_number)+'.h5')
        best_score = score
        best_score_i = i
        
    if acc > best_acc:
        model.save(filepath='best_acc'+str(par_number)+'.h5')
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
#np.savetxt('test.csv', CSV, delimiter=",", header="batch_size,alpha,decay,l2, nl,n1,n2,n3,n4,score,accuracies")
np.savetxt(Result_dir+'RandomSearch'+str(par_number)+'.csv', CSV, delimiter=",", header="batch_size,alpha,decay,l2,nl,n1,n2,n3,n4,score,accuracies")


#A = np.load('/Users/Eigil/Desktop/RandomSearch.npz')

#'''
#Looking at hyperparameters
#'''
#
#A = []
#for i in range(1, 4+1):
#    A.append(np.loadtxt('/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Hyperparameter_tuning1/RandomSearch'+str(i)+'.csv', delimiter=","))
#
#A = np.vstack(A)
#
#fig, ax = plt.subplots()
#plt.scatter(A[:, 0], A[:, -1], alpha=0.5)
#plt.ylim(0.95, 1)
#ax.set_xscale('log')
#plt.title('Batch size')
#
#
#fig, ax = plt.subplots()
#plt.scatter(A[:, 1], A[:, -1], alpha=0.5)
#plt.ylim(0.95, 1)
#plt.title('Alpha')
#
#fig, ax = plt.subplots()
#plt.scatter(A[:, 2], A[:, -1], alpha=0.5)
#plt.xlim([10**(-8), 10**(-4)])
#plt.ylim([0.95, 1])
#ax.set_xscale('log')
#
#
#fig, ax = plt.subplots()
#plt.scatter(A[:, 3], A[:, -1], alpha=0.5)
#plt.ylim([0.95, 1])
#plt.title('Number of layers')
#
##Finding the 10 highest-yielding hyper-parameters
#n = 10
#indices_acc = (-A[:, -1]).argsort()[0:n]
#indices_score = (A[:, -2]).argsort()[0:n]
#
#A[indices_acc, :]
#
#A[indices_acc[0], :]
#
##Retraining network
#opt_hyperparameters = A[indices_acc[0], :]
#n1_opt = (opt_hyperparameters[4]).astype(int)
#n2_opt = (opt_hyperparameters[5]).astype(int)
#batch_size_opt = (opt_hyperparameters[0]).astype(int)
#alpha_opt = (opt_hyperparameters[1])
#decay_opt = (opt_hyperparameters[2])
#
#model = Sequential()
#model.add(Dense(n1_opt, input_dim=8, activation='relu', kernel_initializer = 'he_normal'))
#model.add(Dense(n2_opt, activation='relu', kernel_initializer = 'he_normal'))
#model.add(Dense(1, activation='sigmoid'))
#adam = optimizers.Adam(learning_rate = alpha_opt, decay=decay_opt)
#model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
#history = model.fit(X_train, y_train, epochs=10000, batch_size=batch_size_opt, verbose=1)
#model = keras.models.load_model('/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Hyperparameter_tuning1/Full_model.hdf5')
#model.evaluate(x=X_test, y=y_test)


