#Paths
data_dir = '/zhome/13/e/97883/Documents/Speciale_code/Data/Classification4_50000.mat'
Result_dir = '/zhome/13/e/97883/Documents/Speciale_code/Results/Ensembles/'


#Imports
import scipy.io as sio
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers, regularizers
from sklearn.model_selection import train_test_split

#Hyperparameters
batch_size = 512
alpha = 1.00625393e-03
decay = 3.06898305e-05
l2 = 0
nl = 2
n1 = 64
n2 = 34
number_of_epochs = 5000
number_of_ensembles = 10
    
#Loading in data
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

for ensemble in range(1, number_of_ensembles + 1):
    model = Sequential()
    model.add(Dense(n1, input_dim=8, activation='relu', kernel_regularizer=regularizers.l2(l2)))
    model.add(Dense(n2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    adam = optimizers.Adam(learning_rate = alpha, decay=decay)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=number_of_epochs, batch_size=batch_size, verbose=0)
    score, acc = model.evaluate(X_val, y_val, verbose=0)
    print('Accuracy of mode number: '+str(ensemble))
    model.save(filepath=Result_dir+'model'+str(ensemble)+'.h5')
    

