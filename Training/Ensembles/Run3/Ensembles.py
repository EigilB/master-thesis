#Paths
data_dir = '/zhome/13/e/97883/Documents/Speciale_code/Data/9bus_30150.npz'
Result_dir = '/zhome/13/e/97883/Documents/Speciale_code/Results/Ensembles/Run2/'


#Imports
import scipy.io as sio
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers, regularizers
from sklearn.model_selection import train_test_split
import pandas as pd

#Hyperparameters
batch_size = 128
alpha = 1.63759334e-02
decay = 1.00625393e-03
l2 = 0
nl = 3
n1 = 48
n2 = 12
n3 = 39
number_of_epochs = 2000
number_of_ensembles = 20
    
#Loading in data
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

for ensemble in range(1, number_of_ensembles + 1):
    model = Sequential()
    model.add(Dense(n1, input_dim=8, activation='relu', kernel_regularizer=regularizers.l2(l2)))
    model.add(Dense(n2, activation='relu'))
    model.add(Dense(n3, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    adam = optimizers.Adam(learning_rate = alpha, decay=decay)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=number_of_epochs, batch_size=batch_size, verbose=0)
    score, acc = model.evaluate(X_val, y_val, verbose=0)
    print('Accuracy of mode number: '+str(ensemble))
    model.save(filepath=Result_dir+'model'+str(ensemble)+'.h5')
    
    hist_df = pd.DataFrame(history.history)
    with open(Result_dir+'history'+str(ensemble)+'.csv', mode='w') as f:
            hist_df.to_csv(f)
    
    
    

