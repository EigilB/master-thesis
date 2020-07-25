par_number = 1
#Paths
data_dir = '/zhome/13/e/97883/Documents/Speciale_code/Data/BigCombined1.mat'
Result_dir = '/zhome/13/e/97883/Documents/Speciale_code/Results/Ensembles_118/'


#Imports
import scipy.io as sio
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers, regularizers
from sklearn.model_selection import train_test_split
import pandas as pd

#Hyperparameters
batch_size = 512
alpha = 0.0322376065557294
decay = 2.6139698073174E-07
l2 = 0
nl = 2
n1 = 20
n2 = 91
number_of_epochs = 33000
number_of_ensembles = 1
    
dim = 171

#Loading data
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


for ensemble in range(1, number_of_ensembles + 1):
    model = Sequential()
    model.add(Dense(n1, input_dim=dim, activation='relu', kernel_regularizer=regularizers.l2(l2)))
    model.add(Dense(n2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    adam = optimizers.Adam(learning_rate = alpha, decay=decay)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=number_of_epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=0)
    score, acc = model.evaluate(X_val, y_val, verbose=0)
    print('Accuracy of mode number: '+str(par_number))
    model.save(filepath=Result_dir+'model'+str(par_number)+'.h5')
    
    hist_df = pd.DataFrame(history.history)
    with open(Result_dir+'history'+str(par_number)+'.csv', mode='w') as f:
            hist_df.to_csv(f)
    
    
    


