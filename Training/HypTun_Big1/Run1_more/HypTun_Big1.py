par_number = 1
#Result_dir = '/zhome/13/e/97883/Documents/Speciale_code/Results/HypTun_Big1/'
data_dir = '/zhome/13/e/97883/Documents/Speciale_code/Data/BigCombined1.mat'

past_models_dir = '/zhome/13/e/97883/Documents/Speciale_code/Results/HypTun_Big1/Run1/'
Result_dir = '/zhome/13/e/97883/Documents/Speciale_code/Results/HypTun_Big1/Run1_more/'

number_of_epochs = 10000

import scipy.io as sio
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential, load_model
from keras import optimizers, regularizers
from sklearn.model_selection import train_test_split

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

model = keras.models.load_model(past_models_dir+'best_acc'+str(par_number)+'.h5')
A = np.loadtxt(past_models_dir+'RandomSearch'+str(par_number)+'.csv', delimiter=",")
#A = np.loadtxt('/Users/Eigil/Dropbox/DTU/Speciale/Analysis/HypTun_Big1/Run1/'+'RandomSearch'+str(par_number)+'.csv', delimiter=",")

n = A.shape[0]
indices_acc = (-A[:, -1]).argsort()[0:n]
batch_size,alpha,decay,l2,nl,n1,n2,n3,n4,n5,score,accuracies = A[indices_acc[0], :].tolist()
batch_size = int(batch_size)


history = model.fit(X_train, y_train, epochs=number_of_epochs, batch_size=batch_size, verbose=0, validation_data=(X_val, y_val))
score, acc = model.evaluate(X_val, y_val, verbose=0)


model.save(filepath=Result_dir+'best_acc'+str(par_number))
hist_df = pd.DataFrame(history.history)
with open(Result_dir+'history'+str(par_number)+'.csv', mode='w') as f:
    hist_df.to_csv(f)
