par_number = 1
Result_dir = '/zhome/13/e/97883/Documents/Speciale_code/Results/Bay9bus/'
data_dir = '/zhome/13/e/97883/Documents/Speciale_code/Data/9bus_30150.npz'
#Extra dirs
data_dir_val = '/zhome/13/e/97883/Documents/Speciale_code/Data/Classification_val_50000.mat'
data_dir_test = '/zhome/13/e/97883/Documents/Speciale_code/Data/Classification_test_50000.mat'


#For own computer
#Result_dir = '/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Bay9bus'
#data_dir = '/Users/Eigil/Dropbox/DTU/Speciale/Data/9bus_30150.npz'
#data_dir_val = '/Users/Eigil/Dropbox/DTU/Speciale/Data/Classification_val_50000.mat'
#data_dir_test =  '/Users/Eigil/Dropbox/DTU/Speciale/Data/Classification_test_50000.mat'


'''
Bayesian Neural Network
'''

import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import pandas as pd
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

'''
Data
'''
dim = 8

#Loading data
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




#Test set 2
mat_content = sio.loadmat(data_dir_test)
X_test2 = mat_content['X']
y_test2 = mat_content['y'].ravel()
y_test2 = np.reshape(y_test2, (-1, 1))
X_test2 = X_test2 - X_train_mean


#Validation set 2 (Needed for temperature Platt scaling)
mat_content = sio.loadmat(data_dir_val)
X_val2 = mat_content['X']
y_val2 = mat_content['y'].ravel()
y_val2 = np.reshape(y_val2, (-1, 1))
X_val2 = X_val2 - X_train_mean

X_train_torch = torch.from_numpy(X_train)
X_val_torch = torch.from_numpy(X_val)
y_val_torch = torch.from_numpy(y_val).double()


train_data = []
for i in range(len(X_train)):
   train_data.append([X_train[i], y_train[i]])



'''
Functions
'''



def nll(y_prob, y_true):
    #Negative log likehood for binary classification
    #The contributions are summed as followed:
    # {1 - p    if    y = 0
    # {p        if    y = 1
    
    #Squeezing the vector
    y_prob = y_prob.squeeze().copy()
    y_true = y_true.squeeze().copy()
    y_prob[y_true == 0] = 1 - y_prob[y_true == 0]
    L = -np.sum(np.log(y_prob+1e-7)) #1e-7 for stability reasons
    return(L)

# Binning 
def bin_acc(y_prob, y_true, M, ECE_calc = False, MCE_calc = False):
    #y_prob: Array of probabilities for predictions
    #y_true: Array of actual labels
    #M: Number of bins
    #ECE_calc: Boolean, whether the expected calibration error should be calculated
    
    #Squeeze
    y_prob = y_prob.squeeze()
    y_true = y_true.squeeze()
    
    #Check for same length
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
            conf.append(np.mean(y_prob[index])) #Average confidence
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
    elif MCE_calc:
        MCE_list = []
        for i in range(len(acc)):
            MCE_list.append(np.abs(acc[i] - conf[i]))
        MCE = np.max(np.array(MCE_list))
        return(MCE)
    else:
        return((acc, Bn, conf))


def n_forward_passes(net, X, no_forward_passes=100, verbose=False):
    Y_pred_prob = np.zeros((X.shape[0], no_forward_passes))
    for i in range(no_forward_passes):
        if verbose and i % 10 == 0:
            print('Forward pass number: '+str(i+1))
        y_pred_prob = net.forward_P(X)
        Y_pred_prob[:, i] = y_pred_prob.detach().numpy().squeeze()
    y_pred_prob_mean = np.mean(Y_pred_prob, axis=1)
    return(y_pred_prob_mean)


'''
Bayes by Backprop
'''


#Sampling in Hyper-parameter space
no_samples = 1
batch_size_array = [32, 64, 128, 256]
n1_range = [4, 100]
n2_range = [4, 100]
n3_range = [2, 50]
n4_range = [2, 50]

#BNN
pi_range = [0, 1]
sigma1_range = [-2, 0] #exp( sigma)
sigma2_range = [-8, -6] #exp(sigma)


alpha_range = [-5, -1]

number_of_layers = [2, 3, 4]
number_of_draws = [1, 2, 5, 10, 20]



number_of_epochs = 2000

accuracies = np.empty((no_samples, 1))
scores = np.empty((no_samples, 1))
ECEs = np.empty((no_samples, 1))
MCEs = np.empty((no_samples, 1))
accuracies[:] = np.NaN
scores[:] = np.NaN
ECEs[:] = np.NaN
MCEs[:] = np.NaN
hyperparameters = np.empty((no_samples, 11))
#batch_size, alpha, decay, nl, n1, n2, n3, n4, n5, dropout_p1, dropout_p2, dropout_p3, dropout_p4, dropout_p5

hyperparameters[:] = np.NaN


best_acc = 0

best_acc_i = 0


class ScaleMixture():
    def __init__(self, pi, sigma1, sigma2):
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
    
    def log_P(self, W):
        P1 = torch.distributions.Normal(0, self.sigma1).log_prob(W).exp()
        P2 = torch.distributions.Normal(0, self.sigma2).log_prob(W).exp()
        P = self.pi * P1 + (1-self.pi) * P2
        return(torch.log(P).sum())
    

class BayesianLinear(nn.Module):
    def __init__(self, n_in, n_out, prior_pi = 0.5, prior_sigma1 = np.exp(-1), 
                 prior_sigma2 = np.exp(-5), he_init = True, stddev=None):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        
        if he_init:
            stddev = np.sqrt(2/n_in)
        
        #Mean of distribution, mu
        self.w_mu = nn.Parameter(torch.Tensor(n_out, n_in).normal_(0, stddev))
        
        #Standard deviation sigma = log(1+exp(rho))
        #Maybe the standard deviations should be sampeld to be smaller?
#        self.w_rho = nn.Parameter(torch.Tensor(n_out, n_in).normal_(0, stddev))
        self.w_rho = nn.Parameter(torch.Tensor(n_out, n_in).uniform_(-5,-4))
        
        #Biases
        self.b_mu = nn.Parameter(torch.Tensor(n_out).normal_(0, stddev))
#        self.b_rho = nn.Parameter(torch.Tensor(n_out).normal_(0, stddev))
        self.b_rho = nn.Parameter(torch.Tensor(n_out).uniform_(-5, -4))
        
        #Complexity cost
        self.prior = ScaleMixture(pi=prior_pi, sigma1= prior_sigma1, sigma2 = prior_sigma2)
        self.log_prior = 0
        self.log_posterior = 0
    
    
    def log_P_posterior(self, w, b):
        #W or b as input
        #Calculated the PDF of the drawn samples
        #Weights
        logw1 = -np.log(np.sqrt(2*np.pi))
        logw2 = -torch.log(self.w_sigma)
        logw3 = -1/2*((w - self.w_mu)/self.w_sigma) ** 2
        logw = (logw1 + logw2 + logw3).sum()
        #Biasses
        logb2 = - torch.log(self.b_sigma)
        logb3 = -1/2*((b - self.b_mu)/self.b_sigma) ** 2
        logb = (logw1 + logb2 + logb3).sum()
        return(logw + logb)
        
    
    def forward(self, x, freeze=False):
        self.w_sigma = nn.Softplus(beta=1)(self.w_rho)
        self.b_sigma = nn.Softplus(beta=1)(self.b_rho)
        
        #Sampling epsilon for weights and biases
        if freeze==False:
            self.w_epsilon = torch.distributions.Normal(0, 1).sample(self.w_rho.size())
            self.b_epsilon = torch.distributions.Normal(0, 1).sample(self.b_rho.size())
            
        #Sampling weights
        w = self.w_mu + self.w_epsilon * self.w_sigma
        b = self.b_mu + self.b_epsilon * self.b_sigma
        
        #Calculating output
        if self.training:
            self.log_prior = self.prior.log_P(w) + self.prior.log_P(b)
            self.log_posterior = self.log_P_posterior(w, b)
        output = F.linear(x, w, b)
        return(output)
        
    

class BNN(nn.Module):
    def __init__(self, architechture, hyperparameter_dict=None):
        if type(hyperparameter_dict) == dict:
            self.sigma1 = hyperparameter_dict['sigma1']
            self.sigma2 = hyperparameter_dict['sigma2']
            self.pi = hyperparameter_dict['pi']
            self.draws = hyperparameter_dict['draws']
        else:
            self.sigma1 = np.exp(-1)
            self.sigma2 = np.exp(-5)
            self.pi = 0.5
            self.draws = 10
            
        super(BNN, self).__init__()
        self.nl = len(architechture) - 1
        self.n1 = BayesianLinear(architechture[0], architechture[1],\
                                 prior_pi = self.pi, prior_sigma1 = self.sigma1, prior_sigma2 = self.sigma2).double()
        self.relu1 = nn.ReLU()
        self.n2 = BayesianLinear(architechture[1], architechture[2],\
                                 prior_pi = self.pi, prior_sigma1 = self.sigma1, prior_sigma2 = self.sigma2).double()
        self.relu2 = nn.ReLU()
        if self.nl > 2:
            self.n3 = BayesianLinear(architechture[2], architechture[3],\
                                     prior_pi = self.pi, prior_sigma1 = self.sigma1, prior_sigma2 = self.sigma2).double()
            self.relu3 = nn.ReLU()
        if self.nl > 3:
            self.n4 = BayesianLinear(architechture[3], architechture[4],\
                                     prior_pi = self.pi, prior_sigma1 = self.sigma1, prior_sigma2 = self.sigma2).double()
            self.relu4 = nn.ReLU()
        if self.nl > 5:
            self.n4 = BayesianLinear(architechture[4], architechture[5],\
                                     prior_pi = self.pi, prior_sigma1 = self.sigma1, prior_sigma2 = self.sigma2).double()
            self.relu4 = nn.ReLU()
            
        if self.nl == 2:
            self.output = BayesianLinear(architechture[2], 1, prior_pi = self.pi, prior_sigma1 = self.sigma1, prior_sigma2 = self.sigma2).double()
        elif self.nl == 3:
            self.output = BayesianLinear(architechture[3], 1, prior_pi = self.pi, prior_sigma1 = self.sigma1, prior_sigma2 = self.sigma2).double()
        elif self.nl == 4:
            self.output = BayesianLinear(architechture[4], 1, prior_pi = self.pi, prior_sigma1 = self.sigma1, prior_sigma2 = self.sigma2).double()
        elif self.nl == 5:
            self.output = BayesianLinear(architechture[5], 1, prior_pi = self.pi, prior_sigma1 = self.sigma1, prior_sigma2 = self.sigma2).double()
    
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, freeze=False):
        x = self.n1.forward(x, freeze=freeze)
        x = self.relu1(x)
        x = self.n2.forward(x, freeze=freeze)
        x = self.relu2(x)
        if self.nl > 2:
            x = self.n3(x, freeze=freeze)
            x = self.relu3(x)
        if self.nl > 3:
            x = self.n4(x, freeze=freeze)
            x = self.relu4(x)
        if self.nl > 4:
            x = self.n5(x, freeze=freeze)
            x = self.relu5(x)
        x = self.output(x)
        #Softmax?
        return(x)
    
    def forward_P(self, x, freeze=False):
        x = self.forward(x, freeze=freeze)
        x = self.sigmoid(x)
        return(x)
    
    
    def log_prior(self):
        prior_cost = 0
        for layer in list(self.children()):
            if type(layer) == BayesianLinear:
                prior_cost = prior_cost + layer.log_prior
        return(prior_cost)
    
    def log_posterior(self):
        posterior_cost = 0
        for layer in list(self.children()):
            if type(layer) == BayesianLinear:
                posterior_cost = posterior_cost + layer.log_posterior
        return(posterior_cost)
    
    def loss(self, x, y, no_minibatches):
        outputs = []
        prior_costs = []
        posterior_costs = []
        for i in range(self.draws):
            outputs.append(self.forward_P(x))
            prior_costs.append(self.log_prior())
            posterior_costs.append(self.log_posterior())
            
        output = torch.cat(outputs, dim=1).mean(dim=1, keepdim=True)
        prior_cost = torch.stack(prior_costs).mean()
        posterior_cost = torch.stack(posterior_costs).mean()
        nll = nn.BCELoss(reduction='sum')(output, y)
        loss = 1/no_minibatches*(posterior_cost - prior_cost) + nll
        return(loss)


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
    
    
    #Draw batch size
    index = np.random.randint(0, len(batch_size_array))
    batch_size = batch_size_array[index]
    
    #Draw mixing parameter
    pi = pi_range[0] + np.random.rand()*(pi_range[1] - pi_range[0])
    
    #Draw standard deviations of prior
    sigma1_uni =sigma1_range[0] + np.random.rand()*(sigma1_range[1] - sigma1_range[0])
    sigma1 = np.exp(sigma1_uni)
    
    sigma2_uni =sigma2_range[0] + np.random.rand()*(sigma2_range[1] - sigma2_range[0])
    sigma2 = np.exp(sigma2_uni)
    
    #Draw number of samples to average over
    index = np.random.randint(0, len(number_of_draws))
    draws = number_of_draws[index]
    
    architechture = [dim, n1, n2, n3, n4]
    architechture = architechture[0:(nl+1)]
    
    #Build model
    architechture = [dim, n1, n2, n3, n4]
    architechture = architechture[0:(nl+1)]
    hyperparameter_dict = {}
    hyperparameter_dict['sigma1'] = sigma1
    hyperparameter_dict['sigma2'] = sigma2
    hyperparameter_dict['pi'] = pi
    hyperparameter_dict['draws'] = draws

    bnn = BNN(architechture, hyperparameter_dict=hyperparameter_dict)
    
    #Training
    optimizer = torch.optim.Adam(bnn.parameters(), lr=alpha) 
    
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=256, shuffle=True)
    no_minibatches = len(train_loader)
    
    losses_train=[]
    losses_val = []
    acc_train = []
    acc_val = []
    for epoch in range(number_of_epochs):
        loss_epoch = []
        for j, (X_l, y_l) in enumerate(train_loader):
            #Forward pass
            loss = bnn.loss(X_l, y_l.double(), no_minibatches=no_minibatches)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.item())
            
        losses_train.append(np.mean(loss_epoch))
        # Validation loss
        val_loss = bnn.loss(X_val_torch, y_val_torch, no_minibatches=1)
        losses_val.append(val_loss)
        
        #Training accuracy
        y_pred_prob = n_forward_passes(bnn, X_train_torch, no_forward_passes = 100, verbose=False)
        y_pred = (y_pred_prob > 0.5).astype(int)
        acc_t = np.mean(np.mean(y_pred.squeeze() == y_train.squeeze()))
        acc_train.append(acc_t)
        
        #Validation acuracy
        y_pred_prob = n_forward_passes(bnn, X_val_torch, no_forward_passes = 100, verbose=False)
        y_pred = (y_pred_prob > 0.5).astype(int)
        acc_v = np.mean(np.mean(y_pred.squeeze() == y_val.squeeze()))
        acc_val.append(acc_v)
        
        
    #Validation accuracy
    y_pred_prob = n_forward_passes(bnn, torch.from_numpy(X_val2), no_forward_passes = 1000, verbose=False)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    acc = np.mean(np.mean(y_pred.squeeze() == y_val2.squeeze())) #97.9%
    ECE = bin_acc(y_pred_prob, y_val2, M=20, ECE_calc=True)
    MCE = bin_acc(y_pred_prob, y_val2, M=20, MCE_calc=True)
        
    if acc > best_acc:
        torch.save(bnn, Result_dir+'best_acc'+str(par_number))
        best_acc = acc
        best_acc_i = i
        hist_df = pd.DataFrame({'epoch' : np.arange(1, number_of_epochs+1),\
                                 'Training_loss' : losses_train,\
                                 'Training_accuracy' : acc_train,\
                                 'Val_loss' : losses_val,\
                                 'Val_accuracy' : acc_val})
    
        with open(Result_dir+'history'+str(par_number)+'.csv', mode='w') as f:
            hist_df.to_csv(f)
        
    #Sequential implementation
    accuracies[i, 0] = acc
    ECEs[i, 0] = ECE
    MCEs[i, 0] = MCE
    hyperparameters[i, :] = np.array([batch_size, alpha, nl, n1, n2, n3, n4, pi, sigma1, sigma2, draws])

CSV = np.concatenate((hyperparameters, accuracies, ECEs, MCEs), axis=1)
np.savetxt(Result_dir+'RandomSearch'+str(par_number)+'.csv', CSV, delimiter=",", header="batch_size, alpha, nl, n1, n2, n3, n4, pi, sigma1, sigma2, draws, acc, ECE, MCE")

