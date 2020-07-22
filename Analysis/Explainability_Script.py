####
#Explanability

# This script has several parts
# PART A: Loading models and calculating explanations for all test points
# PART B: Qualitative examples of explanations
# Part C: Global features
# PART D: Sanity checks
# Part E: Evaluation of methods (AOPC)
# Part F: Uncertainty of explanations
     

#####
'''
Imports and functions
'''

dim = 8
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import keras
import scipy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.metrics import accuracy_score
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_curve, auc, brier_score_loss
import scipy.spatial


#LOADING METHODS
exec(open("/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Explainability_Methods.py").read())



#FUNCTIONS
def n_forward_passes(net, X, no_forward_passes=100, verbose=False, return_matrix=False):
    Y_pred_prob = np.zeros((X.shape[0], no_forward_passes))
    for i in range(no_forward_passes):
        if verbose and i % 10 == 0:
            print('Forward pass number: '+str(i+1))
        y_pred_prob = net.forward_P(X)
        Y_pred_prob[:, i] = y_pred_prob.detach().numpy().squeeze()
    y_pred_prob_mean = np.mean(Y_pred_prob, axis=1)
    if return_matrix == True:
        return((y_pred_prob_mean, Y_pred_prob))
    else:
        return(y_pred_prob_mean)

def ensemble_forward_pass(ensemble_dict, X, no_ensembles=10, return_matrix=False):
    Y_pred_prob = np.zeros((X.shape[0], no_ensembles))
    for i, key in enumerate(list(ensemble_dict.keys())[0:no_ensembles]):
        network = ensemble_dict[key]
        y_pred_prob = network.forward_P(X)
        Y_pred_prob[:, i] = y_pred_prob.detach().numpy().squeeze()
    y_pred_prob_mean = np.mean(Y_pred_prob, axis=1)
    if return_matrix == True:
        return((y_pred_prob_mean, Y_pred_prob))
    else:
        return(y_pred_prob_mean)

def attr_most_predicted_class(explain_matrix, y_pred_prob):
    #Converts binary attribution maps for class 1 to attribution maps for
    #predicted class
    if type(y_pred_prob) == torch.Tensor:
        y_pred_prob = y_pred_prob.detach().numpy()
    y_pred = (y_pred_prob > 0.5).astype(int).squeeze()
    
    explain_matrix = explain_matrix.copy()
    #3D array
    if len(explain_matrix.shape) == 3:
        explain_matrix[y_pred == 0, :, :] = - explain_matrix[y_pred == 0, :, :] 
    elif len(explain_matrix.shape) == 2:
        explain_matrix[y_pred == 0, :] = - explain_matrix[y_pred == 0, :] 
    return(explain_matrix)

def normalize_to_one(explain_matrix):
    sums = np.sum(explain_matrix, axis=1, keepdims=True)
    normalized = np.divide(explain_matrix, sums, where=(np.abs(sums) > 1e-8))
    return(normalized)


def bus9_original_values(X, maxmin_dict, PD_dict, X_train_mean):
    X2 = X.copy()
    m1 = len(maxmin_dict.keys())
    m2 = len(PD_dict.keys())
    X2 = X2 + X_train_mean # Get original values again
    
    for i in range(m1):
        mi, ma = maxmin_dict[i]
        X2[:, i] = mi + X2[:, i]*(ma - mi)
    
    for j in range(m1, m1+m2):
        PD = PD_dict[j]
        X2[:, j] = X2[:, j]*PD
    return(X2)

#Creation of dicts




'''
Part A: Loading models and calculating explanations for all tet points
'''

#Classes
#Normal network
class Net(nn.Module):
    def __init__(self, architechture):
        super(Net, self).__init__()
        self.nl = len(architechture) - 1
        self.n1 = nn.Linear(architechture[0], architechture[1]).double()
        self.relu1 = nn.ReLU()
        self.n2 = nn.Linear(architechture[1], architechture[2]).double()
        self.relu2 = nn.ReLU()
        if self.nl > 2:
            self.n3 = nn.Linear(architechture[2], architechture[3]).double()
            self.relu3 = nn.ReLU()
        if self.nl > 3:
            self.n4 = nn.Linear(architechture[3], architechture[4]).double()
            self.relu4 = nn.ReLU()
        if self.nl > 4:
            self.n5 = nn.Linear(architechture[4], architechture[5]).double()
            self.relu4 = nn.ReLU()
        
        
        if self.nl == 2:
            self.output = nn.Linear(architechture[2], 1).double()
        elif self.nl == 3:
            self.output = nn.Linear(architechture[3], 1).double()
        elif self.nl == 4:
            self.output = nn.Linear(architechture[4], 1).double()
        elif self.nl == 5:
            self.output = nn.Linear(architechture[5], 1).double()
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x):
        x = self.n1(x)
        x = self.relu1(x)
        x = self.n2(x)
        x = self.relu2(x)
        if self.nl > 2:
            x = self.n3(x)
            x = self.relu3(x)
        if self.nl > 3:
            x = self.n4(x)
            x = self.relu4(x)
        if self.nl > 4:
            x = self.n5(x)
            x = self.relu5(x)
        x = self.output(x)
        return(x)
        
    def forward_P(self, x):
        x = self.forward(x)
        x = self.sigmoid(x)
        return(x)
        
    def load_weights_w_bias(self, W_list):
        #Loads weights from Keras. W_list is assumed to be a numpy array
        #Loop over all layers. W_list is assumed to be a numpy array
        i = 0
        for child in self.children():
            #Weights
            if type(child) == nn.Linear:
                child.weight.data = torch.from_numpy(W_list[2*i].T).double()
                child.bias.data = torch.from_numpy(W_list[2*i+1].T).double()
                i = i + 1
        
    def get_activations(self, x):
        a1 =  self.relu1(self.n1(x))
        a2 = self.relu2(self.n2(a1))
        if self.nl == 2:
            y = self.sigmoid(self.output(a2))
            activations = [x, a1, a2, y]
        elif self.nl == 3:
            a3 = self.relu3(self.n3(a2))
            y = self.sigmoid(self.output(a3))
            activations = [x, a1, a2, a3, y]
        elif self.nl == 4:
            a3 = self.relu3(self.n3(a2))
            a4 = self.relu4(self.n4(a3))
            y = self.sigmoid(self.output(a4))
            activations = [x, a1, a2, a3, a4, y]
        elif self.nl == 5:
            a3 = self.relu3(self.n3(a2))
            a4 = self.relu4(self.n4(a3))
            a5 = self.relu5(self.n5(a4))
            y = self.sigmoid(self.output(a5))
            activations = [x, a1, a2, a3, a4, a5, y]
        return(activations)
    

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



######################
#(3) Dropout network#
######################



class Net_dropout(nn.Module):
    def __init__(self, hyperparameters_dict):
        super(Net_dropout, self).__init__()
        #Loading architechture
        self.nl = np.array(hyperparameters_dict['nl']).astype(int)
        dim = np.array(hyperparameters_dict['dim']).astype(int)
        n1 = np.array(hyperparameters_dict['n1']).astype(int)
        n2 = np.array(hyperparameters_dict['n2']).astype(int)
        n3 = np.array(hyperparameters_dict['n3']).astype(int)
        n4 = np.array(hyperparameters_dict['n4']).astype(int)
        do_p1 = hyperparameters_dict['do_p1']
        do_p2 = hyperparameters_dict['do_p2']
        do_p3 = hyperparameters_dict['do_p3']
        do_p4 = hyperparameters_dict['do_p4']
        self.n1 = nn.Linear(dim, n1).double()
        self.relu1 = nn.ReLU()
        self.dropout1 =  nn.Dropout(p=do_p1)
        self.n2 = nn.Linear(n1, n2).double()
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=do_p2)
        if self.nl > 2:
            self.n3 = nn.Linear(n2, n3).double()
            self.relu3 = nn.ReLU()
            self.dropout3 = nn.Dropout(p=do_p3)
        if self.nl > 3:
            self.n4 = nn.Linear(n3, n4).double()
            self.relu4 = nn.ReLU()
            self.dropout4 = nn.Dropout(p=do_p4)

        #Defining output layer
        if self.nl == 2:
            self.output = nn.Linear(n2, 1).double()
        elif self.nl == 3:
            self.output = nn.Linear(n3, 1).double()
        elif self.nl == 4:
            self.output = nn.Linear(n4, 1).double()
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x):
        x = self.n1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.n2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        if self.nl > 2:
            x = self.n3(x)
            x = self.relu3(x)
            x = self.dropout3(x)
        if self.nl > 3:
            x = self.n4(x)
            x = self.relu4(x)
            x =self.dropout4(x)
        x = self.output(x)
        return(x)
        
    def forward_P(self, x):
        x = self.forward(x)
        x = self.sigmoid(x)
        return(x)
        
    def load_weights_w_bias(self, W_list):
        #Loads weights from Keras. W_list is assumed to be a numpy array
        #Loop over all layers. W_list is assumed to be a numpy array
        i = 0
        for child in self.children():
            #Weights
            if type(child) == nn.Linear:
                child.weight.data = torch.from_numpy(W_list[2*i].T).double()
                child.bias.data = torch.from_numpy(W_list[2*i+1].T).double()
                i = i + 1

#Loading models
net = torch.load('/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Models/9bus/net_99.3.pt')
bnn= torch.load('/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Models/9bus/bnn_8_notbest.pt')
net_do = torch.load('/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Models/9bus/net_do_99.5.pt')

ensembles = {}
for i in range(1, 10+1):
    ensembles[i] = torch.load('/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Models/9bus/Ensembles/net'+str(i)+'.pt')



#LOADING DATA
    
data_dir = '/Users/Eigil/Dropbox/DTU/Speciale/Data/9bus_30150.npz'
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

#Torching the datasets
X_train_torch = torch.from_numpy(X_train)
y_train_torch = torch.from_numpy(y_train)
X_val_torch = torch.from_numpy(X_val)
y_val_torch = torch.from_numpy(y_val).double()
X_test_torch = torch.from_numpy(X_test)
y_test_torch = torch.from_numpy(y_test)

#Creating original datasets
PG2_min = 10
PG2_max = 300
PG3_max = 270
PG3_min = 10
VG1_min = 0.9
VG2_min = 0.9
VG3_min = 0.9
VG1_max = 1.1
VG2_max = 1.1
VG3_max = 1.1
maxmin_dict = {0: (PG2_min, PG2_max), 1 : (PG3_min, PG3_max), 2 : (VG1_min, VG1_max), 3 : (VG2_min, VG2_max), 4: (VG3_min, VG3_max)} #Dictionary to map iteration to a variable
PD_dict = {5 : 90, 6 : 100, 7 : 125}
X_test_orig =bus9_original_values(X_test, maxmin_dict, PD_dict, X_train_mean)


#Creating dicts
method_dict = {0 : 'Random', 1 : 'Grad', 2 : 'Grad X Input', 3 : 'LRP', 4 : 'ExpGrad',\
               5 : 'IntGrad (med)', 6 : 'IntGrad (mode)', 7 : 'IntGrad (ext)', 8 : 'DeepLIFT (med)',
               9 : 'DeepLift (mode)', 10 : 'DeepLIFT (ext)'}

variable_dict = {0 : 'PG2', 1 : 'PG3', 2 : 'VG1', 3 : 'VG2', 4: 'VG3', 5 : 'PD5', 6 : 'PD7', 7 : 'PD9'}
classification_dict = {0 : 'True Positive', 1 : 'True Negative', 2 : 'False Posive', 3 : 'False Negative'}


#Loading baselines
npzfile = np.load('/Users/Eigil/Dropbox/DTU/Speciale/Data/9bus_baselines.npz')
X_mode0 = npzfile["X_mode0"]
X_mode1 = npzfile["X_mode1"]
X_marg_med0 = npzfile["X_marg_med0"]
X_marg_med1 = npzfile["X_marg_med1"]
X_m0 = npzfile["X_m0"]
X_m1 = npzfile["X_m1"]




#TESTING MODELS
y_pred_prob_net = net.forward_P(X_test_torch).detach().numpy().squeeze()
y_pred_net = (y_pred_prob_net > 0.5).astype(int)
acc_net = np.mean(y_pred_net.squeeze() == y_test.squeeze())
print('Accuracy of net: '+str(acc_net))

#(1) Bayesian Neural Network
y_pred_prob_bnn, Y_pred_prob_bnn = n_forward_passes(bnn, X_test_torch, no_forward_passes=1000, return_matrix=True)
y_pred_bnn = (y_pred_prob_bnn > 0.5).astype(int)
acc_bnn = np.mean(y_pred_bnn.squeeze() == y_test.squeeze())

#(3) MC-Dropout
y_pred_prob_netdo, Y_pred_prob_netdo = n_forward_passes(net_do, X_test_torch, no_forward_passes=1000, verbose=False, return_matrix=True)
y_pred_netdo =  (y_pred_prob_netdo > 0.5).astype(int)
acc_netdo = np.mean(y_pred_netdo.squeeze() == y_test.squeeze())

#(4) Deep Ensembles
y_pred_prob_ens, Y_pred_prob_ens = ensemble_forward_pass(ensembles, X_test_torch, return_matrix=True)
y_pred_ens =  (y_pred_prob_ens > 0.5).astype(int)
acc_ens = np.mean(y_pred_ens.squeeze() == y_test.squeeze())


#CALCULATING THE ATTRIBUTION MAPS
#Structure:
#For now, I don't consider SmoothGrad
#Random | Grad | Grax_x_input | LRP | ExpGrad | IG (median) | IG (mode) | IG (extreme) | DL (median) | DL (mode) | DL (extreme)
#In total: 11 attribution methods


no_obs = len(X_test)
explain_matrix = np.zeros((no_obs, 8, 11))
BL_dict_med = {0 :  torch.from_numpy(X_marg_med1.squeeze()), 1 : torch.from_numpy(X_marg_med0.squeeze())}
BL_dict_mode = {0 :  torch.from_numpy(X_mode1.squeeze()), 1 : torch.from_numpy(X_mode0.squeeze())}
BL_dict_m = {0 :  torch.from_numpy(X_m1.squeeze()), 1 : torch.from_numpy(X_m0.squeeze())}
X_pop_dict = {0 : X_train[y_train.squeeze() != 0, :], 1 : X_train[y_train.squeeze() != 1, :]}

DL1 = DeepLIFT(net)
DL2 = DeepLIFT(net)
DL3 = DeepLIFT(net)
for i in range(no_obs):
    y1 = y_test[i, 0]
    X1 = torch.from_numpy(X_test[i, :])
    X_pop = X_pop_dict[y1] #Opposite population of y1, for expgrad
    X_BL_med = BL_dict_med[y1]
    X_BL_mode = BL_dict_mode[y1]
    X_BL_m = BL_dict_m[y1]
    
    X_BL_med.requires_grad = False
    X_BL_mode.requires_grad = False
    X_BL_m.requires_grad = False
    explain_matrix[i, :, 0] = np.random.uniform(low=-1, high=1, size=8)
    explain_matrix[i, :, 1] = grad(X1, net=net)
    explain_matrix[i, :, 2] = grad_X_inp(X1, net=net)
    explain_matrix[i, :, 3] = LRP_wo_bias(X1, full_relevance=False)
    explain_matrix[i, :, 4] = total_expgrad_from_samples(method='expgrad_simple', X1=X1, X_train = X_pop, no_baselines=400)
    explain_matrix[i, :, 5] = intgrad(X1, X_BL_med, 200)
    explain_matrix[i, :, 6] = intgrad(X1, X_BL_mode, 200)
    explain_matrix[i, :, 7] = intgrad(X1, X_BL_m, 200)
    
    DL1.baseline(X_BL_med)
    attr1 = DL1.attributions(X1)
    DL2.baseline(X_BL_mode)
    attr2 = DL2.attributions(X1)
    DL3.baseline(X_BL_m)
    attr3 = DL3.attributions(X1)
    
    explain_matrix[i, :, 8] = attr1.detach().numpy()
    explain_matrix[i, :, 9] = attr2.detach().numpy()
    explain_matrix[i, :, 10] = attr3.detach().numpy()
    if i % 10 == 0:
        print("Observation number: "+str(i))
    

#Can we speed it up? Y
#np.save('/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Attribution_Maps/Attr_9bus_test.npy', explain_matrix)

explain_matrix = np.load('/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Attribution_Maps/Attr_9bus_test.npy')
#Problems with LRP: Find NaN and replace them
np.isnan(explain_matrix)

explain_matrix[np.isnan(explain_matrix)] = 0


#Most predicted class and normalizing the explain_matrix

explain_matrix_mp = attr_most_predicted_class(explain_matrix, y_pred_prob_net)#Attributions for predicted class
explain_matrix_n = normalize_to_one(explain_matrix_mp)





'''
PART B: Qualitative assessment
'''


'''
PART B (1): Simple pertubations
'''

#First: Predictions outside constraint feasibility hypercube
TP_index = (y_pred_net.squeeze() == 1) & (y_test.squeeze() == 1)
TN_index = (y_pred_net.squeeze() == 0) & (y_test.squeeze() == 0)
FP_index = (y_pred_net.squeeze() == 1) & (y_test.squeeze() == 0)
FN_index = (y_pred_net.squeeze() == 0) & (y_test.squeeze() == 1)

#Find safe points, that are being categorized as unsafe for all pertubations
for i in range(len(TP_index)):
    TP_indices = [i]
    no_pert_obs = len(TP_indices)
    no_pert = 4
    index = np.where(TP_index)[0][TP_indices]
    X_pert = X_test.copy()[index, :]
    X_pert = np.tile(X_pert, (no_pert, 1))
    for var in range(2):
        mi, ma = maxmin_dict[var]
        X_pert[2*var*no_pert_obs:(2*var+1)*no_pert_obs, var] = (1.1*ma - mi)/(ma-mi) - X_train_mean[var]
        X_pert[(2*var+1)*no_pert_obs:(2*var+2)*no_pert_obs, var] = -(1-0.5)*mi/(ma-mi) - X_train_mean[var]
    y_pert = (net.forward_P(torch.from_numpy(X_pert)).detach().numpy() > 0.5).astype(int)
    if np.sum(y_pert) == 0:
        print(TP_indices)



###Creating pertubed matrix
TP_indices = [0, 5, 13, 16]
no_pert_obs = len(TP_indices)
no_pert = 4
index = np.where(TP_index)[0][TP_indices]
X_pert = X_test[index, :]
X_pert = np.tile(X_pert, (no_pert, 1))

#Filling out the values
mi0, ma0 = maxmin_dict[0]
X_pert[0:no_pert_obs, 0] = (1.1*ma0 - mi0)/(ma0-mi0) - X_train_mean[0]
X_pert[no_pert_obs:(2*no_pert_obs), 0] = -(1-0.5)*mi0/(ma0-mi0) - X_train_mean[0]
mi1, ma1 = maxmin_dict[1]
X_pert[(2*no_pert_obs):(3*no_pert_obs), 1] = (1.1*ma1 - mi1)/(ma1-mi1) - X_train_mean[1]
X_pert[(3*no_pert_obs):, 1] = -(1-0.5)*mi1/(ma1-mi1) - X_train_mean[1]


y_pert = (net.forward_P(torch.from_numpy(X_pert)).detach().numpy() > 0.5).astype(int)
print(y_pert)





X_pert_orig = bus9_original_values(X_pert, maxmin_dict, PD_dict, X_train_mean)


no_obs = len(X_pert)
explain_matrix_pert = np.zeros((no_obs, 8, 11))
BL_dict_med = {0 :  torch.from_numpy(X_marg_med1.squeeze()), 1 : torch.from_numpy(X_marg_med0.squeeze())}
BL_dict_mode = {0 :  torch.from_numpy(X_mode1.squeeze()), 1 : torch.from_numpy(X_mode0.squeeze())}
BL_dict_m = {0 :  torch.from_numpy(X_m1.squeeze()), 1 : torch.from_numpy(X_m0.squeeze())}
X_pop_dict = {0 : X_train[y_train.squeeze() != 0, :], 1 : X_train[y_train.squeeze() != 1, :]}

DL1 = DeepLIFT(net)
DL2 = DeepLIFT(net)
DL3 = DeepLIFT(net)
for i in range(no_obs):
    y1 = y_pert[i, 0]
    X1 = torch.from_numpy(X_pert[i, :])
    X_pop = X_pop_dict[y1] #Opposite population of y1, for expgrad
    X_BL_med = BL_dict_med[y1]
    X_BL_mode = BL_dict_mode[y1]
    X_BL_m = BL_dict_m[y1]
    
    X_BL_med.requires_grad = False
    X_BL_mode.requires_grad = False
    X_BL_m.requires_grad = False
    explain_matrix_pert[i, :, 0] = np.random.uniform(low=-1, high=1, size=8)
    explain_matrix_pert[i, :, 1] = grad(X1, net=net)
    explain_matrix_pert[i, :, 2] = grad_X_inp(X1, net=net)
    explain_matrix_pert[i, :, 3] = LRP_wo_bias(X1, net=net, full_relevance=False)
    explain_matrix_pert[i, :, 4] = total_expgrad_from_samples(method='expgrad_simple', X1=X1, X_train = X_pop, no_baselines=400, net=net)
    explain_matrix_pert[i, :, 5] = intgrad(X1, X_BL_med, 200, net=net)
    explain_matrix_pert[i, :, 6] = intgrad(X1, X_BL_mode, 200, net=net)
    explain_matrix_pert[i, :, 7] = intgrad(X1, X_BL_m, 200, net=net)
    
    DL1.baseline(X_BL_med)
    attr1 = DL1.attributions(X1)
    DL2.baseline(X_BL_mode)
    attr2 = DL2.attributions(X1)
    DL3.baseline(X_BL_m)
    attr3 = DL3.attributions(X1)
    
    explain_matrix_pert[i, :, 8] = attr1.detach().numpy()
    explain_matrix_pert[i, :, 9] = attr2.detach().numpy()
    explain_matrix_pert[i, :, 10] = attr3.detach().numpy()


#Normalizing explanations
explain_matrix_pert_mp = attr_most_predicted_class(explain_matrix_pert, y_pert)#Attributions for predicted class
explain_matrix_pert_n = normalize_to_one(explain_matrix_pert_mp)


ex = explain_matrix_pert_n
indices = np.array(list(np.ndindex(ex.shape)))
ex_df = pd.DataFrame({'Value' : ex.flatten(),\
                      'Observation' : indices[:, 0], 'Feature' : indices[:, 1],\
                      'Method' :indices[:, 2]})
ex_df['Method'] = ex_df['Method'].replace(method_dict) #Replace strings
ex_df['Feature'] = ex_df['Feature'].replace(variable_dict)
ex_df = ex_df[ex_df['Method'] != "Random"] #Remove random as explanability method


grid_lines = (np.arange(dim)+0.5)[:-1]


for j in range(no_pert):
    print('Pertubation number: '+str(j+1))
    for i in range(no_pert_obs):
        iterator = no_pert_obs*j+i
        obs_number = index[i] #Number of pertubations per observation
        fig, ax = plt.subplots()
        df = ex_df[ex_df['Observation'] == iterator]
        g = sns.barplot(x="Feature", y="Value", hue="Method", data=df, ax=ax)
        g.legend_.remove()
        plt.ylim([-2, 2])
        plt.ylabel("Attribution Value")
        #Create xticklabels
        xticklabels = []
        X_orig = X_pert_orig[iterator, :]
        for i_l in range(len(X_orig)):
            feature_val = str(X_orig.squeeze()[i_l])
            feature_name = variable_dict[i_l]
            if feature_val[3] != '.':
                xticklabels.append(feature_name+'\n'+feature_val[0:4])
            else:
                xticklabels.append(feature_name+'\n'+feature_val[0:3])
        
        ax.set_xticklabels(xticklabels)
        for k in range(len(grid_lines)):
                ax.axvline(x=grid_lines[k], color='black', linewidth=1)
            
        ax.axhline(y=0, color='black', linewidth=1)
        plt.show()




'''
Part B (II): Actual predictions
'''
#Secondly: Actual predictions

#Indexing the explanations according to prediction type
ex_TP = explain_matrix_n[TP_index, :, :]
ex_TN = explain_matrix_n[TN_index, :, :]
ex_FP = explain_matrix_n[FP_index, :, :]
ex_FN = explain_matrix_n[FN_index, :, :]

X_test_orig_TP = X_test_orig[TP_index, :]
X_test_orig_TN = X_test_orig[TN_index, :]
X_test_orig_FP = X_test_orig[FP_index, :]
X_test_orig_FN = X_test_orig[FN_index, :]

X_test_orig_list = [X_test_orig_TP, X_test_orig_TN, X_test_orig_FP, X_test_orig_FN]



ex_list = [ex_TP, ex_TN, ex_FP, ex_FN]
ex_df_list = []
for j in range(4):
    ex = ex_list[j]
    indices = np.array(list(np.ndindex(ex.shape)))
    ex_df = pd.DataFrame({'Value' : ex.flatten(),\
                          'Observation' : indices[:, 0], 'Feature' : indices[:, 1],\
                          'Method' :indices[:, 2]})
    ex_df['Method'] = ex_df['Method'].replace(method_dict) #Replace strings
    ex_df['Feature'] = ex_df['Feature'].replace(variable_dict)
    ex_df = ex_df[ex_df['Method'] != "Random"] #Remove random as explanability method
    ex_df['Classification_type'] = classification_dict[j]
    ex_df_list.append(ex_df)
    

ex_TP_df, ex_TN_df, ex_FP_df, ex_FN_df = ex_df_list


#
TP_list = [2, 10]
TP_list_tup = []
for l in range(int(len(TP_list)/2)):
    TP_list_tup.append([TP_list[2*l], TP_list[2*l+1]])

TN_list = [0, 1]
TN_list_tup = []
for l in range(int(len(TN_list)/2)):
    TN_list_tup.append([TN_list[2*l], TN_list[2*l+1]])

FP_list = [0, 1]
FP_list_tup = []
for l in range(int(len(FP_list)/2)):
    FP_list_tup.append([FP_list[2*l], FP_list[2*l+1]])

FN_list = [0, 1]
FN_list_tup = []
for l in range(int(len(FN_list)/2)):
    FN_list_tup.append([FN_list[2*l], FN_list[2*l+1]])


list_tup_dict = {0 : TP_list_tup, 1 : TN_list_tup, 2 : FP_list_tup, 3 : FN_list_tup}

for i in range(4): #TP, TN
    list_tup = list_tup_dict[i]
    for indices in list_tup:
        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(12, 4))
        axes = axes.reshape(-1)
        for j, obs in enumerate(indices):
            
            X_orig = X_test_orig_list[i][obs, :]
            
            #Creation of labels
            xticklabels = []
            for i_l in range(len(X_orig)):
                feature_val = str(X_orig.squeeze()[i_l])
                feature_name = variable_dict[i_l]
                if feature_val[3] != '.':
                    xticklabels.append(feature_name+'\n'+feature_val[0:4])
                else:
                    xticklabels.append(feature_name+'\n'+feature_val[0:3])
            
            
            ex_df = ex_df_list[i]
            df = ex_df[ex_df['Observation'] == obs]
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x="Feature", y="Value", hue="Method", data=df, ax=axes[j])
            ax.legend_.remove()
            ax.set(xticklabels=xticklabels)
            ax.set_ylim([-2, 2])
            for k in range(len(grid_lines)):
                axes[j].axvline(x=grid_lines[k], color='black', linewidth=1)
            
            axes[j].axhline(y=0, color='black', linewidth=1)
            plt.sca(axes[j])
            axes[j].set_ylabel('Attribution value')
            axes[j].set_xticklabels(xticklabels)
        
        axes[-1].legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
        st = fig.suptitle(classification_dict[i] + ' attributions for predicted class (normalized)')
        fig.tight_layout()
        st.set_y(0.95)
        fig.subplots_adjust(top=0.85)
        plt.show()


'''
PART C: Evaluation of attribution maps
'''


sorted_indices = np.argsort(-explain_matrix_n, axis=1)


def AOPC_2(sorted_indices2D, X_test=X_test, X_train=X_train, no_samples=10, net=net):
    if len(sorted_indices2D.shape) != 2:
        print('Error: Indices need to be 2D')
    n, dim = X_test.shape
    P0 = net.forward_P(torch.from_numpy(X_test)).detach().numpy().squeeze()
    pred = (P0 > 0.5).astype(int).squeeze()
    prob_matrix = np.zeros((n, dim, no_samples))
    for i in range(no_samples):
        X_temp = X_test.copy()
        for j in range(dim):
            row_ind = np.arange(n)
            col_ind = sorted_indices2D[:, j].tolist()
            out_index = np.random.randint(len(X_train), size=n)
            X_temp[row_ind, col_ind] = X_train[out_index, col_ind]
            prob_matrix[:, j, i] = net.forward_P(torch.from_numpy(X_temp)).detach().numpy().squeeze()
    
    #Meaning fall over all runs
    prob_matrix_meaned = np.mean(prob_matrix, axis=2)
    AOPC_scores = dim*P0 - np.sum(prob_matrix_meaned, axis=1)
    AOPC_scores[pred == 0] = - AOPC_scores[pred == 0]
    AOPC_scores = 1/dim*AOPC_scores
    return(AOPC_scores)
    
    



def stoch_ablation_k(sorted_indices2D, X_test=X_test, X_train=X_train, no_samples=10, net=net, k=1):
    if len(sorted_indices2D.shape) != 2:
        print('Error: Indices need to be 2D')
    n, dim = X_test.shape
    P0 = net.forward_P(torch.from_numpy(X_test)).detach().numpy().squeeze()
    pred = (P0 > 0.5).astype(int).squeeze()
    diff_matrix = np.zeros((n, no_samples))
    for i in range(no_samples):
        X_temp = X_test.copy()
        repl_fea = sorted_indices2D[:, 0:k]
        out_index = np.random.randint(len(X_train), size=(n, k))
        # Create X_temp
        for j in range(k):
            row_ind = np.arange(n)
            col_ind = repl_fea[:, j].tolist()
            out_index = np.random.randint(len(X_train), size=n)
            X_temp[row_ind, col_ind] = X_train[out_index, col_ind]
        
        P1 = net.forward_P(torch.from_numpy(X_temp)).detach().numpy().squeeze()
        diff_matrix[:, i] = P0 - P1
    
    diff_matrix[pred == 0, :] = - diff_matrix[pred == 0, :]
    diff_mean = np.mean(diff_matrix, axis=1)
    return(diff_mean)
        
    

#Trying out AOPC matrix with fewer samples



AOPC_matrix = np.zeros((len(X_test), 11))
for i in range(11):
    print('Trying out method: '+method_dict[i])
    AOPC_matrix[:, i] = AOPC_2(sorted_indices[:, :, i], no_samples=1000)

AOPC_scores = np.mean(AOPC_matrix, axis=0)


fig, ax = plt.subplots()  
ax.barh(['Random', 'Grad', 'Grad X Input', 'LRP','ExpGrad', 'IntGrad (med)',\
          'IntGrad (mode)', 'IntGrad (ext)', 'DeepLIFT (med)',\
          'DeepLift (mode)','DeepLIFT (ext)'], AOPC_scores)
for i, v in enumerate(AOPC_scores):
    ax.text(v + 0.01, i + .25, str(v)[0:5])
plt.xlim([0, 0.6])
plt.xlabel('AOPC value')
plt.ylabel('Attribution Map method')
plt.gca().invert_yaxis()
plt.show()


#Stochastic ablations
k_list = [1, 2, 3]
stoch_abl_mat = np.zeros((len(X_test), 11, len(k_list)))

for j in range(len(k_list)):
    k = k_list[j]
    print('Trying out k-value: '+str(k))
    for i in range(11):
        print('Trying out method: '+method_dict[i])
        stoch_abl_mat[:, i, j] = stoch_ablation_k(sorted_indices[:, :, i], X_test=X_test, X_train=X_train, no_samples=100, net=net, k=k)

stoch_abl_scores = np.mean(stoch_abl_mat, axis=0)
stoch_abl_scores_df = pd.DataFrame(stoch_abl_scores, columns=['k=1', 'k=2', 'k=3'])
stoch_abl_scores_df = pd.concat([pd.DataFrame({'Method' :  list(method_dict.values())}), stoch_abl_scores_df], axis=1)

stoch_abl_scores_df.round(3)


'''
PART D: Global phenomena
'''
#Most 'telling' for security/insecurit
index_secure = np.where(y_test.squeeze() == 1)[0]
index_insecure = np.where(y_test.squeeze() == 0)[0]


#Overall: Most important features for the 11 methods
explain_matrix_a = np.abs(explain_matrix)
most_occ = np.argmax(explain_matrix_a, axis=1)
np.unique(most_occ, axis=0, return_counts=True)


feature_occ = np.zeros((dim, 11))
for feature_index in range(dim):
    feature_occ[feature_index, :] = np.mean(most_occ == feature_index, axis=0)

print('Grad occurences: ' + str(np.around(feature_occ[:, 1]*100, decimals=2)))
print('Grad_X_input occurences: ' + str(np.around(feature_occ[:, 2]*100, decimals=2)))
print('LRP occurences: ' + str(np.around(feature_occ[:, 3]*100, decimals=2)))
print('ExpGrad occurences: ' + str(np.around(feature_occ[:, 4]*100, decimals=2)))
print('IntGrad occurences: ' + str(np.around(feature_occ[:, 5]*100, decimals=2)))
print('IntGrad occurences (mo: ' + str(np.around(feature_occ[:, 6]*100, decimals=2)))
print('IntGrad occurences (extreme): ' + str(np.around(feature_occ[:, 7]*100, decimals=2)))
print('DeepLIFT occurences (med): ' + str(np.around(feature_occ[:, 5]*100, decimals=2)))
print('DeepLIFT occurences (mode): ' + str(np.around(feature_occ[:, 6]*100, decimals=2)))
print('DeepLIFT occurences (extreme): ' + str(np.around(feature_occ[:, 7]*100, decimals=2)))



feature_occ_df = pd.DataFrame(feature_occ.T, columns=list(variable_dict.values()))
feature_occ_df = pd.concat([pd.DataFrame({'Method' : list(method_dict.values())}), feature_occ_df], axis=1)



#What is the most telling for secure classes?
fig, axes = plt.subplots(ncols=dim, nrows=11, figsize=(8.27, 11.69), dpi=100, sharex = True, sharey = True)
for i in range(11):
    for j in range(dim):
        axes[i, j].scatter(X_test[:, j], explain_matrix_n[:, j, i], c=(y_test.squeeze() == 1), alpha=0.1, s=2)
        axes[i, j].set_ylim([-1, 1])

method_dict_plot = {0 : 'Random', 1 : 'Grad', 2 : 'Grad X Input', 3 : 'LRP', 4 : 'ExpGrad',\
               5 : 'IntGrad\n(median)', 6 : 'IntGrad\n(mode)', 7 : 'IntGrad\n(extreme)', 8 : 'DeepLIFT\n (median)',
               9 : 'DeepLIFT\n(mode)', 10 : 'DeepLIFT\n(extreme)'}
for i in range(11):
    axes[i, 0].set_ylabel(method_dict_plot[i])
    
for j in range(dim):
    axes[11-1, j].set_xlabel(variable_dict[j])
#
fig.add_subplot(111, frame_on=False)
plt.tick_params(labelcolor="none", bottom=False, left=False)

plt.xlabel("Value of feature", labelpad=30)
plt.ylabel("Normalized attribution for predicted class", labelpad=30)


