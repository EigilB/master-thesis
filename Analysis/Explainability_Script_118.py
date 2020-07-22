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

dim = 171
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
from matplotlib.lines import Line2D



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
            #y = self.sigmoid(self.output(a2))
            y = self.output(a2)
            activations = [x, a1, a2, y]
        elif self.nl == 3:
            a3 = self.relu3(self.n3(a2))
            #y = self.sigmoid(self.output(a3))
            y = self.output(a3)
            activations = [x, a1, a2, a3, y]
        elif self.nl == 4:
            a3 = self.relu3(self.n3(a2))
            a4 = self.relu4(self.n4(a3))
            #y = self.sigmoid(self.output(a4))
            y = self.output(a4)
            activations = [x, a1, a2, a3, a4, y]
        elif self.nl == 5:
            a3 = self.relu3(self.n3(a2))
            a4 = self.relu4(self.n4(a3))
            a5 = self.relu5(self.n5(a4))
            #y = self.sigmoid(self.output(a5))
            y = self.output(a5)
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
net = torch.load('/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Models/118bus/net.pt')
bnn= torch.load('/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Models/118bus/net_do.pt')
net_do = torch.load('/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Models/118bus/BNN_best_acc287.pt')



ensembles = {}
for i in range(1, 10+1):
    ensembles[i] = torch.load('/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Models/118bus/Ensembles/net'+str(i)+'.pt')



#LOADING DATA
data_dir = '/Users/Eigil/Dropbox/DTU/Speciale/Data/BigCombined1.mat'
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

#Torching the datasets
X_train_torch = torch.from_numpy(X_train)
y_train_torch = torch.from_numpy(y_train)
X_val_torch = torch.from_numpy(X_val)
y_val_torch = torch.from_numpy(y_val).double()
X_test_torch = torch.from_numpy(X_test)
y_test_torch = torch.from_numpy(y_test)


#Loading baselines
npzfile = np.load('/Users/Eigil/Dropbox/DTU/Speciale/Data/BigCombined1_baselines.npz')
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
explain_matrix = np.zeros((no_obs, dim, 11))
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
    explain_matrix[i, :, 0] = np.random.uniform(low=-1, high=1, size=dim)
    explain_matrix[i, :, 1] = grad(X1, net=net)
    explain_matrix[i, :, 2] = grad_X_inp(X1, net=net)
    explain_matrix[i, :, 3] = LRP_wo_bias(X1, net=net, full_relevance=False)
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
#np.save('/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Attribution_Maps/Attr_118bus_test.npy', explain_matrix)

#explain_matrix = np.load('/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Attribution_Maps/Attr_118bus_test.npy')
#np.save('/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Attribution_Maps/Attr_118bus_test2.npy', explain_matrix)


#Problems with LRP: Find NaN and replace them
explain_matrix[np.isnan(explain_matrix)] = 0


explain_matrix_mp = attr_most_predicted_class(explain_matrix, y_pred_prob_net)#Attributions for predicted class
explain_matrix_n = normalize_to_one(explain_matrix_mp)




'''
PART B: Qualitative assessment
'''

'''
Part B (I): Pertubation
'''
data_dir = '/Users/Eigil/Dropbox/DTU/Speciale/Data/InterestingPoints118.mat'
mat_content = sio.loadmat(data_dir)

X_ip0 = mat_content['X_not_secure']
y_ip0 = mat_content['y_not_secure']

X_ip1 = mat_content['X_secure']
y_ip1 = mat_content['y_secure']


    
X_ip0 = X_ip0 - X_train_mean
X_ip1 = X_ip1 - X_train_mean

net.forward_P(torch.from_numpy(X_ip0))
#net.forward_P(torch.from_numpy(X_ip1))

no_obs = len(X_ip0)
explain_matrix_ip = np.zeros((no_obs, dim, 11))
BL_dict_med = {0 :  torch.from_numpy(X_marg_med1.squeeze()), 1 : torch.from_numpy(X_marg_med0.squeeze())}
BL_dict_mode = {0 :  torch.from_numpy(X_mode1.squeeze()), 1 : torch.from_numpy(X_mode0.squeeze())}
BL_dict_m = {0 :  torch.from_numpy(X_m1.squeeze()), 1 : torch.from_numpy(X_m0.squeeze())}
X_pop_dict = {0 : X_train[y_train.squeeze() != 0, :], 1 : X_train[y_train.squeeze() != 1, :]}

DL1 = DeepLIFT(net)
DL2 = DeepLIFT(net)
DL3 = DeepLIFT(net)
for i in range(no_obs):
    y1 = y_ip0[i, 0]
    X1 = torch.from_numpy(X_ip0[i, :])
    X_pop = X_pop_dict[y1] #Opposite population of y1, for expgrad
    X_BL_med = BL_dict_med[y1]
    X_BL_mode = BL_dict_mode[y1]
    X_BL_m = BL_dict_m[y1]
    
    X_BL_med.requires_grad = False
    X_BL_mode.requires_grad = False
    X_BL_m.requires_grad = False
    explain_matrix_ip[i, :, 0] = np.random.uniform(low=-1, high=1, size=dim)
    explain_matrix_ip[i, :, 1] = grad(X1, net=net)
    explain_matrix_ip[i, :, 2] = grad_X_inp(X1, net=net)
    explain_matrix_ip[i, :, 3] = LRP_wo_bias(X1, net=net, full_relevance=False)
    explain_matrix_ip[i, :, 4] = total_expgrad_from_samples(method='expgrad_simple', X1=X1, X_train = X_pop, no_baselines=1000)
    explain_matrix_ip[i, :, 5] = intgrad(X1, X_BL_med, 200, net=net)
    explain_matrix_ip[i, :, 6] = intgrad(X1, X_BL_mode, 200, net=net)
    explain_matrix_ip[i, :, 7] = intgrad(X1, X_BL_m, 200, net=net)
    
    DL1.baseline(X_BL_med)
    attr1 = DL1.attributions(X1)
    DL2.baseline(X_BL_mode)
    attr2 = DL2.attributions(X1)
    DL3.baseline(X_BL_m)
    attr3 = DL3.attributions(X1)
    
    explain_matrix_ip[i, :, 8] = attr1.detach().numpy()
    explain_matrix_ip[i, :, 9] = attr2.detach().numpy()
    explain_matrix_ip[i, :, 10] = attr3.detach().numpy()

y_ip0_pred_prob = net.forward_P(torch.from_numpy(X_ip0))

explain_matrix_ip_mp = attr_most_predicted_class(explain_matrix_ip, y_ip0_pred_prob)
explain_matrix_ip_n = normalize_to_one(explain_matrix_ip_mp)



#First point

for j in range(11):
    print(method_dict[j])
    for i in [115, 116]:
        print(explain_matrix_ip_n[0, i, j])

for i in range(dim):
    print(i+1, explain_matrix_ip_n[0, i, 4])

ID_gens = np.array([115, 116]) #MATLAB
ID_demands = np.array([74, 76, 80, 81, 82, 83, 85, 86, 87, 88, 92])


fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(13, 13))
axes = np.array(axes).reshape(-1)
for i in range(11):
    axes[i].plot(np.arange(dim)+1, explain_matrix_ip_n[0, :, i])
    axes[i].plot(ID_gens, explain_matrix_ip_n[0, ID_gens-1, i], 'or')
    axes[i].plot(ID_demands, explain_matrix_ip_n[0, ID_demands-1, i], 'og')
    axes[i].set_title(method_dict[i])
    if i == 0:
        axes[i].set_ylim([-1, 1])
    axes[i].set_xlabel('Feature number')
    axes[i].set_ylabel('Attribution value')


legend_elements = [Line2D([0], [0], color="w", marker='o', markersize=14, markerfacecolor='r', label="Bus 100 and 103"),
                   Line2D([0], [0], color="w", marker='o', markersize=14, markerfacecolor='g', label="Demand busses connected\nto 100/103")]

axes[11].axis("off")
axes[11].legend(handles=legend_elements, loc='center', fontsize=14)
plt.tight_layout()
st = fig.suptitle('Attribution maps for pertubed point 1, 118-bus system')
st.set_y(0.95)
fig.subplots_adjust(top=0.9)


#Second point
ID_gens = np.array([115]) #MATLAB #(Not true?)
ID_demands = np.array([74, 76, 80, 81, 83, 85, 86, 88])

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(13, 13))
axes = np.array(axes).reshape(-1)
for i in range(11):
    axes[i].plot(np.arange(dim)+1, explain_matrix_ip_n[1, :, i])
    axes[i].plot(ID_gens, explain_matrix_ip_n[1, ID_gens-1, i], 'or')
    axes[i].plot(ID_demands, explain_matrix_ip_n[1, ID_demands-1, i], 'og')
    axes[i].set_title(method_dict[i])
    if i == 0:
        axes[i].set_ylim([-1, 1])
    axes[i].set_xlabel('Feature number')
    axes[i].set_ylabel('Attribution value')


legend_elements = [Line2D([0], [0], color="w", marker='o', markersize=14, markerfacecolor='r', label="Bus 100"),
                   Line2D([0], [0], color="w", marker='o', markersize=14, markerfacecolor='g', label="Demand busses\nconnected to 100")]

axes[11].axis("off")
axes[11].legend(handles=legend_elements, loc='center', fontsize=14)
plt.tight_layout()
st = fig.suptitle('Attribution maps for pertubed point 2, 118-bus system')
st.set_y(0.95)
fig.subplots_adjust(top=0.9)

#Third point

ID_volt_gen_69 = np.array([117+30])

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(13, 13))
axes = np.array(axes).reshape(-1)
for i in range(11):
    axes[i].plot(np.arange(dim)+1, explain_matrix_ip_n[2, :, i])
    axes[i].plot(ID_volt_gen_69 , explain_matrix_ip_n[2, ID_volt_gen_69 -1, i], 'or')
    axes[i].set_title(method_dict[i])
    if i == 0:
        axes[i].set_ylim([-1, 1])
    axes[i].set_xlabel('Feature number')
    axes[i].set_ylabel('Attribution value')


legend_elements = [Line2D([0], [0], color="w", marker='o', markersize=14, markerfacecolor='r', label="Voltage at bus 69")]

axes[11].axis("off")
axes[11].legend(handles=legend_elements, loc='center', fontsize=14)
plt.tight_layout()
st = fig.suptitle('Attribution maps for pertubed point 3, 118-bus system')
st.set_y(0.95)
fig.subplots_adjust(top=0.9)




'''
PART D: AOPC
'''


sorted_indices = np.argsort(-explain_matrix_n, axis=1)



def AOPC(sorted_indices2D, X_test=X_test, X_train=X_train, no_samples=10, net=net):
    if len(sorted_indices2D.shape) != 2:
        print('Error: Indices need to be 2D')
    n, dim = X_test.shape
    P0 = net.forward_P(torch.from_numpy(X_test)).detach().numpy()
    pred = (P0 > 0.5).astype(int).squeeze()
    prob_matrix = np.zeros((n, dim, no_samples))
    for i in range(no_samples):
        X_temp = X_test.copy()
        for j in range(dim):
            repl_fea = sorted_indices2D[:, j]
            out_index = np.random.randint(len(X_train), size=n)
            X_temp[:, repl_fea] = X_train[out_index, repl_fea]
            prob_matrix[:, j, i] = net.forward_P(torch.from_numpy(X_temp)).detach().numpy().squeeze()
    
    #Meaning fall over all runs
    prob_matrix_meaned = np.mean(prob_matrix, axis=2)
    AOPC_scores = dim*P0.squeeze() - np.sum(prob_matrix_meaned, axis=1)
    AOPC_scores[pred == 0] = - AOPC_scores[pred == 0]
    AOPC_scores = 1/dim*AOPC_scores
    return(AOPC_scores)
    
    
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
        
    

AOPC_matrix = np.zeros((len(X_test), 11))
for i in range(11):
    print('Trying out method: '+method_dict[i])
    AOPC_matrix[:, i] = AOPC_2(sorted_indices[:, :, i], X_train=X_train, X_test=X_test, no_samples=100, net=net)

AOPC_scores = np.mean(AOPC_matrix, axis=0)


fig, ax = plt.subplots()  
plt.barh(['Random', 'Grad', 'Grad X Input', 'LRP','ExpGrad', 'IntGrad (med)',\
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
# k = 1
k_list = [1, 2, 5, 10]
stoch_abl_mat = np.zeros((len(X_test), 11, len(k_list)))

for j in range(len(k_list)):
    k = k_list[j]
    print('Trying out k-value: '+str(k))
    for i in range(11):
        print('Trying out method: '+method_dict[i])
        stoch_abl_mat[:, i, j] = stoch_ablation_k(sorted_indices[:, :, i], X_test=X_test, X_train=X_train, no_samples=100, net=net, k=k)

stoch_abl_scores = np.mean(stoch_abl_mat, axis=0)



stoch_abl_scores_df = pd.DataFrame(stoch_abl_scores, columns=['k=1', 'k=2', 'k=5', 'k=10'])
stoch_abl_scores_df = pd.concat([pd.DataFrame({'Method' :  list(method_dict.values())}), stoch_abl_scores_df], axis=1)

stoch_abl_scores_df.round(3)


stoch_abl_scores_df.to_csv('/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Attribution_Maps/Stoch_ablation_118bus.csv', sep=",")



    