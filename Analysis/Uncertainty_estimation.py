#This script performs uncertianty estimation in 3 parts:
#Part A: Accuracy and AUROC




# Part B: (Main part) Performance on test distribution
# Metrics: ECE, MCE, NLL, Brier score, uncertainty, variance between ensembles
# Plots: Reliability diagrams, histogram of predictions (total and wrong)

# Part C: Performance on OOD distribution
# Uncertainty on shift

#The models asses in all three parts are
#(1) Normal neural network
#(2) Bayesian Neural Network
#(3) MC-Dropout
#(4) Deep ensembles

'''
Loading models
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
import matplotlib

################
#Normal network#
################
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


#model = keras.models.load_model('/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Models/9bus/best_acc42_best.h5')
#W_list = model.weights
#architechture = [W_list[0].shape[0]]
#for i in range(2, len(W_list), 2):
#    architechture.append(W_list[i].shape[0])
##Save weights as numpy ndarrays
#for i in range(len(W_list)):
#    W_list[i] = W_list[i].numpy()
#
#net = Net(architechture)
#net.load_weights_w_bias(W_list)

#torch.save(net, '/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Models/9bus/net_99.3.pt')
net = torch.load('/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Models/9bus/net_99.3.pt')

#############################
#(2) Bayesian Neural Network#
#############################

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
    
bnn = torch.load('/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Models/9bus/bnn_8_notbest.pt')
#bnn = torch.load('/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Models/9bus/bnn_1.pt')


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

#
#model = keras.models.load_model('/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Models/9bus/MC_dropout_best_acc46.h5')
#W_list = model.weights
#for i in range(len(W_list)):
#    W_list[i] = W_list[i].numpy()
#
#
#
#hyperparameters = pd.read_csv('/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Models/9bus/MC_dropout_RandomSearch46.csv', delimiter=",")
#hyperparameters2 = hyperparameters.iloc[hyperparameters[[' accuracies']].idxmax()]
#hyperparameters3 = hyperparameters2.values.squeeze().tolist()
#batch_size, alpha, lr_decay, nl, n1, n2, n3, n4, do_p1, do_p2, do_p3, do_p4, _, _ = hyperparameters3
#hyperparameters_dict = {'batch_size' : batch_size, 'alpha' : alpha, 'lr_decay' : lr_decay, 'nl' : nl,\
#                        'n1' : n1, 'n2' : n2, 'n3' : n3, 'n4' : n4, 'do_p1' : do_p1, 'do_p2' : do_p2,\
#                        'do_p3' : do_p3, 'do_p4' : do_p4}
#
#hyperparameters_dict['dim'] = dim
#
#net_do = Net_dropout(hyperparameters_dict)
#net_do.load_weights_w_bias(W_list)

#torch.save(net_do, '/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Models/9bus/net_do_99.5.pt')
net_do = torch.load('/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Models/9bus/net_do_99.5.pt')

######################
#(4) Deep ensembles #
######################

#ensembles = {}
#for i in range(1, 10+1):
#    model = keras.models.load_model('/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Models/9bus/Ensembles/Keras/model'+str(i)+'.h5')
#    W_list = model.weights
#    architechture = [W_list[0].shape[0]]
#    for j in range(2, len(W_list), 2):
#        architechture.append(W_list[j].shape[0])
#    #Save weights as numpy ndarrays
#    for j in range(len(W_list)):
#        W_list[j] = W_list[j].numpy()
#    
#    ensembles[i] = Net(architechture)
#    ensembles[i].load_weights_w_bias(W_list)

#Saving the ensembles for Pytorch
#for i in range(1, 10+1):
#    torch.save(ensembles[i], '/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Models/9bus/Ensembles/net'+str(i)+'.pt')
ensembles = {}
for i in range(1, 10+1):
    ensembles[i] = torch.load('/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Models/9bus/Ensembles/net'+str(i)+'.pt')



'''
Loading data
'''
data_dir = '/Users/Eigil/Dropbox/DTU/Speciale/Data/9bus_30150.npz'
data_dir_val = '/Users/Eigil/Dropbox/DTU/Speciale/Data/Classification_val_50000.mat'
data_dir_test =  '/Users/Eigil/Dropbox/DTU/Speciale/Data/Classification_test_50000.mat'
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

#Torching the datasets
X_train_torch = torch.from_numpy(X_train)
y_train_torch = torch.from_numpy(y_train)
X_val_torch = torch.from_numpy(X_val)
y_val_torch = torch.from_numpy(y_val).double()
X_test_torch = torch.from_numpy(X_test)
y_test_torch = torch.from_numpy(y_test)
X_val2_torch = torch.from_numpy(X_val2)
y_val2_torch = torch.from_numpy(y_val2)
X_test2_torch = torch.from_numpy(X_test2)
y_test2_torch = torch.from_numpy(y_test2)

'''
Defining functions
'''
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

def ensemble_forward_pass_logit(ensemble_dict, X, no_ensembles=10, return_matrix=False):
    Y_pred_prob = np.zeros((X.shape[0], no_ensembles))
    for i, key in enumerate(list(ensemble_dict.keys())[0:no_ensembles]):
        network = ensemble_dict[key]
        y_pred_prob = network.forward(X)
        Y_pred_prob[:, i] = y_pred_prob.detach().numpy().squeeze()
    y_pred_prob_mean = np.mean(Y_pred_prob, axis=1)
    if return_matrix == True:
        return((y_pred_prob_mean, Y_pred_prob))
    else:
        return(y_pred_prob_mean)


def bin_acc(y_prob, y_true, M, ECE_calc = False, MCE_calc = False):
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



def binary_prob_to_confidence(y_prob):
    y_prob = y_prob.squeeze()
    y_prob = y_prob.reshape(-1, 1)
    p_vector = np.concatenate((y_prob, 1-y_prob), axis=1)
    confidence = np.max(p_vector, axis=1)
    return(confidence.squeeze())

def bin_acc2(y_prob, y_true, M, ECE_calc = False, MCE_calc = False):
    #Creates binned accuracies, but on confidence rather than P(y=1|x)
    y_pred = (y_prob > 0.5).astype(int)
    confidence = binary_prob_to_confidence(y_prob)
    n = len(y_prob)
    #Sorting the predictions into bins
    sorted_indices = np.argsort(confidence)
    confidence = confidence[sorted_indices]
    y_true = y_true[sorted_indices]
    y_pred = y_pred[sorted_indices]
    
    
    conf_bins = np.linspace(0, 1, M+1)
    acc = [] #List of accuracies for bins
    Bn = []  #List of number of observations in each bin
    conf = [] #Average confidence in each bin
    
    for i in range(1, M+1):
        index = (conf_bins[i] >= confidence) & (conf_bins[i-1] <= confidence)
        #Check if there are any observations in the range
        if np.any(index):
            Bn.append(np.sum(index))
            conf.append(np.mean(confidence[index])) #Average confidence
            acc.append(np.mean(y_pred[index].squeeze() == y_true[index].squeeze()))
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

def prob_to_logit(y_pred_prob):
    y_prob = y_pred_prob.copy()
    index_1 = (np.abs(y_pred_prob - 1) < 1e-12)
    index_0 = (np.abs(y_pred_prob - 0) < 1e-12)
    y_prob[index_1] = 0.5
    y_prob[index_0] = 0.5
    z = - np.log(1/y_prob - 1)
    z[index_1] = 100
    z[index_0] = -100
    return(z)

    

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

def unc_entropy(y_prob):
    y_prob2 = y_prob.copy()
    index_1 = np.abs(y_prob2 - 1) < 1e-60
    index_0 =  np.abs(y_prob2 - 0) < 1e-60
    y_prob2[index_1] = 0.5
    y_prob2[index_0] = 0.5
    uncertainty =  -y_prob2*np.log(y_prob2)-(1-y_prob2)*np.log(1-y_prob2)
    uncertainty[index_1] = 0
    uncertainty[index_0] = 0
    return(uncertainty)




'''
PART A: Accuracy and AUROC
'''

#Accuracy

#(1) Normal network
y_pred_prob_net = net.forward_P(X_test2_torch).detach().numpy().squeeze()
y_pred_net = (y_pred_prob_net > 0.5).astype(int)
acc_net = np.mean(y_pred_net.squeeze() == y_test2.squeeze())
y_pred_prob_net_val = net.forward_P(X_val2_torch).detach().numpy().squeeze()#Validation probability


#(2) Bayesian Neural Network
y_pred_prob_bnn, Y_pred_prob_bnn = n_forward_passes(bnn, X_test2_torch, no_forward_passes=1000, return_matrix=True)
y_pred_bnn = (y_pred_prob_bnn > 0.5).astype(int)
acc_bnn = np.mean(y_pred_bnn.squeeze() == y_test2.squeeze())
y_pred_prob_bnn_val = n_forward_passes(bnn, X_val2_torch, no_forward_passes=1000, return_matrix=False)
y_bnn_var = np.var(Y_pred_prob_bnn, axis=1)


#(3) MC-Dropout
y_pred_prob_netdo, Y_pred_prob_netdo = n_forward_passes(net_do, X_test2_torch, no_forward_passes=1000, verbose=False, return_matrix=True)
y_pred_netdo =  (y_pred_prob_netdo > 0.5).astype(int)
acc_netdo = np.mean(y_pred_netdo.squeeze() == y_test2.squeeze())
y_pred_prob_netdo_val = n_forward_passes(net_do, X_val2_torch, no_forward_passes=1000, verbose=False, return_matrix=False)
y_netdo_var = np.var(Y_pred_prob_netdo, axis=1)


#(4) Deep Ensembles
y_pred_prob_ens, Y_pred_prob_ens = ensemble_forward_pass(ensembles, X_test2_torch, return_matrix=True)
y_pred_ens =  (y_pred_prob_ens > 0.5).astype(int)
acc_ens = np.mean(y_pred_ens.squeeze() == y_test2.squeeze())
y_pred_prob_ens_val = ensemble_forward_pass(ensembles, X_val2_torch, return_matrix=False)
y_ens_var =  np.var(Y_pred_prob_ens, axis=1)


print('Accuracies:')
print(acc_net, acc_bnn, acc_netdo, acc_ens)


#prob_dict = {1: y_pred_net, 2: y_pred_prob_bnn, 3: y_pred_prob_netdo, 4: }
prob_list = [y_pred_prob_net, y_pred_prob_bnn, y_pred_prob_netdo, y_pred_prob_ens]
#AUROC
auc_list = []
plt.rcParams.update({'font.size': 14})
fig, ax = plt.subplots(figsize=(8, 6))

for i in range(len(prob_list)):
    y_pred_prob = prob_list[i]
    if y_pred_prob is None:
        continue
    fpr, tpr, thresholds = roc_curve(y_test2, y_pred_prob)
    plt.plot(fpr, tpr)
    auc_list.append(auc(fpr, tpr))
    
plt.xlabel('False Positive Rate (FPR)', fontsize=14)
plt.ylabel('True Positive Rate (TPR)', fontsize=14)
plt.legend(['NN', 'BNN', 'MC-Dropout', 'Deep Ensembles'])
plt.title('ROC plot for the 9-bus models')
plt.tight_layout()



print(auc_list)


print('Normal network')
print(confusion_matrix(y_test2, y_pred_net))
print('BNN')
print(confusion_matrix(y_test2, y_pred_bnn))
print('MC-Dropout')
print(confusion_matrix(y_test2, y_pred_netdo))
print('Deep Ensemble')
print(confusion_matrix(y_test2, y_pred_ens))



confusion_matrix(y_test2, y_pred_net).ravel()

'''
PART B: Performance on test distribution
'''

'''
Part B (I): Simple metrics
'''

#PLOTS
fig, ax = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=False, figsize=(11, 8))

M = 15
x = np.linspace(0, 1, M+1)
width = x[1] - x[0]
#Outer loop: Models
model_dict = {0 : 'NN', 1 : 'BNN', 2: 'MC-Dropout', 3: 'Deep Ensembles'}
for j in range(4):
    #Inner loop:
    y_pred_prob = prob_list[j]
    #First plot: Reliability diagram p(y=1|x)
    acc, Bn, conf = bin_acc(y_pred_prob, y_test2.squeeze(), M)
    ax[0, j].bar(x[:-1], acc, width=0.85*width, align='edge')
    ax[0, j].plot([0, 1], [0, 1], color='red')
    #Labels
    ax[0, 0].set_ylabel('Accuracy')
    ax[0, j].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    if j != 0: #If it isn't the firs tplot, remove labels
        ax[0, j].set_yticklabels(['']*6)
    ax[0, j].set_xlabel('p(y=1|x)')
    
    ax[0, j].set_title(model_dict[j])
    
    #Third plot: Histogram
    ax[2, j].bar(x[:-1], Bn/np.sum(Bn), width=0.85*width, align='edge')
    ax[2, j].set_yscale('log')
    ax[2, j].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax[2, j].set_xticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], rotation=-90)
    ax[2, 0].set_ylabel('Normalized count')
    if j != 0:
        ax[2, j].set_yticklabels([''])
    ax[2, j].set_xlabel('p(y=1|x)')
    
    #Second plot: Reliability diagram 
    acc, Bn, conf = bin_acc2(y_pred_prob, y_test2.squeeze(), M)
    ax[1, j].bar(x[:-1], acc, width=0.85*width, align='edge')
    ax[1, j].plot([0, 1], [0, 1], color='red')
    #Labels
    ax[1, 0].set_ylabel("Accuracy")
    ax[1, j].set_xlabel("Confidence $\hat{p}$")
    ax[1, j].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    if j != 0: #If it isn't the firs tplot, remove labels
        ax[0, j].set_yticklabels(['']*6)
        

plt.tight_layout()

#Platt temperature scaling
logit_net = prob_to_logit(y_pred_prob_net)
logit_bnn = prob_to_logit(y_pred_prob_bnn)
logit_netdo = prob_to_logit(y_pred_prob_netdo)
logit_ens = prob_to_logit(y_pred_prob_ens)


logit_net_val = prob_to_logit(y_pred_prob_net_val)
logit_bnn_val = prob_to_logit(y_pred_prob_bnn_val)
logit_netdo_val = prob_to_logit(y_pred_prob_netdo_val)
logit_ens_val = prob_to_logit(y_pred_prob_ens_val)

logit_list = [logit_net, logit_bnn, logit_netdo, logit_ens]
logit_val_list = [logit_net_val, logit_bnn_val, logit_netdo_val, logit_ens_val]


T_list = np.linspace(0.5, 2, num=200)

ECE_matrix = np.zeros((len(T_list), 4))
MCE_matrix = np.zeros((len(T_list), 4))
for j in range(4): 
    logit = torch.from_numpy(logit_val_list[j])
    for i, T in enumerate(T_list):
        y_pred_prob_T = torch.sigmoid(logit/T).detach().numpy()
        ECE = bin_acc(y_pred_prob_T, y_val2, M=15, ECE_calc=True)
        MCE = bin_acc(y_pred_prob_T, y_val2, M=15, MCE_calc=True)
        ECE_matrix[i, j] = ECE
        MCE_matrix[i, j] = MCE

T_ECE = T_list[np.argmin(ECE_matrix, axis=0)]
T_MCE = T_list[np.argmin(MCE_matrix, axis=0)]

#Plots
plt.figure(figsize=(12, 5))     
plt.subplot(1, 2, 1)   
for j in range(4):
    plt.plot(T_list, ECE_matrix[:, j])
plt.ylim([0, 0.005])
plt.legend(['NN', 'BNN', 'MC-Dropout', 'Ensemble'])
plt.xlabel('Temperature T')
plt.ylabel('ECE')
plt.title('Plot of validation ECE as function of T')

plt.subplot(1, 2, 2)    
for j in range(4):
    plt.plot(T_list, MCE_matrix[:, j])
#plt.ylim([0, 0.005])
plt.legend(['NN', 'BNN', 'MC-Dropout', 'Ensemble'])
plt.xlabel('Temperature T')
plt.ylabel('MCE')
plt.title('Plot of validation MCE as function of T')
plt.tight_layout()


y_pred_prob_net_ECE = torch.sigmoid(torch.from_numpy(logit_net/T_ECE[0])).detach().numpy()
y_pred_prob_bnn_ECE = torch.sigmoid(torch.from_numpy(logit_bnn/T_ECE[1])).detach().numpy()
y_pred_prob_netdo_ECE = torch.sigmoid(torch.from_numpy(logit_netdo/T_ECE[2])).detach().numpy()
y_pred_prob_ens_ECE = torch.sigmoid(torch.from_numpy(logit_ens/T_ECE[3])).detach().numpy()

y_pred_prob_net_MCE = torch.sigmoid(torch.from_numpy(logit_net/T_MCE[0])).detach().numpy()
y_pred_prob_bnn_MCE = torch.sigmoid(torch.from_numpy(logit_bnn/T_MCE[1])).detach().numpy()
y_pred_prob_netdo_MCE = torch.sigmoid(torch.from_numpy(logit_netdo/T_MCE[2])).detach().numpy()
y_pred_prob_ens_MCE = torch.sigmoid(torch.from_numpy(logit_ens/T_MCE[3])).detach().numpy()


#
prob_list_full = [y_pred_prob_net, y_pred_prob_bnn, y_pred_prob_netdo, y_pred_prob_ens,\
                  y_pred_prob_net_ECE, y_pred_prob_bnn_ECE, y_pred_prob_netdo_ECE, y_pred_prob_ens_ECE,\
                  y_pred_prob_net_MCE, y_pred_prob_bnn_MCE, y_pred_prob_netdo_MCE, y_pred_prob_ens_MCE]


prob_list_T_ECE = [y_pred_prob_net_ECE, y_pred_prob_bnn_ECE, y_pred_prob_netdo_ECE, y_pred_prob_ens_ECE]
prob_list_T_MCE = [y_pred_prob_net_MCE, y_pred_prob_bnn_MCE, y_pred_prob_netdo_MCE, y_pred_prob_ens_MCE]
#Plots for Platt scaling
fig, ax = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=False, figsize=(11, 8))
M = 15
x = np.linspace(0, 1, M+1)
width = x[1] - x[0]
prob_list_T_ECE = [y_pred_prob_net_ECE, y_pred_prob_bnn_ECE, y_pred_prob_netdo_ECE, y_pred_prob_ens_ECE]
for j in range(4):
    #Inner loop:
    y_pred_prob = prob_list_T_ECE[j]
    #First plot: Reliability diagram p(y=1|x)
    acc, Bn, conf = bin_acc(y_pred_prob, y_test2.squeeze(), M)
    ax[0, j].bar(x[:-1], acc, width=0.85*width, align='edge')
    ax[0, j].plot([0, 1], [0, 1], color='red')
    #Labels
    ax[0, 0].set_ylabel('Accuracy')
    ax[0, j].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    if j != 0: #If it isn't the firs tplot, remove labels
        ax[0, j].set_yticklabels(['']*6)
    ax[0, j].set_xlabel('p(y=1|x)')
    
    ax[0, j].set_title(model_dict[j])
    
    #Third plot: Histogram
    ax[2, j].bar(x[:-1], Bn/np.sum(Bn), width=0.85*width, align='edge')
    ax[2, j].set_yscale('log')
    ax[2, j].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax[2, j].set_xticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], rotation=-90)
    ax[2, 0].set_ylabel('Normalized count')
    if j != 0:
        ax[2, j].set_yticklabels([''])
    ax[2, j].set_xlabel('p(y=1|x)')
    
    #Second plot: Reliability diagram 
    acc, Bn, conf = bin_acc2(y_pred_prob, y_test2.squeeze(), M)
    ax[1, j].bar(x[:-1], acc, width=0.85*width, align='edge')
    ax[1, j].plot([0, 1], [0, 1], color='red')
    #Labels
    ax[1, 0].set_ylabel("Accuracy")
    ax[1, j].set_xlabel("Confidence $\hat{p}$")
    ax[1, j].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    if j != 0: #If it isn't the firs tplot, remove labels
        ax[0, j].set_yticklabels(['']*6)
st = fig.suptitle('Calibration plots, Temperature scaled for ECE')
fig.tight_layout()
st.set_y(0.95)
fig.subplots_adjust(top=0.85)
plt.show()  

fig, ax = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=False, figsize=(11, 8))
M = 15
x = np.linspace(0, 1, M+1)
width = x[1] - x[0]
prob_list_T_ECE = [y_pred_prob_net_ECE, y_pred_prob_bnn_ECE, y_pred_prob_netdo_ECE, y_pred_prob_ens_ECE]
for j in range(4):
    #Inner loop:
    y_pred_prob = prob_list_T_MCE[j]
    #First plot: Reliability diagram p(y=1|x)
    acc, Bn, conf = bin_acc(y_pred_prob, y_test2.squeeze(), M)
    ax[0, j].bar(x[:-1], acc, width=0.85*width, align='edge')
    ax[0, j].plot([0, 1], [0, 1], color='red')
    #Labels
    ax[0, 0].set_ylabel('Accuracy')
    ax[0, j].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    if j != 0: #If it isn't the firs tplot, remove labels
        ax[0, j].set_yticklabels(['']*6)
    ax[0, j].set_xlabel('p(y=1|x)')
    
    ax[0, j].set_title(model_dict[j])
    
    #Third plot: Histogram
    ax[2, j].bar(x[:-1], Bn/np.sum(Bn), width=0.85*width, align='edge')
    ax[2, j].set_yscale('log')
    ax[2, j].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax[2, j].set_xticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], rotation=-90)
    ax[2, 0].set_ylabel('Normalized count')
    if j != 0:
        ax[2, j].set_yticklabels([''])
    ax[2, j].set_xlabel('p(y=1|x)')
    
    #Second plot: Reliability diagram 
    acc, Bn, conf = bin_acc2(y_pred_prob, y_test2.squeeze(), M)
    ax[1, j].bar(x[:-1], acc, width=0.85*width, align='edge')
    ax[1, j].plot([0, 1], [0, 1], color='red')
    #Labels
    ax[1, 0].set_ylabel("Accuracy")
    ax[1, j].set_xlabel("Confidence $\hat{p}$")
    ax[1, j].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    if j != 0: #If it isn't the firs tplot, remove labels
        ax[0, j].set_yticklabels(['']*6)
st = fig.suptitle('Calibration plots, Temperature scaled for MCE')
fig.tight_layout()
st.set_y(0.95)
fig.subplots_adjust(top=0.85)
plt.show()  









#Metrics: Acc, ECE, MCE, Brier Score, NLL
metrics_names = ['ECE', 'MCE', 'NLL', 'Brier_score']
metrics = np.zeros((len(prob_list_full), len(metrics_names)))

for i in range(len(prob_list_full)):
    y_pred_prob = prob_list_full[i]
    ECE = bin_acc(y_pred_prob, y_test2, M=15, ECE_calc=True)
    MCE = bin_acc(y_pred_prob, y_test2, M=15, MCE_calc=True)
    BS = brier_score_loss(y_test2, y_pred_prob)
    NLL = nll(y_pred_prob, y_test2)
    metrics[i, :] = np.array([ECE, MCE, BS, NLL])

model_names = ['NN', 'BNN', 'MC-Dropout', 'Ensemble', 'NN_TE', 'BNN_TE', 'MC_Dropout_TE', 'Ensemble_TE',\
               'NN_ME', 'BNN_ME', 'MC_Dropout_ME', 'Ensemble_ME']

model_names_df = pd.DataFrame(model_names, columns=['Model'])
metrics_df = pd.DataFrame(metrics, columns=metrics_names)
metrics_df = pd.concat([model_names_df, metrics_df], axis=1)

#metrics_df.to_csv('/Users/Eigil/Dropbox/DTU/Speciale/Saved tables/Uncertainty_metrics_9bus.csv', sep=",")


'''
Part B (II): Quantification of uncertain estimations
'''

#Question: Can we remove the most uncertain points and achieve higher accuracy?
#In other words, is there correlation between uncertainty and whether or not the
#classifier predicts correctly?

cmap = plt.get_cmap("tab10")


unc_net = unc_entropy(y_pred_prob_net)
unc_bnn = unc_entropy(y_pred_prob_bnn)
unc_netdo = unc_entropy(y_pred_prob_netdo)
unc_ens = unc_entropy(y_pred_prob_ens)

unc_list = [unc_net, unc_bnn, unc_netdo, unc_ens]
var_list = [None, y_bnn_var, y_netdo_var, y_ens_var]
pred_list = [y_pred_net, y_pred_bnn, y_pred_netdo, y_pred_ens]


removal_vector_unc = []
removal_vector_var = []
#Calculation of accuracy improvement
for j in range(4):
    #First index: Uncertainty
    uncertainty = unc_list[j]
    y_var = var_list[j]
    y_pred = pred_list[j]
    
    index = np.argsort(-uncertainty) #From most uncertain to most certain
    correct_pred = (y_pred.squeeze()[index] == y_test2.squeeze()[index]).astype(int) #Prediction vector
    TP_vector = ((y_pred[index] == 1) & (y_test2.squeeze()[index] == 1)).astype(int) 
    FN_vector = ((y_pred[index] == 0) & (y_test2.squeeze()[index] == 1)).astype(int) 
    FP_vector = ((y_pred[index] == 1) & (y_test2.squeeze()[index] == 0)).astype(int) 
    TN_vector = ((y_pred[index] == 1) & (y_test2.squeeze()[index] == 1)).astype(int)
    
    acc_vec = []
    tpr_vec = []
    fpr_vec = []
    for i in range(len(correct_pred)):
        acc = np.mean(correct_pred[i:])
        tpr = np.sum(TP_vector[i:])/(np.sum(TP_vector[i:])+np.sum(FN_vector[i:]))
        fpr = np.sum(FP_vector[i:])/(np.sum(FP_vector[i:])+np.sum(TN_vector[i:]))
        acc_vec.append(acc)
        tpr_vec.append(tpr)
        fpr_vec.append(fpr)
    
    removal_vector_unc.append((acc_vec, tpr_vec, fpr_vec))
    
    #Variance
    if j == 0:
        removal_vector_var.append(None)
        continue
    
    index = np.argsort(-y_var) #From most uncertain to most certain
    correct_pred = (y_pred.squeeze()[index] == y_test2.squeeze()[index]).astype(int) #Prediction vector
    TP_vector = ((y_pred[index] == 1) & (y_test2.squeeze()[index] == 1)).astype(int) 
    FN_vector = ((y_pred[index] == 0) & (y_test2.squeeze()[index] == 1)).astype(int) 
    FP_vector = ((y_pred[index] == 1) & (y_test2.squeeze()[index] == 0)).astype(int) 
    TN_vector = ((y_pred[index] == 1) & (y_test2.squeeze()[index] == 1)).astype(int)
    
    acc_vec = []
    tpr_vec = []
    fpr_vec = []
    for i in range(len(correct_pred)):
        acc = np.mean(correct_pred[i:])
        tpr = np.sum(TP_vector[i:])/(np.sum(TP_vector[i:])+np.sum(FN_vector[i:])+1e-12)
        fpr = np.sum(FP_vector[i:])/(np.sum(FP_vector[i:])+np.sum(TN_vector[i:])+1e-12)
        acc_vec.append(acc)
        tpr_vec.append(tpr)
        fpr_vec.append(fpr)
    
    removal_vector_var.append((acc_vec, tpr_vec, fpr_vec))
        

#Plotting
no_points_plot = 5000
fraction_removed = np.arange(len(y_pred_net))/len(y_pred_net)
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 3.5))
for j in range(4):
    acc_vec, tpr_vec, fpr_vec = removal_vector_unc[j]
    ax[0].plot(fraction_removed[:no_points_plot], acc_vec[:no_points_plot], c=cmap(j))
    ax[1].plot(fraction_removed[:no_points_plot], tpr_vec[:no_points_plot], c=cmap(j))
    ax[2].plot(fraction_removed[:no_points_plot], fpr_vec[:no_points_plot], c=cmap(j))
    
    if j == 0:
        continue
    acc_vec, tpr_vec, fpr_vec = removal_vector_var[j]
    ax[0].plot(fraction_removed[:no_points_plot], acc_vec[:no_points_plot], linestyle='dashed', c=cmap(j))
    ax[1].plot(fraction_removed[:no_points_plot], tpr_vec[:no_points_plot], linestyle='dashed', c =cmap(j))
    ax[2].plot(fraction_removed[:no_points_plot], fpr_vec[:no_points_plot], linestyle='dashed', c=cmap(j))
    ax[0].set_ylim(ymax=1+1e-4)
    ax[1].set_ylim(ymax=1+1e-4)
    ax[2].set_ylim(ymax=0.008)
    
    ax[0].set_ylabel('Accuracy')
    ax[1].set_ylabel('TPR')
    ax[2].set_ylabel('FPR')
    ax[0].set_xlabel('Fraction removed')
    ax[1].set_xlabel('Fraction removed')
    ax[2].set_xlabel('Fraction removed')

plt.tight_layout()
ax[1].legend(labels=['NN', 'BNN (unc)', 'BNN (dis)', 'MC_DO (unc)', 'MC_DO(dis)', 'Ens (unc)', 'Ens (dis)'],\
  loc='upper center', bbox_to_anchor = (0.5, -0.3), fancybox=False, ncol=7, fontsize=12)

st = fig.suptitle('Improvement on metrics upon removal of uncertain predictions')
st.set_y(0.95)
fig.subplots_adjust(top=0.8)
plt.show()  



'''
PART C: Distribution shift
'''


#Creation of outliers dataset
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

X_outliers = np.repeat([[0.5, 0.5, 0.5, 0.5, 0.5, 1, 1, 1]], repeats=35, axis=0)
#
for i in range (5):
    mi, ma = maxmin_dict[i]
    X_outliers[i*7, i] = (1.1*ma - mi)/(ma - mi)
    X_outliers[i*7+1, i] = (1.3*ma - mi)/(ma - mi)
    X_outliers[i*7+2, i] = (1.5*ma - mi)/(ma - mi)
    X_outliers[i*7+3, i] = (2*ma - mi)/(ma - mi)
    X_outliers[i*7+4, i] = -(1-0.9)*mi/(ma - mi)
    X_outliers[i*7+5, i] = -(1-0.7)*mi/(ma - mi)
    X_outliers[i*7+6, i] = -(1-0.5)*mi/(ma - mi)

X_outliers = X_outliers - X_train_mean
X_outliers_torch = torch.from_numpy(X_outliers)



#Qualitative plots
y_prob_outliers_net = net.forward_P(X_outliers_torch).detach().numpy().squeeze()
y_prob_outliers_bnn, Y_prob_outliers_bnn = n_forward_passes(bnn, X_outliers_torch, return_matrix=True)
y_prob_outliers_netdo, Y_prob_outliers_netdo = n_forward_passes(net_do, X_outliers_torch, return_matrix=True)
y_prob_outliers_ens, Y_prob_outliers_ens = ensemble_forward_pass(ensembles, X_outliers_torch, no_ensembles=10, return_matrix=True)

unc_ol_net = unc_entropy(y_prob_outliers_net)
unc_ol_bnn =  unc_entropy(y_prob_outliers_bnn)
unc_ol_netdo =  unc_entropy(y_prob_outliers_netdo)
unc_ol_ens =  unc_entropy(y_prob_outliers_ens)

y_ol_bnn_var = np.var(Y_prob_outliers_bnn, axis=1)
y_ol_netdo_var = np.var(Y_prob_outliers_netdo, axis=1)
y_ol_ens_var = np.var(Y_prob_outliers_ens, axis=1)





plt.hist(y_bnn_var, bins=100, color='blue', density=True)
plt.hist(y_ol_bnn_var, bins=100, color='red', density=True)

plt.hist(unc_ens, bins=100, color='blue', density=True)
plt.hist(unc_ol_ens, bins=100, color='red', density=True)

plt.hist(unc_net, density=True, bins=100)

sns.distplot(unc_ens)

plt.ylim([0, 1])




pd.Series(unc_ens).plot.kde(0.1)
pd.Series(unc_ol_ens).plot.kde(0.1)


pd.Series(unc_net).plot.kde(0.1)
pd.Series(unc_ol_net).plot.kde(0.1)

pd.Series(unc_net).plot.kde(0.1)
pd.Series(unc_ol_net).plot.kde(0.1)


pd.Series(unc_bnn).plot.kde(0.1)
pd.Series(unc_ol_bnn).plot.kde(0.1)

pd.Series(unc_netdo).plot.kde(0.1)
pd.Series(unc_ol_netdo).plot.kde(0.1)


#Uncertainty as function of strength of shift (1 variable)
bins = np.linspace(0, 1, num=100)

hist_id, _ = np.histogram(unc_ens, bins=bins, density=True)
hist_ood, _ = np.histogram(unc_ol_ens, bins=bins, density=True)
hist_id[hist_id == 0] = 1e-12
hist_ood[hist_ood == 0] = 1e-12
print(scipy.stats.entropy(hist_id, hist_ood) + scipy.stats.entropy(hist_ood, hist_id))
print(scipy.spatial.distance.jensenshannon(hist_id, hist_ood))

hist_id, _ = np.histogram(unc_net, bins=bins, density=True)
hist_ood, _ = np.histogram(unc_ol_net, bins=bins, density=True)
hist_id[hist_id == 0] = 1e-12
hist_ood[hist_ood == 0] = 1e-12
print(scipy.stats.entropy(hist_id, hist_ood) + scipy.stats.entropy(hist_ood, hist_id))
print(scipy.spatial.distance.jensenshannon(hist_id, hist_ood))


hist_id, _ = np.histogram(unc_netdo, bins=bins, density=True)
hist_ood, _ = np.histogram(unc_ol_netdo, bins=bins, density=True)
hist_id[hist_id == 0] = 1e-12
hist_ood[hist_ood == 0] = 1e-12
print(scipy.stats.entropy(hist_id, hist_ood) + scipy.stats.entropy(hist_ood, hist_id))
print(scipy.spatial.distance.jensenshannon(hist_id, hist_ood))


hist_id, _ = np.histogram(unc_bnn, bins=bins, density=True)
hist_ood, _ = np.histogram(unc_ol_bnn, bins=bins, density=True)
hist_id[hist_id == 0] = 1e-12
hist_ood[hist_ood == 0] = 1e-12
print(scipy.stats.entropy(hist_id, hist_ood) + scipy.stats.entropy(hist_ood, hist_id))
print(scipy.spatial.distance.jensenshannon(hist_id, hist_ood))

#It works!


# Sampling out-of-distribution points by letting one variable work at a time

#Sampling between -2 and 3 for all variables
no_samples = 10000
X_outliers2 = np.random.uniform(low=-2, high=3, size=(no_samples, dim))
X_outliers2 = X_outliers2 - X_train_mean
X_outliers2_torch = torch.from_numpy(X_outliers2)

K = 5 #Number of neighbours to consider

dist_mat = scipy.spatial.distance_matrix(X_outliers2, X_train, p=1)
dist_mat_index = np.argpartition(dist_mat, K, axis=1)[:, :K]

dist = []
for i in range(len(X_outliers2)):
    indices = dist_mat_index[i, :]
    dist.append(np.mean(dist_mat[i, indices]))
dist = np.array(dist)

y_prob_ol2_net = net.forward_P(X_outliers2_torch).detach().numpy().squeeze()
y_prob_ol2_bnn, Y_prob_ol2_bnn = n_forward_passes(bnn, X_outliers2_torch, return_matrix=True)
y_prob_ol2_netdo, Y_prob_ol2_netdo = n_forward_passes(net_do, X_outliers2_torch, return_matrix=True)
y_prob_ol2_ens, Y_prob_ol2_ens = ensemble_forward_pass(ensembles, X_outliers2_torch, no_ensembles=10, return_matrix=True)

unc_ol2_net = unc_entropy(y_prob_ol2_net)
unc_ol2_bnn =  unc_entropy(y_prob_ol2_bnn)
unc_ol2_netdo =  unc_entropy(y_prob_ol2_netdo)
unc_ol2_ens =  unc_entropy(y_prob_ol2_ens)

y_ol2_bnn_var = np.var(Y_prob_ol2_bnn, axis=1)
y_ol2_netdo_var = np.var(Y_prob_ol2_netdo, axis=1)
y_ol2_ens_var = np.var(Y_prob_ol2_ens, axis=1)

plt.scatter(dist, unc_ol2_bnn, alpha=0.15)
plt.scatter(dist, unc_ol2_ens, alpha=0.15)
plt.scatter(dist, unc_ol2_net, alpha=0.15)
plt.scatter(dist, unc_ol2_netdo, alpha=0.15)


plt.scatter(dist, y_ol2_bnn_var, alpha=0.15)
plt.scatter(dist, y_ol2_netdo_var, alpha=0.15)
plt.scatter(dist, y_ol2_ens_var, alpha=0.15)



y_ol_bnn_var = np.var(Y_prob_outliers_bnn, axis=1)
y_ol_netdo_var = np.var(Y_prob_outliers_netdo, axis=1)
y_ol_ens_var = np.var(Y_prob_outliers_ens, axis=1)


#Out-of-distribution are now defined to have distance to 5 newest training ponts more than double
X_train_distmat = scipy.spatial.distance_matrix(X_train, X_train, p=1)


dist_mat_index = np.argpartition(X_train_distmat, K, axis=1)[:, :K]

dist_train = []
for i in range(len(X_train)):
    indices = dist_mat_index[i, :]
    dist_train.append(np.mean(X_train_distmat[i, indices]))
dist_train = np.array(dist_train)


#Mean of mean distance to 5 nearest neighbours
mean_dist = np.mean(dist_train)

#Max of mean distance to 5 nearest neighbours
max_dist = np.max(dist_train)

#Outliers are defined as having 2*max_dist
outlier_index = (dist >= 2*max_dist)
X_outliers3 = X_outliers2[outlier_index, :]
X_outliers3 = X_outliers2
X_outliers3_torch = torch.from_numpy(X_outliers3)

#Everything is outliers?
y_prob_ol3_net = net.forward_P(X_outliers3_torch).detach().numpy().squeeze()
y_prob_ol3_bnn, Y_prob_ol3_bnn = n_forward_passes(bnn, X_outliers3_torch, return_matrix=True)
y_prob_ol3_netdo, Y_prob_ol3_netdo = n_forward_passes(net_do, X_outliers3_torch, return_matrix=True)
y_prob_ol3_ens, Y_prob_ol3_ens = ensemble_forward_pass(ensembles, X_outliers3_torch, no_ensembles=10, return_matrix=True)

unc_ol3_net = unc_entropy(y_prob_ol3_net)
unc_ol3_bnn =  unc_entropy(y_prob_ol3_bnn)
unc_ol3_netdo =  unc_entropy(y_prob_ol3_netdo)
unc_ol3_ens =  unc_entropy(y_prob_ol3_ens)

y_ol3_bnn_var = np.var(Y_prob_ol3_bnn, axis=1)
y_ol3_netdo_var = np.var(Y_prob_ol3_netdo, axis=1)
y_ol3_ens_var = np.var(Y_prob_ol3_ens, axis=1)



unc_list_ood = [unc_ol3_net, unc_ol3_bnn, unc_ol3_netdo, unc_ol3_ens]

fraction_removed = np.arange(len(y_pred_net))/len(y_pred_net)

removed_ood_list_unc = []
for j in range(4):
    unc = unc_list[j]
    unc_ood = unc_list_ood[j]
    unc_sorted = np.sort(unc)[::-1] #From smallest to biggest
    unc_ood_sorted = np.sort(unc_ood)[::-1]
    removed_ood = np.zeros(len(unc_sorted))
    for i in range(len(unc_sorted)):
        threshold = unc_sorted[i]
        removed_ood[i] = np.mean(unc_ood_sorted > threshold)
    removed_ood_list_unc.append(removed_ood)







#We also try with variance
var_list_ood = [None, y_ol3_bnn_var, y_ol3_netdo_var, y_ol3_ens_var]
removed_ood_list_var = []

for j in range(4):
    if j == 0:
        removed_ood_list_var.append(None)
        continue
    var = var_list[j]
    var_ood = var_list_ood[j]
    var_sorted = np.sort(var)[::-1] #From smallest to biggest
    var_ood_sorted = np.sort(var_ood)[::-1]
    removed_ood = np.zeros(len(var_sorted))
    for i in range(len(unc_sorted)):
        threshold = var_sorted[i]
        removed_ood[i] = np.mean(var_ood_sorted > threshold)
    removed_ood_list_var.append(removed_ood)




fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
for j in range(4):
    ax[0].plot(fraction_removed, removed_ood_list_unc[j], c=cmap(j))
    if j == 0:
        continue
    else:
        ax[1].plot(fraction_removed, removed_ood_list_var[j], c=cmap(j))

ax[0].set_xlabel('Fraction of training data removed')
ax[1].set_xlabel('Fraction of training data removed')
ax[0].set_ylabel('Fraction of OOD samples removed')
ax[1].set_ylabel('Fraction of OOD samples removed')
#l1 = ax[0].legend(['NN', 'BNN', 'MC-Dropout', 'Ensemble'])
ax[0].set_title('Using entropy\n as OOD detector')
ax[1].set_title('Using disagreement\n as OOD detector')
fig.tight_layout()

fig.legend(handles=list(ax[0].get_lines()), labels=['NN', 'BNN', 'MC-Dropout', 'Ensemble'], loc="center right")
plt.subplots_adjust(right=0.8, wspace=0.3)

#fig.tight_layout()
plt.show()


#Figuring out what the thresholds are










