##Explainability utilities

import scipy.io as sio
import os
import sklearn
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
from keras import optimizers
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import decomposition
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr #Spearmancorrelation
from functools import partial #For DeepLIFT
#import Explainability_Methods
import seaborn as sns
from itertools import compress #For indexing a list with another list

#Loading in the network (for nets)
class Net(nn.Module):
    def __init__(self, architechture):
        super(Net, self).__init__()
        self.n1 = nn.Linear(architechture[0], architechture[1]).double()
        self.relu1 = nn.ReLU()
        self.n2 = nn.Linear(architechture[1], architechture[2]).double()
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(architechture[2], 1).double()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.n1(x)
        x = self.relu1(x)
        x = self.n2(x)
        x = self.relu2(x)
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
        a1 = self.relu1(self.n1(x))
#        a1 =  F.relu(self.n1(x))
        a1.retain_grad()
        a2 = self.relu2(self.n2(a1))
#        a2 = F.relu(self.n2(a1))
        a2.retain_grad()
#        a3 = self.output(a2)
        a3 = self.sigmoid(self.output(a2))
        a3.retain_grad()
        return([x, a1, a2, a3])


net = Net(architechture)
net.load_weights_w_bias(W_list)





'''
Gradient
'''
def grad(X1, net=net):
    #Computes gradient of probability
    X_temp = X1.clone()
    X_temp.requires_grad = True
    net.zero_grad()
    P = net.forward_P(X_temp)
    P.backward()
    return(X_temp.grad)

'''
Gradient X input
'''

def grad_X_inp(X1, net=net):
    X_grad = grad(X1, net=net)
    return(X_grad*X1)



'''
LRP

'''
def LRP_wo_bias(X1, net=net, full_relevance=False):
    
    
    
    A = net.get_activations(X1) #Calculate activations
    layers = [l for l in list(net.children()) if type(l) == nn.Linear]
    L = len(layers)
    #My own implementation 2
    R = [None]*L + [(A[-1]).data]
    for l in range(L-1, -1, -1):
        w = list(layers[l].parameters())[0].data #Weights
#        output_dim = w.shape[0]
#        input_dim = w.shape[1]
        
        b = list(layers[l].parameters())[1].data #Bias
        x = (A[l]).data #Activations in input layer
    #    Xij = torch.repeat_interleave(x.unsqueeze(0), output_dim, dim=0)
        Zij = torch.mul(w, x)
        Zij = Zij.T #Transpose to reflect paper: R^[From; To]
        Zj = torch.sum(Zij, dim=0)
        epsilon = torch.sign(Zj) * 1e-12
        Rij = Zij / (Zj.unsqueeze(0)+epsilon.unsqueeze(0)) * (R[l+1]).unsqueeze(0)
        R[l] = torch.sum(Rij, dim=1)
    if full_relevance:
        return(R)
    else:
        return(R[0])
        
        
'''
IntGrad
'''

def intgrad(X1, X_baseline, m, return_probs = False, net=net):
    #Arguments:
    #X: Variable to be explained
    #X_baseline: Variable to explain from (path integral)
    #m: Number of steps
    #Return_probs: Whether the function call should return the calculated probabilities a
    # along the path integral
    #net: Which network to use
    X_sum = 0
    probs = [] #Placeholder if the probs along the path integral are kept
    for k in range(1, m+1):
        X_temp = X_baseline + k/m*(X1 - X_baseline)
        X_temp = X_temp.data
        X_temp.requires_grad = True
        
        net.zero_grad()
        P = net.forward_P(X_temp)
        if return_probs == True:
            probs.append(P)
        P.backward(torch.ones(P.shape))
        
        X_sum = X_sum + X_temp.grad 
    intgrads = (X1 - X_baseline)/m*X_sum
    if return_probs == False:
        return(intgrads)
    else:
        return((intgrads, probs))
        
def intgrad_bnn(X1, X_baseline, m, return_probs = False, net=net):
    #Arguments:
    #X: Variable to be explained
    #X_baseline: Variable to explain from (path integral)
    #m: Number of steps
    #Return_probs: Whether the function call should return the calculated probabilities a
    # along the path integral
    #net: Which network to use
    X_sum = 0
    probs = [] #Placeholder if the probs along the path integral are kept
    #First iteration: Sample weights
    k = 1
    X_temp = (X_baseline + k/m*(X1 - X_baseline)).data
    X_temp.requires_grad = True
    net.zero_grad()
    P = net.forward_P(X_temp, freeze=False)
    if return_probs == True:
        probs.append(P)
    P.backward(torch.ones(P.shape))
    X_sum = X_sum + X_temp.grad
    #All remaining iterations: Keep weights
    for k in range(2, m+1):
        X_temp = X_baseline + k/m*(X1 - X_baseline)
        X_temp = X_temp.data
        X_temp.requires_grad = True
        net.zero_grad()
        P = net.forward_P(X_temp, freeze=True)
        if return_probs == True:
            probs.append(P)
        P.backward(torch.ones(P.shape))
        
        X_sum = X_sum + X_temp.grad 
    intgrads = (X1 - X_baseline)/m*X_sum
    if return_probs == False:
        return(intgrads)
    else:
        return((intgrads, probs))




'''
Expected gradients
'''

def expgrad_simple_samples(X1, X_train, no_baselines, net=net):
    #X: Observation to be explained
    #no_baselines: Number of samples draw
    samples = []
    N_train = len(X_train)
    indices = np.random.choice(a=N_train, size=no_baselines, replace=True)
    
    for k in range(no_baselines):
        index = indices[k]
        X_baseline = torch.from_numpy(X_train[index, :])
        alpha = np.random.uniform(low=0, high=1.0)
        
        #Create temporarily input
        X_temp = X_baseline + alpha*(X1-X_baseline)
        X_temp = X_temp.data
        X_temp.requires_grad = True
        net.zero_grad()
        
        P = net.forward_P(X_temp)
        P.backward(torch.ones(P.shape))
        samples.append((X1 - X_baseline)*X_temp.grad)
    return(samples)
    
    
def expgrad_comp_samples(X1, X_train, m, no_baselines, net=net):
    #X: Observation to be explained
    #m: Number of gradient evaluations at each baseline
    #no_baselines: Number of different baselines drawn from the training set
    samples = []
    indices = np.random.choice(a=X_train.shape[0], size = no_baselines, replace=False)
    for k in range(no_baselines):
        index = indices[k]
        X_baseline = torch.from_numpy(X_train[index, :])
        samples.append(intgrad(X1=X1, X_baseline=X_baseline, m=m))
    return(samples)
    
def total_expgrad_from_samples(method='samples', samples=None, X1=None, X_train=None, no_baselines=None, m=None, net=net):
    #How to calculate samples: (1) They are given, (2) Simple method, (3) Exhaustive method
    if method == 'samples':
        samples = samples
    elif method == 'expgrad_simple':
        samples = expgrad_simple_samples(X1, X_train, no_baselines, net=net)
    elif method == 'expgrad_comp':
        samples = expgrad_comp_samples(X1, X_train, m, no_baselines, net=net)
        
    mean_value = torch.stack(samples, dim=0).mean(dim=0)
    return(mean_value)
    
    
'''
DeepLIFT
'''

class DeepLIFT():
    def __init__(self, net):
        self.net = net
        self.activations_BL = {}
        self.activations = {}
        self.handle_list = []
        self.baselineflag = None
        self.baseline_tensor = None
        
    def save_forward_BL(self, name, module, input, output):
        #Function for saving baseline activations
        self.activations_BL[name] = (input[0], output)
        
    def forward_deeplift(self, name, module, input, output):
        #Function for saving activations for a given observations
        self.activations[name] = (input[0], output)
    
    def remove_all_handles(self, handle_list):
        for handle in handle_list:
            handle.remove()
    
    def baseline(self, baseline):
        baseline.requires_grad = True
        self.baseline_tensor = baseline.clone()        
        
        self.remove_all_handles(self.handle_list)
        for name, m in self.net.named_modules():
            if type(m) == nn.Sigmoid or type(m) == nn.Tanh or type(m) == nn.ReLU:
                handle = m.register_forward_hook(partial(self.save_forward_BL, name))
                self.handle_list.append(handle)
        
        self.net.forward_P(baseline)
        self.remove_all_handles(self.handle_list)
        self.baselineflag = True
        
    
    def activations_fill(self, X1):
        X1.requires_grad = True
        
        self.remove_all_handles(self.handle_list)
        for name, m in self.net.named_modules():
            if type(m) == nn.Sigmoid or type(m) == nn.Tanh or type(m) == nn.ReLU: 
                handle = m.register_forward_hook(partial(self.forward_deeplift, name))
                self.handle_list.append(handle)
        
        self.net.forward_P(X1)
        self.remove_all_handles(self.handle_list)
    
    def backward_deeplift(self, activation, activation_BL, module, grad_input, grad_output):
        print("", end=" ")
        input, output = activation
        input_BL, output_BL = activation_BL
        index = (input - input_BL).abs() > 1e-10 #For values close to ref, the normal grad_input is used
        
        new_grad_input = grad_input[0].clone()
        new_grad_input[index] = ((output[index] - output_BL[index])/(input[index] - input_BL[index]))
        new_grad_input[index] = new_grad_input[index] * grad_output[0][index]
#        forward_grad = (output-output_BL)/(input-input_BL)  
        return((new_grad_input, ))
        
    def attributions(self, X1):
        if self.baselineflag != True:
            raise Exception('Provide a baseline before calculating attributions')
        
        X1.requires_grad = True
        self.activations_fill(X1)
        
        for name, m in self.net.named_modules():
            if type(m) == nn.Sigmoid or type(m) == nn.Tanh or type(m) == nn.ReLU:
                activation = self.activations[name]
                activation_BL = self.activations_BL[name]
                handle = m.register_backward_hook(partial(self.backward_deeplift, activation, activation_BL))
                self.handle_list.append(handle)
        out = self.net.forward_P(X1)
        out.backward()
        self.remove_all_handles(self.handle_list)
        attr = X1.grad
        attr = (X1 - self.baseline_tensor)*attr
        return(attr)

    
'''
SmoothGrad
'''


def smoothgrad(method, X1, sigma, n=50, m=10, X_pop=None):
    #Sigma: Sigma in each direction
    #m: Number of steps
    pert = np.random.multivariate_normal(mean=np.zeros(8), cov=np.diag(np.square(sigma)), size=n)
    X_sampled = X1 + pert
    
    if method == 'intgrad':
        explain = intgrad(X1=X_sampled, X_baseline=X_baseline, m=m)
    elif method =='expgrad':
        explain = total_expgrad_from_samples(method='expgrad_simple', X1=X_sampled, X_train = X_pop, no_baselines=m)
    elif method == 'lrp':
        explain = []
        for i in range(n):
            explain.append(LRP_wo_bias(X_sampled[i, :], full_relevance=False))
        explain = torch.stack(explain, dim=0)
    else: #Errorneous method
        print('Error: Method has to be either inttgrad, expgrad or lrp')
        return(None)
    
    explain_meaned = explain.mean(dim=0).data
    return(explain_meaned)
