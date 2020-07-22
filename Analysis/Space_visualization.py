#We might need to retrain, as observations have been discarded randomly.
#I create a new dataset, that only includes the actual data
#mat_content = sio.loadmat('/Users/Eigil/Dropbox/DTU/Speciale/Data Generation/Classification4_50000.mat')
#X = mat_content['X']
#y = mat_content['y'].ravel()
#counts = np.unique(y, return_counts=True)
#n = np.min(counts[1])
#np.random.seed(0)
#mask = np.hstack([np.random.choice(np.where(y == l)[0], n, replace=False)
#                      for l in np.unique(y)])
#    
#print('Number of discarded observations: '+str(len(y) - len(mask)))
#X = X[mask]
#y = y[mask]
#y = np.reshape(y, (-1, 1))
#np.savez('/Users/Eigil/Dropbox/DTU/Speciale/Data/9bus_30150.npz', X=X, y=y)

from sklearn.neighbors import KernelDensity
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

datadir_9 = '/Users/Eigil/Dropbox/DTU/Speciale/Data/9bus_30150.npz'
datadir_118 = '/Users/Eigil/Dropbox/DTU/Speciale/Data/BigCombined1.mat'

npzfile = np.load('/Users/Eigil/Dropbox/DTU/Speciale/Data/9bus_30150.npz')
X = npzfile['X']
y = npzfile['y']



'''
4 bus system
'''


'''
9 bus system
'''
#Load data
npzfile = np.load(datadir_9)
X = npzfile['X']
y = npzfile['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.14, stratify = y, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify = y_test, random_state=0)
#Normalising the data
X_train_mean = np.mean(X_train, axis=0)
X_train = X_train - X_train_mean
X_test = X_test - X_train_mean
X_val = X_val - X_train_mean


#Class-conditional marginal distributions
variable_dict = {'0' : 'PG2', '1' : 'PG3', '2' : 'VG1', '3' : 'VG2', '4': 'VG3', '5' : 'PD5', '6' : 'PD7', '7' : 'PD9'}
fig = plt.figure(figsize=(10, 10))
for i in range(8):
    plt.subplot(3, 3, i+1)
    ax1 = sns.kdeplot(X_train[y_train.squeeze() == 1, i], shade=True, legend=False)
    ax2 = sns.kdeplot(X_train[y_train.squeeze() == 0, i], shade=True, legend=False)
    plt.xlabel(variable_dict[str(i)], fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.xlim([-0.7, 0.7])
    plt.xticks([-0.5, 0, 0.5], fontsize=11)
    plt.yticks(fontsize=11)

#Creation of legend
marg_c1 = list(ax1.get_lines())[0].get_c()
marg_c0 =  list(ax1.get_lines())[1].get_c()
legend_elements = [Patch(facecolor=marg_c1, label='Safe class (1)', alpha=0.75),\
                   Patch(facecolor=marg_c0, label='Unsafe class (0)', alpha=0.75)]

plt.subplot(3, 3, 9)
plt.axis('off')
plt.legend(handles=legend_elements, loc='center', fontsize=12) 
plt.tight_layout()
st = fig.suptitle('KDE-plots of class-conditional marginal distributions', fontsize=14)
st.set_y(0.95)
fig.subplots_adjust(top=0.9)



#PCA plot with different baselines

#Baseline 1: Average
X_avg = np.mean(X_train, axis=0).reshape(1, -1)
X_avg0 =np.mean(X_train[y_train.squeeze() == 0, :], axis=0).reshape(1, -1)
X_avg1 = np.mean(X_train[y_train.squeeze() == 1, :], axis=0).reshape(1, -1)


#Baseline 1: Marginal median
X_marg_med= np.median(X_train, axis=0).reshape(1, -1)
X_marg_med0 =np.median(X_train[y_train.squeeze() == 0, :], axis=0).reshape(1, -1)
X_marg_med1 = np.median(X_train[y_train.squeeze() == 1, :], axis=0).reshape(1, -1)


#Baseline 2: Mode

# Mode 0
#bandwidth_list = [1e-3, 1e-2, 2*1e-2, 3*1e-2, 5*1e-2, 7*1e-2, 0.1]
#logprob0 = []
#for i in range(len(bandwidth_list)):
#    print('Trying bandwidth number: '+str(i+1))
#    bandwidth = bandwidth_list[i]
#    kde0 = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(X_train[y_train.squeeze() == 0, :])
#    logprob0.append(kde0.score(X_val[y_val.squeeze() == 0, :]))
#    
#best_bandwidth0 = bandwidth_list[np.argmax(logprob0)]
#print('Best bandwidth: '+str(best_bandwidth0))
#kde0 = KernelDensity(kernel="gaussian", bandwidth=best_bandwidth0).fit(X_train[y_train.squeeze() == 0, :])
#kde0_scores = kde0.score_samples(X_train[y_train.squeeze() == 0, :])
#best_index0 = np.argmax(kde0_scores)
#X_mode0 = X_train[best_index0, :]
#print('Mode of class 0: )
#print(X_mode0)
#
#
##Mode 1
#bandwidth_list = [1e-3, 1e-2, 2*1e-2, 3*1e-2, 5*1e-2, 7*1e-2, 0.1]
#logprob1 = []
#for i in range(len(bandwidth_list)):
#    print('Trying bandwidth number: '+str(i+1))
#    bandwidth = bandwidth_list[i]
#    kde1 = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(X_train[y_train.squeeze() == 1, :])
#    logprob1.append(kde1.score(X_val[y_val.squeeze() == 1, :]))
#    
#best_bandwidth1 = bandwidth_list[np.argmax(logprob1)]
#print('Best bandwidth: '+str(best_bandwidth1))
#
#kde1 = KernelDensity(kernel="gaussian", bandwidth=best_bandwidth1).fit(X_train[y_train.squeeze() == 1, :])
#kde1_scores = kde1.score_samples(X_train[y_train.squeeze() == 1, :])
#best_index1 = np.argmax(kde1_scores)
#X_mode1 = X_train[best_index1, :]
#print('Mode of class 1: ')
#print(X_mode1)

X_mode0 = np.array([0.1785727099239086, 0.18055137006219624, -0.21471497474167162, -0.46217515925979175, 0.4097814237676807, 0.04865088432551268, 0.04432775616068607, -0.007010827499369432])
X_mode1 = np.array([-0.12245480257973912, 0.2815597799175341, 0.1598055177382811, -0.08816221434415339, 0.2928599311484916, 0.19733346636205495, -0.1538339318408808, -0.034965653347758474])

X_mode0 = X_mode0.reshape(1, -1)
X_mode1 = X_mode1.reshape(1, -1)

#Baseline 3: Most safe prediciton
model = keras.models.load_model('/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Models/9bus/best_acc42_best.h5')
W_list = model.weights
architechture = [W_list[0].shape[0]]
for i in range(2, len(W_list), 2):
    architechture.append(W_list[i].shape[0])
#Save weights as numpy ndarrays
for i in range(len(W_list)):
    W_list[i] = W_list[i].numpy()

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
        if self.nl > 5:
            self.n4 = nn.Linear(architechture[4], architechture[5]).double()
            self.relu4 = nn.ReLU()
        
        
        if self.nl == 2:
            self.output = nn.Linear(architechture[2], 1).double()
        elif self.nl == 3:
            self.output = nn.Linear(architechture[3], 1).double()
        elif self.nl == 4:
            self.output = nn.Linear(architechture[4], 1).double()
        elif self.nl == 5:
            self.output = nn.Linear(architechture[4], 1).double()
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


net = Net(architechture)
net.load_weights_w_bias(W_list)


X_meaned = X - X_train_mean
logits = net.forward(torch.from_numpy(X_meaned)).detach().numpy()
print('Highest logit: '+str(np.max(logits))[0:7])
print('Lowest logit: '+str(np.min(logits))[0:7])


X_m0 = X_meaned[np.argmin(logits), :].reshape(1, -1)
X_m1 = X_meaned[np.argmax(logits), :].reshape(1, -1)


#Saving all baselines
#np.savez('/Users/Eigil/Dropbox/DTU/Speciale/Data/9bus_baselines.npz',\
#         X_mode0=X_mode0, X_mode1=X_mode1, X_marg_med=X_marg_med, X_marg_med0=X_marg_med0,\
#         X_marg_med1 = X_marg_med1, X_m0 = X_m0, X_m1=X_m1)


#Visualizing the PCA
pca = decomposition.PCA(n_components=2)
pca.fit(X_train)
X2 = pca.transform(X_train)
X_avg_pca = pca.transform(X_avg)
X_avg0_pca = pca.transform(X_avg0)
X_avg1_pca = pca.transform(X_avg1)
X_marg_med_pca = pca.transform(X_marg_med)
X_marg_med0_pca = pca.transform(X_marg_med0)
X_marg_med1_pca = pca.transform(X_marg_med1)
X_mode0_pca = pca.transform(X_mode0)
X_mode1_pca = pca.transform(X_mode1)
X_m0_pca = pca.transform(X_m0)
X_m1_pca = pca.transform(X_m1)



#X_mode0 = np.array([0.1785727099239086, 0.18055137006219624, -0.21471497474167162, -0.46217515925979175, 0.4097814237676807, 0.04865088432551268, 0.04432775616068607, -0.007010827499369432])
#X_mode1 = np.array([-0.12245480257973912, 0.2815597799175341, 0.1598055177382811, -0.08816221434415339, 0.2928599311484916, 0.19733346636205495, -0.1538339318408808, -0.034965653347758474])



plt.figure(figsize=(10, 8))
sns.set_palette("deep")
c1, c0 = sns.color_palette(n_colors=2)
c1 = np.array([c1])
c0 = np.array([c0])

#Create first scatterplot
ax = sns.scatterplot(X2[:, 0], X2[:, 1], hue=y_train.squeeze(), hue_order=[1, 0], alpha=0.15)
legends, _ = ax.get_legend_handles_labels()
leg1 = ax.legend(legends, ['1', '0'], title='Class', loc='upper left')

#plt.scatter(X_avg_pca[0, 0], X_avg_pca[0, 1], s=75, marker='v', edgecolors='k', c='k')
#plt.scatter(X_avg0_pca[0, 0], X_avg0_pca[0, 1], s=75, marker='v', edgecolors='k', c=c0)
#plt.scatter(X_avg1_pca[0, 0], X_avg1_pca[0, 1], s=75, marker='v', edgecolors='k', c=c1)
plt.scatter(X_marg_med0_pca[0, 0], X_marg_med0_pca[0, 1], s=130, marker='*', edgecolors='k', c=c0)
plt.scatter(X_marg_med1_pca[0, 0], X_marg_med1_pca[0, 1], s=130, marker='*', edgecolors='k', c=c1)
plt.scatter(X_mode0_pca[0, 0], X_mode0_pca[0, 1], s=75, marker='s', edgecolors='k', c=c0)
plt.scatter(X_mode1_pca[0, 0], X_mode1_pca[0, 1], s=75, marker='s', edgecolors='k', c=c1)
plt.scatter(X_m0_pca[0, 0], X_m0_pca[0, 1], s=60, marker='D', edgecolors='k', c=c0)
plt.scatter(X_m1_pca[0, 0], X_m1_pca[0, 1], s=60, marker='D', edgecolors='k', c=c1)

#Add secondary legend

legend_elements = [Line2D([0], [0], color="w", markersize=10, marker="*", markeredgecolor="k"),\
                   Line2D([0], [0], color="w", markersize=10, marker="s", markeredgecolor="k"),\
                   Line2D([0], [0], color="w", markersize=10, marker="D", markeredgecolor="k")]
leg2 = ax.legend(handles=legend_elements, title='Baseline', labels=['Median', 'Mode', 'Extreme'], loc='upper right', fontsize=10)
ax.add_artist(leg1)
plt.xlim([-0.9, 0.9])
plt.ylim([-0.9, 0.9])
plt.xlabel('PCA component 1')
plt.ylabel('PCA component 2')
plt.title('PCA plot of the training dataset and different baselines')


### Neighbors
def NN_cum(X_baseline):
    subtract = X_train - X_baseline
    dist = np.sqrt(np.sum(np.square(subtract), axis=1))
    sort_index = np.argsort(dist)
    NN = y_train[sort_index]
    NN_cum_mean = np.cumsum(NN) / np.arange(1, len(X_train)+1)
    return(NN_cum_mean)

x = np.arange(1, len(X_train)+1)
y1 = NN_cum(X_marg_med0)
y2 = NN_cum(X_marg_med1)
y3 = NN_cum(X_mode0)
y4 = NN_cum(X_mode1)
y5 = NN_cum(X_m0)
y6 = NN_cum(X_m1)
y7 = NN_cum(X_avg)
y8 = NN_cum(X_marg_med)

plt.figure(figsize=(10, 8))
plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)
plt.plot(x, y4)
plt.plot(x, y5)
plt.plot(x, y6)
plt.plot(x, y7)
plt.plot(x, y8)


plt.legend(['Median 0', 'Median 1', 'Mode0', 'Mode 1', 'Extreme 0', 'Extreme 1', 'Overall average', 'Overall median'])
plt.title('Class ratio of nearest neighbours')
plt.xlabel('Number of neighbours considered')
plt.ylabel('Ratio')
plt.ylim([0.0, 1.05])







'''
118 bus system
'''

mat_content = sio.loadmat(datadir_118)

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


#Marginal plots
variable_dict = {'0' : 'PG2', '1' : 'PG3', '2' : 'VG1', '3' : 'VG2', '4': 'VG3', '5' : 'PD5', '6' : 'PD7', '7' : 'PD9'}
fig = plt.figure(figsize=(10, 10))
for i in range(100):
    plt.subplot(10, 10, i+1)
    ax1 = sns.kdeplot(X_train[y_train.squeeze() == 1, i], shade=True, legend=False)
    ax2 = sns.kdeplot(X_train[y_train.squeeze() == 0, i], shade=True, legend=False)
#    plt.xlabel(variable_dict[str(i)], fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.xlim([-0.7, 0.7])
    plt.xticks([-0.5, 0, 0.5], fontsize=11)
    plt.yticks(fontsize=11)
    
    

#Creation of legend
marg_c1 = list(ax1.get_lines())[0].get_c()
marg_c0 =  list(ax1.get_lines())[1].get_c()
legend_elements = [Patch(facecolor=marg_c1, label='Safe class (1)', alpha=0.75),\
                   Patch(facecolor=marg_c0, label='Unsafe class (0)', alpha=0.75)]

plt.subplot(3, 3, 9)
plt.axis('off')
plt.legend(handles=legend_elements, loc='center', fontsize=12) 
plt.tight_layout()
st = fig.suptitle('KDE-plots of class-conditional marginal distributions', fontsize=14)
st.set_y(0.95)
fig.subplots_adjust(top=0.9)




#Baseline 1: Marginal Median
X_marg_med= np.median(X_train, axis=0).reshape(1, -1)
X_marg_med0 =np.median(X_train[y_train.squeeze() == 0, :], axis=0).reshape(1, -1)
X_marg_med1 = np.median(X_train[y_train.squeeze() == 1, :], axis=0).reshape(1, -1)


#Baseline 2: Mode


#Baseline 2: Mode

# Mode 0
bandwidth_list = [1e-3, 1e-2, 2*1e-2, 3*1e-2, 5*1e-2, 7*1e-2, 0.1]
bandwidth_list = [0.1, 0.2, 0.5, 1, 2]
bandwidth_list = [0.22, 0.23]

logprob0 = []
for i in range(len(bandwidth_list)):
    print('Trying bandwidth number: '+str(i+1))
    bandwidth = bandwidth_list[i]
    kde0 = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(X_train[y_train.squeeze() == 0, :])
    logprob0.append(kde0.score(X_val[y_val.squeeze() == 0, :]))
    
best_bandwidth0 = bandwidth_list[np.argmax(logprob0)]
print('Best bandwidth: '+str(best_bandwidth0))
kde0 = KernelDensity(kernel="gaussian", bandwidth=best_bandwidth0).fit(X_train[y_train.squeeze() == 0, :])
kde0_scores = kde0.score_samples(X_train[y_train.squeeze() == 0, :])
best_index0 = np.argmax(kde0_scores)
X_mode0 = X_train[best_index0, :].reshape(1, -1)
print('Mode of class 0: ')
print(X_mode0)




#
#Mode 1
bandwidth_list = [9*1e-2, 1.1, 1.2]
logprob1 = []
for i in range(len(bandwidth_list)):
    print('Trying bandwidth number: '+str(i+1))
    bandwidth = bandwidth_list[i]
    kde1 = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(X_train[y_train.squeeze() == 1, :])
    logprob1.append(kde1.score(X_val[y_val.squeeze() == 1, :]))
    
best_bandwidth1 = bandwidth_list[np.argmax(logprob1)]
print('Best bandwidth: '+str(best_bandwidth1))

kde1 = KernelDensity(kernel="gaussian", bandwidth=best_bandwidth1).fit(X_train[y_train.squeeze() == 1, :])
kde1_scores = kde1.score_samples(X_train[y_train.squeeze() == 1, :])
best_index1 = np.argmax(kde1_scores)
X_mode1 = X_train[best_index1, :].reshape(1, -1)
print('Mode of class 1: ')
print(X_mode1)


# Baseline 3: Most extreme case
model = keras.models.load_model('/Users/Eigil/Dropbox/DTU/Speciale/Analysis/Models/118bus/Big1_best_acc74')
W_list = model.weights
architechture = [W_list[0].shape[0]]
for i in range(2, len(W_list), 2):
    architechture.append(W_list[i].shape[0])
#Save weights as numpy ndarrays
for i in range(len(W_list)):
    W_list[i] = W_list[i].numpy()

X_meaned = X - X_train_mean
logits = net.forward(torch.from_numpy(X_meaned)).detach().numpy()
print('Highest logit: '+str(np.max(logits))[0:7])
print('Lowest logit: '+str(np.min(logits))[0:7])

X_m0 = X_meaned[np.argmin(logits), :].reshape(1, -1)
X_m1 = X_meaned[np.argmax(logits), :].reshape(1, -1)

X_avg = np.mean(X_train, axis=0).reshape(1, -1)


np.savez('/Users/Eigil/Dropbox/DTU/Speciale/Data/BigCombined1_baselines.npz',\
         X_mode0=X_mode0, X_mode1=X_mode1, X_marg_med=X_marg_med, X_marg_med0=X_marg_med0,\
         X_marg_med1 = X_marg_med1, X_m0 = X_m0, X_m1=X_m1)




pca = decomposition.PCA(n_components=2)
pca.fit(X_train)

X2 = pca.transform(X_train)
X_marg_med_pca = pca.transform(X_marg_med)
X_marg_med0_pca = pca.transform(X_marg_med0)
X_marg_med1_pca = pca.transform(X_marg_med1)
X_mode0_pca = pca.transform(X_mode0)
X_mode1_pca = pca.transform(X_mode1)
X_m0_pca = pca.transform(X_m0)
X_m1_pca = pca.transform(X_m1)


plt.figure(figsize=(10, 8))
sns.set_palette("deep")
c1, c0 = sns.color_palette(n_colors=2)
c1 = np.array([c1])
c0 = np.array([c0])

#Create first scatterplot
ax = sns.scatterplot(X2[:, 0], X2[:, 1], hue=y_train.squeeze(), hue_order=[1, 0], alpha=0.15)
legends, _ = ax.get_legend_handles_labels()
leg1 = ax.legend(legends, ['1', '0'], title='Class', loc='upper left')
plt.scatter(X_marg_med0_pca[0, 0], X_marg_med0_pca[0, 1], s=130, marker='*', edgecolors='k', c=c0)
plt.scatter(X_marg_med1_pca[0, 0], X_marg_med1_pca[0, 1], s=130, marker='*', edgecolors='k', c=c1)
plt.scatter(X_mode0_pca[0, 0], X_mode0_pca[0, 1], s=75, marker='s', edgecolors='k', c=c0)
plt.scatter(X_mode1_pca[0, 0], X_mode1_pca[0, 1], s=75, marker='s', edgecolors='k', c=c1)
plt.scatter(X_m0_pca[0, 0], X_m0_pca[0, 1], s=60, marker='D', edgecolors='k', c=c0)
plt.scatter(X_m1_pca[0, 0], X_m1_pca[0, 1], s=60, marker='D', edgecolors='k', c=c1)

#Add secondary legend

legend_elements = [Line2D([0], [0], color="w", markersize=10, marker="*", markeredgecolor="k"),\
                   Line2D([0], [0], color="w", markersize=10, marker="s", markeredgecolor="k"),\
                   Line2D([0], [0], color="w", markersize=10, marker="D", markeredgecolor="k")]
leg2 = ax.legend(handles=legend_elements, title='Baseline', labels=['Median', 'Mode', 'Extreme'], loc='upper right', fontsize=10)
ax.add_artist(leg1)
plt.xlabel('PCA component 1')
plt.ylabel('PCA component 2')
plt.title('PCA plot of the training dataset and different baselines')

##Neighbors




x = np.arange(1, len(X_train)+1)
y1 = NN_cum(X_marg_med0)
y2 = NN_cum(X_marg_med1)
y3 = NN_cum(X_mode0)
y4 = NN_cum(X_mode1)
y5 = NN_cum(X_m0)
y6 = NN_cum(X_m1)
y7 = NN_cum(X_avg)
y8 = NN_cum(X_marg_med)

plt.figure(figsize=(10, 8))
plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)
plt.plot(x, y4)
plt.plot(x, y5)
plt.plot(x, y6)
plt.plot(x, y7)
plt.plot(x, y8)


plt.legend(['Median 0', 'Median 1', 'Mode0', 'Mode 1', 'Extreme 0', 'Extreme 1', 'Overall average', 'Overall median'])
plt.title('Class ratio of nearest neighbours')
plt.xlabel('Number of neighbours considered')
plt.ylabel('Ratio')
plt.ylim([0.0, 1.05])


def NN_cum_L1(X_baseline):
    subtract = X_train - X_baseline
    dist = np.sum(np.abs(subtract), axis=1)
    sort_index = np.argsort(dist)
    NN = y_train[sort_index]
    NN_cum_mean = np.cumsum(NN) / np.arange(1, len(X_train)+1)
    return(NN_cum_mean)






x = np.arange(1, len(X_train)+1)
y1 = NN_cum_L1(X_marg_med0)
y2 = NN_cum_L1(X_marg_med1)
y3 = NN_cum_L1(X_mode0)
y4 = NN_cum_L1(X_mode1)
y5 = NN_cum_L1(X_m0)
y6 = NN_cum_L1(X_m1)
y7 = NN_cum_L1(X_avg)
y8 = NN_cum_L1(X_marg_med)

plt.figure(figsize=(10, 8))
plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)
plt.plot(x, y4)
plt.plot(x, y5)
plt.plot(x, y6)
plt.plot(x, y7)
plt.plot(x, y8)


plt.legend(['Median 0', 'Median 1', 'Mode0', 'Mode 1', 'Extreme 0', 'Extreme 1', 'Overall average', 'Overall median'])
plt.title('Class ratio of nearest neighbours')
plt.xlabel('Number of neighbours considered')
plt.ylabel('Ratio')
plt.ylim([0.0, 1.05])





#Where does the data points come fro?
idx_procedure1 = mat_content['idx_procedure1']
idx_procedure2 = mat_content['idx_procedure2']

X_procedure1 = X_meaned[idx_procedure1.squeeze()-1, :]
X_procedure2 = X_meaned[idx_procedure2.squeeze()-1, :]
X_procedure1_pca = pca.transform(X_procedure1)
X_procedure2_pca = pca.transform(X_procedure2)
X_meaned_pca = pca.transform(X_meaned)

c_vector = np.zeros((X_meaned.shape[0], 1))
c_vector[idx_procedure1.squeeze()-1, 0] = 1

ax = sns.scatterplot(X_meaned_pca[:, 0], X_meaned_pca[:, 1], hue=c_vector.squeeze(), alpha=0.15, legend=True)


fig, ax = plt.subplots(figsize=(8, 6))    
scatter = ax.scatter(X_meaned_pca[:, 0], X_meaned_pca[:, 1], c=c_vector.squeeze(), alpha=0.15)
#legend1 = ax.legend(*scatter.legend_elements())
#ax.add_artist(legend1)


legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor="purple", label='LHS', markersize=12),
                   Line2D([0], [0], marker='o', color='w', label='OPF-GMM',
                          markerfacecolor='yellow', markersize=12)]
ax.legend(handles=legend_elements, title='Sampling method')
plt.xlabel('PCA component 1')
plt.ylabel('PCA cpmponent 2')
plt.title('PCA plot of the dataset, listed by sampling method')





plt.scatter(X_procedure1_pca[:, 0], X_procedure1_pca[:, 1], alpha=0)
plt.scatter(X_procedure2_pca[:, 0], X_procedure2_pca[:, 1], alpha=0.5)





