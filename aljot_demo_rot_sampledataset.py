# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 19:26:41 2018

@author: damodara
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 12:31:33 2018

@author: damodara
"""
import numpy as np
import utilstransport
import importlib
importlib.reload(utilstransport)
import pylab as pl 
import matplotlib.pyplot as plt
import dnn
from scipy.spatial.distance import cdist 
import ot

seed=1985
np.random.seed(seed)

#%% data generation
n =10000
ntest=30000
nz=.3
d=2
p=3

theta=0.8

dataset='3gauss'
X,y=utilstransport.get_dataset(dataset,n,nz)

Xtest,ytest=utilstransport.get_dataset(dataset,ntest,nz)

rindex = np.random.permutation(np.shape(Xtest)[0])
Xtest = Xtest[rindex,:]
ytest = ytest[rindex]

rindex = np.random.permutation(np.shape(X)[0])
X = X[rindex,:]
y = y[rindex]



print('Angle='+str(theta*180./np.pi))
rotation = np.array([[np.cos(theta),np.sin(theta)],
                          [-np.sin(theta),np.cos(theta)]])
 
Xtest=np.dot(Xtest,rotation.T)
 

nbnoise=0
if nbnoise:
    X=np.hstack((X,np.random.randn(n,nbnoise)))
    Xtest=np.hstack((Xtest,np.random.randn(ntest,nbnoise)))

vals=np.unique(y)
nbclass=len(vals)

Y=np.zeros((len(y),len(vals)))
Y0=np.zeros((len(y),len(vals)))
YT=np.zeros((len(ytest),len(vals)))
YT0=np.zeros((len(ytest),len(vals)))
for i,val in enumerate(vals):
    Y[:,i]=2*((y==val)-.5)
    Y0[:,i]=(y==val)
    YT[:,i]=2*((ytest==val)-.5)
    YT0[:,i]=(ytest==val)    

#%% visu dataset

if 1:
    i1=0
    i2=1;
    pl.figure(1)
    pl.clf()
    # plot 3D relations
    #t1=pl.plot(X[y==1,i1],X[y==1,i2] , 'b+') #s=WiniScaled[0:N1],
    #t2=pl.plot(X[y==-1,i1], X[y==-1,i2], 'rx')
    pl.subplot(2,2,1)
    pl.scatter(X[:,i1],X[:,i2],c=y)
    pl.title('Source data')

    pl.subplot(2,2,2)
    pl.scatter(Xtest[:,i1],Xtest[:,i2],c=ytest)
    pl.title('Target data')  
#%%
source_traindata, source_trainlabel_cat = X, Y0
target_traindata, target_trainlabel_cat = Xtest, YT0

#%%
n_class = nbclass
n_dim = np.shape(source_traindata)
optim = dnn.keras.optimizers.SGD(lr=0.01)

#%%

def feat_ext(main_input, l2_weight=0.0):
    net = dnn.Dense(500, activation='relu', name='fe')(main_input)
    net = dnn.Dense(100, activation='relu', name='feat_ext')(net)
    return net
    
def classifier(model_input, nclass, l2_weight=0.0):
    net = dnn.Dense(100, activation='relu', name='cl')(model_input)
    net = dnn.Dense(nclass, activation='softmax', name='cl_output')(net)
    return net
#%%
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val       
#%% Feature extraction model
main_input = dnn.Input(shape=(n_dim[1],))
fe = feat_ext(main_input)
fe_size=fe.get_shape().as_list()[1]
fe_model = dnn.Model(main_input, fe, name= 'fe_model')
# Classifier model
cl_input = dnn.Input(shape =(fe.get_shape().as_list()[1],))
net = classifier(cl_input , n_class)
cl_model = dnn.Model(cl_input, net, name ='classifier')
#%% source model
ms = dnn.Input(shape=(n_dim[1],))
fes = feat_ext(ms)
nets = classifier(fes,n_class)
source_model = dnn.Model(ms, nets)
source_model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
source_model.fit(source_traindata, source_trainlabel_cat, batch_size=128, epochs=10)
source_acc = source_model.evaluate(source_traindata, source_trainlabel_cat)
target_acc = source_model.evaluate(target_traindata, target_trainlabel_cat)
print("source acc", source_acc)
print("target acc", target_acc)

#%% Target model
main_input = dnn.Input(shape=(n_dim[1],))
ffe=fe_model(main_input)
net = cl_model(ffe)
#con_cat = dnn.concatenate([net, ffe ], axis=1)
model = dnn.Model(inputs=main_input, outputs=[net, ffe])
model.set_weights(source_model.get_weights())
#%% Target model loss and fit function
optim = dnn.keras.optimizers.SGD(lr=0.001,momentum=0.9, decay=0.0001, nesterov=True)

class jdot_align(object):
    def __init__(self, model, batch_size, n_class, optim, allign_loss=1.0, tar_cl_loss=1.0, verbose=1):
        self.model = model
        self.batch_size = batch_size
        self.n_class= n_class
        self.optimizer= optim
        self.gamma=dnn.K.zeros(shape=(self.batch_size, self.batch_size))
        self.train_cl =dnn.K.variable(tar_cl_loss)
        self.train_algn=dnn.K.variable(allign_loss)
        self.source_m = dnn.K.variable(1.)
        self.verbose = verbose
        
        # target classification L2 loss
        def classifier_l2_loss(y_true, y_pred):
            '''
            update the classifier based on L2 loss in the target domain
            1:batch_size - is source samples
            batch_size:end - is target samples
            self.gamma - is the optimal transport plan
            '''
            # source true labels
            ys = y_true[:batch_size,:]
            # target prediction
            ypred_t = y_pred[batch_size:,:]
            # L2 distance
            dist = dnn.K.reshape(dnn.K.sum(dnn.K.square(ys),1), (-1,1))
            dist += dnn.K.reshape(dnn.K.sum(dnn.K.square(ypred_t),1), (1,-1))
            dist -= 2.0*dnn.K.dot(ys, dnn.K.transpose(ypred_t))
            # JDOT classification loss
            loss = dnn.K.sum(self.gamma*dist)
            return self.train_cl*loss
        self.classifier_l2_loss = classifier_l2_loss
        
        def classifier_cat_loss(y_true, y_pred):
            '''
            classifier loss based on categorical cross entropy in the target domain
            1:batch_size - is source samples
            batch_size:end - is target samples
            self.gamma - is the optimal transport plan
            '''
            # source true labels
            ys = y_true[:batch_size,:]
            # target prediction
            ypred_t = y_pred[batch_size:,:]
#            source_ypred = y_pred[:batch_size,:]            
#            source_loss = dnn.K.sum(dnn.K.categorical_crossentropy(ys, source_ypred))
            
            # categorical cross entropy loss
            ypred_t = dnn.K.log(ypred_t)
            # loss calculation based on double sum (sum_ij (ys^i, ypred_t^j))
            loss = -dnn.K.dot(ys, dnn.K.transpose(ypred_t))
#            return self.train_cl*(dnn.K.sum(self.gamma * loss)+ source_loss)
            return self.train_cl*(dnn.K.sum(self.gamma * loss))
        self.classifier_cat_loss = classifier_cat_loss
        
        def L2_dist(x,y):
            '''
            compute the squared L2 distance between two matrics
            '''
            dist = dnn.K.reshape(dnn.K.sum(dnn.K.square(x),1), (-1,1))
            dist += dnn.K.reshape(dnn.K.sum(dnn.K.square(y),1), (1,-1))
            dist -= 2.0*dnn.K.dot(x, dnn.K.transpose(y))  
            return dist
 
        def align_loss(y_true, y_pred):
            '''
            source and target alignment loss in the intermediate layers of the target model
            allignment is performed in the target model (both source and target features are from targte model)
            y-true - is dummy value( that is full of zeros)
            y-pred - is the value of intermediate layers in the target model
            1:batch_size - is source samples
            batch_size:end - is target samples            
            '''
            # source domain features            
            gs = y_pred[:batch_size,:]
            # target domain features
            gt = y_pred[batch_size:,:]
            gdist = L2_dist(gs,gt)  
            
#            loss = dnn.K.sum(self.gamma*(gdist+fdist))
            loss = dnn.K.sum(self.gamma*(gdist))
            
            return self.train_algn*loss
        self.align_loss= align_loss
 

 
    def fit(self, source_traindata, ys_label, target_traindata, n_iter=1000):
        '''
        ys_label - source data true labels
        '''
        n_iter = 1000
        ns = source_traindata.shape[0]
        nt= target_traindata.shape[0]
        method='sinkhorn' # for optimal transport
        reg =0.01   # for sinkhorn
        alpha=0.00001
        self.model.compile(optimizer= optim, loss =[self.classifier_cat_loss, self.align_loss])
        for i in range(500):
#            p = float(i) / 500.0
#            lr = 0.01 / (1. + 10 * p)**0.75
#            dnn.K.set_value(self.model.optimizer.lr, lr)
            # fixing f and g, and computing optimal transport plan (gamma)
            s_ind = np.random.choice(ns, self.batch_size)
            t_ind = np.random.choice(nt,self.batch_size)
            
            xs_batch, ys = source_traindata[s_ind], ys_label[s_ind]
            xt_batch = target_traindata[t_ind] 
            
            l_dummy = np.zeros_like(ys)
            g_dummy = np.zeros((2*batch_size, fe_size))
            s = xs_batch.shape
            
            # concat of source and target samples and prediction
            modelpred = self.model.predict(np.vstack((xs_batch, xt_batch)))
            # intermediate features
            gs_batch = modelpred[1][:batch_size, :]
            gt_batch = modelpred[1][batch_size:, :]
            # softmax prediction of target samples
            ft_pred = modelpred[0][batch_size:,:]
            
            C0 = cdist(gs_batch, gt_batch, metric='sqeuclidean')
            
            C1 = cdist(ys, ft_pred, metric='sqeuclidean')
            
            C= alpha*C0+C1
                             
            # transportation metric
            
            if method == 'emd':
                 gamma=ot.emd(ot.unif(gs_batch.shape[0]),ot.unif(gt_batch.shape[0]),C)
            elif method =='sinkhorn':
                 gamma=ot.sinkhorn(ot.unif(gs_batch.shape[0]),ot.unif(gt_batch.shape[0]),C,reg)
            # update the computed gamma                      
            dnn.K.set_value(self.gamma, gamma)

            # activate the classifier loss 
#            dnn.K.set_value(self.train_cl,1.0)
#            dnn.K.set_value(self.source_m,0.0)
 #           dnn.K.set_value(self.train_algn,1.0)
            
            data = np.vstack((xs_batch, xt_batch))    
            hist= self.model.train_on_batch([data], [np.vstack((ys,l_dummy)), g_dummy])
            if self.verbose:
               print ('cl_loss ={:f}, fe_loss ={:f}'.format(hist[1], hist[2]))
            
        

    def predict(self, data):
        ypred = self.model.predict(data)
        return ypred

    def evaluate(self, data, label):
        ypred = self.model.predict(data)
        score = np.mean(np.argmax(label,1)==np.argmax(ypred[0],1))
        return score
    
#%% target model training
batch_size=500
al_model = jdot_align(model, batch_size, n_class, optim)
al_model.fit(source_traindata, source_trainlabel_cat, target_traindata)
    
#%% accuracy assesment
acc = al_model.evaluate(target_traindata, target_trainlabel_cat)
tarmodel_sacc = al_model.evaluate(source_traindata, source_trainlabel_cat)    
print("target domain acc", acc)
print("trained on target, source acc", tarmodel_sacc)
#%% feature ext
def feature_extraction(model, data, out_layer_num=-2, out_layer_name=None):
    '''
    extract the features from the pre-trained model
    inp_layer_num - input layer
    out_layer_num -- from which layer to extract the features
    out_layer_name -- name of the layer to extract the features
    '''
    if out_layer_name is None:
        intermediate_layer_model = dnn.Model(inputs=model.layers[0].input,
                             outputs=model.layers[out_layer_num].output)
        intermediate_output = intermediate_layer_model.predict(data)
    else:
        intermediate_layer_model = dnn.Model(inputs=model.layers[0].input,
                             outputs=model.get_layer(out_layer_name).output)
        intermediate_output = intermediate_layer_model.predict(data)
        
    
    return intermediate_output
#%%    
smodel_source_feat = feature_extraction(source_model, source_traindata[:1000,],
                                        out_layer_name='feat_ext')
smodel_target_feat  = feature_extraction(source_model, target_traindata[:1000,],
                                        out_layer_name='feat_ext')

#%% intermediate layers of source and target domain for TSNE plot of target model
subset = 1000
al_sourcedata = model.predict(source_traindata[:subset,])[1]
al_targetdata = model.predict(target_traindata[:subset,])[1]

#%%
def tsne_plot(xs, xt, xs_label, xt_label, subset=True, title=None, pname=None):

    num_test=1000
    if subset:
        combined_imgs = np.vstack([xs[0:num_test, :], xt[0:num_test, :]])
        combined_labels = np.vstack([xs_label[0:num_test, :],xt_label[0:num_test, :]])
        combined_labels = combined_labels.astype('int')
        combined_domain = np.vstack([np.zeros((num_test,1)),np.ones((num_test,1))])
    
    from sklearn.manifold import TSNE
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    source_only_tsne = tsne.fit_transform(combined_imgs)
    plt.figure()
    plt.scatter(source_only_tsne[:num_test,0], source_only_tsne[:num_test,1], c=combined_labels[:num_test].argmax(1), marker='o', label='source')
    plt.scatter(source_only_tsne[num_test:,0], source_only_tsne[num_test:,1], c=combined_labels[num_test:].argmax(1),marker='+',label='target')
    plt.legend(loc='best')
    plt.title(title)

#%%
title = 'tsne plot of source and target data with source model'
tsne_plot(smodel_source_feat, smodel_target_feat, source_trainlabel_cat, target_trainlabel_cat, title=title)

title = 'tsne plot of source and target data with target model'
tsne_plot(al_sourcedata, al_targetdata, source_trainlabel_cat, target_trainlabel_cat, title=title)

#plt.figure(num=7)
#plt.scatter(al_sourcedata[:,0], al_sourcedata[:,1], c=y, alpha=0.9)
#plt.scatter(al_targetdata[:,0], al_targetdata[:,1], c=ytest, cmap='cool', alpha=0.4)
 
