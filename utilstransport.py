# -*- coding: utf-8 -*-
"""
Created on Tue May 20 08:48:41 2014

@author: rflamary
"""

import os, sys, traceback,time
sys.path.append( "../scikit-learn/" )
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pylab as pylab
import numpy as np
import scipy as sp
import pylab as pl
from math import exp
from sklearn import svm
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.datasets import fetch_olivetti_faces
from sklearn import metrics
from sklearn import preprocessing, grid_search,neighbors
from sklearn.metrics.pairwise import euclidean_distances as skdist
from cvxopt import matrix, spmatrix, solvers, printing
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import linear_kernel
import sklearn.metrics
from sklearn.linear_model import LogisticRegression
from scipy.io import loadmat,savemat
from scipy.spatial.distance import cdist, pdist
from sklearn.datasets import load_svmlight_file
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import transport
#import mosek
#import cvxopt.msk
#cvxopt.msk.options = {mosek.iparam.log:0}

#pl.rcParams['figure.figsize'] = (10.0, 8.0)

def dist(x1,x2=None,metric='sqeuclidean'):
    """Compute distance between samples in x1 and x2"""
    if x2 is None:
        return pdist(x1,metric=metric)
    else:
        return cdist(x1,x2,metric=metric)


def computeTransportDistance(cost,transp):
    return np.trace(np.dot(cost.T,transp))
    
def get_W(x,method='unif',param=None):
    """ returns the density estimation for a discrete distribution"""
    if method.lower()=='rbf':
        K = rbf_kernel(x,x,param)
        W = np.sum(K,1)
        W = W/sum(W)
    else:
        if not method.lower()=='unif':
            print("Warning: unknown density estimation, revert to uniform")
        W = np.ones(x.shape[0])/x.shape[0]
    return W
    
    
def get_dataset(dataset,n,nz=.5,**kwargs):
    """
    dataset generation
    """
    if dataset.lower()=='twomoons':
        x, y = datasets.make_moons(n_samples=n, noise=nz)
        ind=np.argsort(y)
        x=x[ind,:]
        y=y[ind]
        
    elif dataset.lower()=='3gauss':
        y=np.floor((np.arange(n)*1.0/n*3))+1
        x=np.zeros((n,2))
        # class 1
        x[y==1,0]=-1.; x[y==1,1]=-1.
        x[y==2,0]=-1.; x[y==2,1]=1.
        x[y==3,0]=1. ; x[y==3,1]=0
        
        x[y!=3,:]+=nz*np.random.randn(sum(y!=3),2)
        x[y==3,:]+=2*nz*np.random.randn(sum(y==3),2)
        
    elif dataset.lower()=='3gauss2':
        y=np.floor((np.arange(n)*1.0/n*4))+1
        x=np.zeros((n,2))
        y[y==4]=3
        # class 1
        x[y==1,0]=-1.; x[y==1,1]=-1.
        x[y==2,0]=-1.; x[y==2,1]=1.
        x[y==3,0]=1. ; x[y==3,1]=0
        
        x[y!=3,:]+=nz*np.random.randn(sum(y!=3),2)
        x[y==3,:]+=2*nz*np.random.randn(sum(y==3),2)   
    elif dataset.lower()=='sinreg':
        
        x=np.random.rand(n,1)
        y=4*x+np.sin(2*np.pi*x)+nz*np.random.randn(n,1) 
         
    else:
        x=0
        y=0
        print("unknown dataset")
    
    return x,y
    
def transp_source(xt,transp):
    """ transport source samples to target space (requires the target samples xt)"""
    transp1 = np.dot(np.diag(1/np.sum(transp,1)),transp)
    return np.dot(transp1,xt)

def transp_source_t(xs,xt,transp,t=1):
    """ transport source samples to target space (requires the target samples xt)"""
    return (1-t)*xs + t * transp_source(xt,transp)

def transp_target(x,transp):
    """ transport target samples to source space (requires the source samples x)"""
    transp2 = np.dot(transp,np.diag(1/np.sum(transp,0)))
    return np.dot(transp2.T,x)

def transp_target_t(xs,xt,transp,t=1):
    """ transport source samples to target space (requires the target samples xt)"""
    return (1-t)*xt + t * transp_target(xs,transp)
    

def cross_valid_one_param(xs,ys,xt,param_name,test_range,classifier,method_CV='cross_class',yt=None,disp=False,**kwargs):
    params=kwargs.copy()
    results=[]
    for test in test_range:
        params[param_name]=test
        transp = transport.compute_transport(xs,xt,**params)    
        results.append(get_perf_estimate(xs,ys,xt,transp,classifier,method_CV,yt))
    best_index = np.argmax(results)
    best_perf = np.max(results)
    best_param = test_range[best_index]
    if disp:
        pl.plot(test_range,results,'+-')
        pl.title('Param '+param_name)
        pl.show()
    return {'best_param':best_param,'best_perf':best_perf,'index':best_index,'results':results}
    
    
#def valid_reg_sinkhorn(x,y,xt,w,wt,M=None,reglist=[.01,.1,1,10,100,1000],classifier=LDA,method='cross_class'):
#    """ Find the best parameters for a given transport problem                     """
#    """ output: transport matrix, regularization parameter, a performance table    """
#    
#    nbreg=len(reglist)
#
#    # if the distance matrix is not given, compute it
#    if M==None:
#        M = dist(x,xt)
#        M=M/np.median(M)    
#        
#    perf=np.zeros(nbreg)
#    # test every possible regulaeization possibility
#    for i in range(nbreg):
#        reg=reglist[i]
#        transp = transport.computeTransportSinkhorn(w,wt,M,reg)
#        perf[i]=get_perf_estimate(x,y,xt,transp,classifier,method)
#        
#    # select the best, and output the corresponding values
#    i=np.argmax(perf)
#    reg=reglist[i]
#    transp = transport.computeTransportSinkhorn(w,wt,M,reg)
#
#    return transp,reg,perf
#
#    
#def valid_reg_class(x,y,xt,w,wt,M=None,reglist=[.01,.1,1,10,100,1000],etalist=[.01,.1,1,10,100,1000],classifier=LDA,optim='sinkhorn',method='cross_class',yt=None):
#    """ Find the best parameters for a given transport problem                     """
#    """ output: transport matrix, regularization  and eta parameters, a performance table (ravel)   """
#
#    nbreg=len(reglist) 
#    nbeta=len(etalist)
#
#    # if the distance matrix is not given, compute it
#    if M==None:
#        M = dist(x,xt)
#        M=M/np.median(M)    
#        
#    perf=np.zeros((nbreg,nbeta))
#    # test every possible regulaeization possibility
#    for i in range(nbreg):
#        for j in range(nbeta):
#            reg=reglist[i]
#            eta=etalist[j]
#            if optim.lower()=='sinkhorn':
#                transp = transport.computeTransportSinkhornLabelsLpL1(w,y,wt,M,reg,eta)
#            if optim.lower()=='qp':
#                transp = transport.computeTransportQPLabelsLpL1(w,y,wt,M,reg,eta,solver='mosek')
#            perf[i,j]=get_perf_estimate(x,y,xt,transp,classifier,method,yt)
#
#    # select the best, and output the corresponding values
#    temp=np.argmax(perf)
#    i,j=np.unravel_index(temp,perf.shape)
#    reg=reglist[i]
#    eta=etalist[j]
#    if optim.lower()=='sinkhorn':
#        transp = transport.computeTransportSinkhornLabelsLpL1(w,y,wt,M,reg,eta)
#    if optim.lower()=='qp':
#        transp = transport.computeTransportQPLabelsLpL1(w,y,wt,M,reg,eta,solver='mosek')
#
#    return transp,reg,eta,perf

def get_perf_estimate(x,y,xt,transp,classifier,method='cross_class',yt=None):
    """ Estimate the performance of the transport by cross validating the classification along the transport      """
    """ output: performance score of the classifier for a given set of parameters   """

    lstclass=np.unique(y);

    # transform source points to target
    xp=transp_source(xt,transp)
    # transform target points to source
    xpt=transp_target(x,transp)


    clf = classifier.fit(x,y)
    ypred=clf.predict(xpt)
    #print ypred
    #score=clf.transform(xpt)

    clft = classifier.fit(xp,y)
    ypredt=clft.predict(xt)
    #print ypredt
    #scoret=clft.transform(xt)

    # number of temporal discretization along the transport
    nbt=10
    
    if method.lower()=='cross_class_integrate':
        tlist=np.linspace(0,1,nbt)
        ypredt=np.zeros((xt.shape[0],nbt))
        for j in range(nbt):
            xp2=transp_source_t(x,xt,transp,tlist[j])
            xptest2=transp_target_t(x,xt,transp,1-tlist[j])
            clft = classifier.fit(xp2,y)
            ypredt[:,j] = clft.predict(xptest2)
        ypredt2=np.zeros((xt.shape[0],len(lstclass)))
        for j in range(len(lstclass)):
            ypredt2[:,j]=np.sum(ypredt==lstclass[j],1)
        perf=np.mean(np.max(ypredt2,1))/nbt
    elif method.lower()=='cross_class_integrate_nico':
        tlist=np.linspace(0,1,nbt)
        xp_bigvector = np.empty(shape=(len(x),0))
        xptest_bigvector = np.empty(shape=(len(xt),0))
        for j in range(nbt):
            xp_bigvector=np.concatenate((xp_bigvector,transp_source_t(x,xt,transp,tlist[j])),axis=1)
            xptest_bigvector=np.concatenate((xptest_bigvector,transp_target_t(x,xt,transp,1-tlist[j])),axis=1)
        clft = classifier.fit(xp_bigvector,y)
        ypredt = clft.predict(xptest_bigvector)
        perf=np.mean(ypred==ypredt)
    elif method.lower()=='model_selection':
        # model selection in the spirit of Bruzzone DASVM (PAMI'10)
        clf1 = classifier.fit(xpt,ypredt)
        ypreds = clf1.predict(x)
        perf=1-np.mean(y==ypreds)
    elif method.lower()=='heuristic':
        dim_source = np.max(dist(x))
        dim_target = np.max(dist(xt))
        dim_transp = np.max(dist(xp))
        #print dim_source,dim_target,dim_transp
        #perf = np.exp(-np.abs(dim_target - dim_transp)/dim_target)        
        perf = np.exp(-np.abs(dim_source/20.0 - dim_transp)/dim_source)        
    elif method.lower()=='gt':
        perf=np.mean(yt==ypredt)
    else:
        # cross class
        perf=np.mean(ypred==ypredt)         

    return perf

    
    
def transp_dirac(xs,xt,t=1):
    ns=xs.shape[0]
    nt=xt.shape[0]
    
    xres=np.zeros((ns*nt,xs.shape[1]))
    
    for i in range(ns):
        for j in range(nt):
            xres[i+j*ns,:]=(1-t)*xs[i,:]+t*xt[j,:]
            
    return xres
        
    
def predict_trans_class1(y,transp):
    transp2 = np.dot(transp,np.diag(1/np.sum(transp,0)))
    lstclass=np.unique(y);
    nbclass=len(lstclass)
    ytemp=np.zeros((transp.shape[0],nbclass))
    for i in range(nbclass):
        ytemp[y==lstclass[i],i]=1
    scores= np.dot(transp2.T,ytemp)
    return lstclass[np.argmax(scores,1)]
    
def predict_trans_class2(y,transp):
    transp1 = np.dot(np.diag(1/np.sum(transp,1)),transp)
    lstclass=np.unique(y);
    nbclass=len(lstclass)
    ytemp=np.ones((transp.shape[1],nbclass))
    
    scores= np.dot(transp1,ytemp)
    return lstclass[np.argmin(scores,1)]    
        
    
    
    
    