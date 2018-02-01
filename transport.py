#!/usr/bin/env python

import os, sys, traceback
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pylab as pylab
import numpy as np
import pylab as pl
import scipy as sci
import scipy.optimize.linesearch  as ln

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph as kn_graph

from cvxopt import matrix, spmatrix, solvers, printing
solvers.options['show_progress'] = False

### ------------------------------- Optimal Transport ---------------------------------------

########### Compute transport with a LP Solver

def computeTransportLP(distribWeightS,distribWeightT, distances):
	# init data
	Nini = len(distribWeightS)
	Nfin = len(distribWeightT)
	

	# generate probability distribution of each class
	p1p2 = np.concatenate((distribWeightS,distribWeightT))
	p1p2 = p1p2[0:-1]
	# generate cost matrix
	costMatrix = distances.flatten()

	# express the constraints matrix
	I = []
	J = []
	for i in range(Nini):
		for j in range(Nfin):
			I.append(i)
			J.append(i*Nfin+j)
	for i in range(Nfin-1):
		for j in range(Nini):
			I.append(i+Nini)
			J.append(j*Nfin+i)

	A = spmatrix(1.0,I,J)

	# positivity condition
	G = spmatrix(-1.0,range(Nini*Nfin),range(Nini*Nfin))

	sol = solvers.lp(matrix(costMatrix),G,matrix(np.zeros(Nini*Nfin)),A,matrix(p1p2))
	S = np.array(sol['x'])

	Gamma = np.reshape([l[0] for l in S],(Nini,Nfin))
	return Gamma

########### Compute transport with the Sinkhorn algorithm
## ref "Sinkhorn distances: Lightspeed computation of Optimal Transport", NIPS 2013, Marco Cuturi

def computeTransportSinkhorn(distribS,distribT, M, reg,Mmax=0,numItermax = 200,stopThr=1e-9):
    # init data
    Nini = len(distribS)
    Nfin = len(distribT)
    
    
    cpt = 0
    
    # we assume that no distances are null except those of the diagonal of distances
    u = np.ones(Nini)/Nini
    v = np.ones(Nfin)/Nfin 
    uprev=np.zeros(Nini)
    vprev=np.zeros(Nini)
    if Mmax:
        regmax=300./Mmax
    else:
        regmax=300./np.max(M)
    reg=regmax*(1-np.exp(-reg/regmax)) 
    #print reg
 
    K = np.exp(-reg*M)
    #print np.min(K)
      
    Kp = np.dot(np.diag(1/distribS),K)
    transp = K
    cpt = 0
    err=1
    while (err>stopThr and cpt<numItermax):
        if np.any(np.dot(K.T,u)==0) or np.any(np.isnan(u)) or np.any(np.isnan(v)):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errrors')
            if cpt!=0:
                u = uprev
                v = vprev     
            break
        uprev = u
        vprev = v  
        v = np.divide(distribT,np.dot(K.T,u))
        u = 1./np.dot(Kp,v)
        if cpt%10==0:
            # we can speed up the process by checking for the error only all the 10th iterations
            transp = np.dot(np.diag(u),np.dot(K,np.diag(v)))
            err = np.linalg.norm((np.sum(transp,axis=0)-distribT))**2
        cpt = cpt +1
    #print 'err=',err,' cpt=',cpt  

    return np.dot(np.diag(u),np.dot(K,np.diag(v)))


########### Compute transport with the Sinkhorn algorithm + Class regularization
## ref "Domain adaptation with regularized optimal transport ", ECML 2014, 


def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]


def computeTransportSinkhornLabelsLpL1(distribS,LabelsS, distribT, M, reg, eta=0.1,nbitermax=10):
    p=0.5
    epsilon = 1e-3

    # init data
    Nini = len(distribS)
    Nfin = len(distribT)
     
    W=np.zeros(M.shape)

    for cpt in range(nbitermax):
        Mreg = M + eta*W
        transp=computeTransportSinkhorn(distribS,distribT,Mreg,reg,numItermax = 200)
        # the transport has been computed. Check if classes are really separated
        W = np.ones((Nini,Nfin))
        for t in range(Nfin):            
            for c in np.unique(LabelsS):
                maj = p*((np.sum(transp[LabelsS==c,t])+epsilon)**(p-1))
                W[LabelsS==c,t]=maj
    
    return transp


########### Compute transport with the Generalized conditionnal gradient method + Group-Lasso  Class regularization
## ref "Optimal transport for Domain Adaptation ", T PAMI 2016



def get_W_L1L2(transp,labels,lstlab):
    W=np.zeros(transp.shape)
    
    for i in range(transp.shape[1]):
        for lab in lstlab:
            temp=transp[labels==lab,i]
            n=np.linalg.norm(temp)
            if n:
                W[labels==lab,i]=temp/n 
    return W

def loss_L1L2(transp,labels,lstlab):
    res=0
    
    for i in range(transp.shape[1]):
        for lab in lstlab:
            temp=transp[labels==lab,i]
            #W[]
            res+=np.linalg.norm(temp)
             
    return res



def computeTransportL1L2_CGS(distribS,LabelsS, distribT, M, reg, eta=0.1,nbitermax=10,thr_stop=1e-8,**kwargs):    
    Nini = len(distribS)
    Nfin = len(distribT)

    W=np.zeros(M.shape)
      
    maxdist = np.max(M) 
    distances=M
    
    lstlab=np.unique(LabelsS)
    
    regmax=300./maxdist
    reg0=regmax*(1-np.exp(-reg/regmax))   
   
    transp= computeTransportSinkhorn(distribS,distribT,distances,reg,maxdist)
    
    niter=1;
    while True:       
        old_transp=transp.copy()    
        
        W = get_W_L1L2(old_transp,LabelsS,lstlab)
        G=eta*W      
        transp0= computeTransportSinkhorn(distribS,distribT,distances + G,reg,maxdist)
        deltatransp = transp0 - old_transp
        # do a line search for best tau
        def f(tau):
            T = old_transp+tau*deltatransp
            return np.sum(T*distances)+1./reg0*np.sum(T*np.log(T))+eta*loss_L1L2(T,LabelsS,lstlab)
        
        # compute f'(0)
        res=0
        for i in range(transp.shape[1]):
            for lab in lstlab:
                temp1=old_transp[LabelsS==lab,i]
                temp2=deltatransp[LabelsS==lab,i]
                res+=np.dot(temp1,temp2)/np.linalg.norm(temp1)
        derphi_zero = np.sum(deltatransp*distances) + np.sum(deltatransp*(1+np.log(old_transp)))/reg0 + eta*res

        tau,cost = ln.scalar_search_armijo(f, f(0), derphi_zero,alpha0=0.99)
        if tau is None:
            break
        transp=(1-tau)*old_transp+tau*transp0
        
        if niter>=nbitermax or np.sum(np.fabs(deltatransp))<thr_stop:
            break
        niter+=1
    #print 'nbiter=',niter
    return transp
    
    
########### Compute transport with the Generalized conditionnal gradient method + Laplacian regularization
## ref "Optimal transport for Domain Adaptation ", T PAMI 2016
def get_sim(x,sim,**kwargs):
    if sim=='gauss':
        try: 
            rbfparam=kwargs['rbfparam']
        except KeyError:      
            rbfparam=1
        S=rbf_kernel(x,x,rbfparam)  
    elif sim=='gaussthr':
        try: 
            rbfparam=kwargs['rbfparam']
        except KeyError:      
            rbfparam=1
        try: 
            thrg=kwargs['thrg']
        except KeyError:      
            thrg=.5
        S=np.float64(rbf_kernel(x,x,rbfparam)>thrg)
    elif sim=='gaussclass':
        try: 
            rbfparam=kwargs['rbfparam']
        except KeyError:      
            rbfparam=1
        try: 
            y=kwargs['labels']
        except KeyError:      
            raise KeyError('sim="gaussclass" require the source labels "labels" to be passed as parameters')
        S=rbf_kernel(x,x,rbfparam) 
        temp=np.tile(y.T,(y.shape[0],1))
        temp2=temp==temp.T
        S=S*temp2        
    elif sim=='knn':
        try: 
            num_neighbors=kwargs['nn']
        except KeyError('sim="knn" requires the number of neighbors nn to be set'):      
            num_neighbors=3
        S=kn_graph(x,num_neighbors,include_self=True).toarray()
        S=(S+S.T)/2
    elif sim=='knnclass':
        try: 
            num_neighbors=kwargs['nn']
        except KeyError('sim="knnclass" requires the number of neighbors nn to be set'):      
            num_neighbors=3
        try: 
            y=kwargs['labels']
        except KeyError:      
            raise KeyError('sim="gaussclass" requires the source labels "labels" to be passed as parameters')
        S=kn_graph(x,num_neighbors,include_self=True).toarray() 
        # handle unlabelled data (class=-1)
        temp=np.tile(y.T,(y.shape[0],1))
        temp2=(temp==temp.T)* (temp!=-1)
        S=(S+S.T)/2
        S=S*temp2   
        
    return S

def get_gradient(transp,K):
    s=transp.shape
    res=np.dot(K,transp.flatten())
    return res.reshape(s)
    
def get_gradient1(L,X,transp):  
    """
    Compute gradient for the laplacian reg term on transported sources
    """
    return np.dot(L+L.T,np.dot(transp,np.dot(X,X.T)))
    
def get_gradient2(L,X,transp):    
    """
    Compute gradient for the laplacian reg term on transported targets
    """    
    return np.dot(X,np.dot(X.T,np.dot(transp,L+L.T)))
    
def get_laplacian(S):
    L=np.diag(np.sum(S,axis=0))-S
    return L
    
def quadloss(transp,K):
    """
    Compute quadratic loss with matrix K
    """
    return np.sum(transp.flatten()*np.dot(K,transp.flatten()))
    
def quadloss1(transp,L,X):
    """
    Compute loss for the laplacian reg term on transported sources
    """
    return np.trace(np.dot(X.T,np.dot(transp.T,np.dot(L,np.dot(transp,X)))))
    
def quadloss2(transp,L,X):
    """
    Compute loss for the laplacian reg term on transported sources
    """
    return np.trace(np.dot(X.T,np.dot(transp,np.dot(L,np.dot(transp.T,X)))))


def get_laplacian(S):
    L=np.diag(np.sum(S,axis=0))-S
    return L
    
def computeTransportLaplacian_CGS(distribS,LabelsS, distribT,distances,xs,xt,reg=1e-9,regls=0,reglt=0,nbitermax=10,thr_stop=1e-8,**kwargs):
    
    Ss=get_sim(xs,'knnclass',nn=7,labels=LabelsS)
    St=get_sim(xt,'knn',nn=7)
            
    Ls=get_laplacian(Ss)
    Lt=get_laplacian(St)
      
    maxdist = np.max(distances) 

    regmax=300./maxdist
    reg0=regmax*(1-np.exp(-reg/regmax))   
       
    transp= computeTransportSinkhorn(distribS,distribT,distances,reg,maxdist)
            
    niter=1;
    while True:       
        old_transp=transp.copy()       
        G=regls*get_gradient1(Ls,xt,old_transp)+reglt*get_gradient2(Lt,xs,old_transp)      
        transp0= computeTransportSinkhorn(distribS,distribT,distances + G,reg,maxdist)       
        E=transp0-old_transp
        # do a line search for best tau
        def f(tau):
            T = (1-tau)*old_transp+tau*transp0
            return np.sum(T*distances)+1./reg0*np.sum(T*np.log(T))+regls*quadloss1(T,Ls,xt)+reglt*quadloss2(T,Lt,xs)
        
        # compute f'(0)
        res = regls*(np.trace(np.dot(xt.T,np.dot(E.T,np.dot(Ls,np.dot(old_transp,xt)))))+\
                     np.trace(np.dot(xt.T,np.dot(old_transp.T,np.dot(Ls,np.dot(E,xt))))))\
             +reglt*(np.trace(np.dot(xs.T,np.dot(E,np.dot(Lt,np.dot(old_transp.T,xs)))))+\
                     np.trace(np.dot(xs.T,np.dot(old_transp,np.dot(Lt,np.dot(E.T,xs))))))

                     
        derphi_zero = np.sum(E*distances) + np.sum(E*(1+np.log(old_transp)))/reg0 + res  
        
        tau,cost = ln.scalar_search_armijo(f, f(0),derphi_zero,alpha0=0.99)
       


        if tau is None:
            break
        transp=(1-tau)*old_transp+tau*transp0
        
        if niter>=nbitermax or np.sum(np.fabs(E))<thr_stop:
            break
        niter+=1

    return transp