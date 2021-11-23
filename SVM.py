# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 10:51:10 2021
 
@author: vitfv
"""
import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets import make_moons,make_circles
from sklearn.svm import SVC

points = 50
M1 = [-1,-1]
M2 = [1,0]

B1_linsep = [[0.05,0.1],
             [0.1,0.05]]
B2_linsep = [[0.1,0.01],
             [0.01,0.02]]

B1_non_linsep = [[0.2,0.1],
                 [0.1,0.2]]
B2_non_linsep = [[0.3,0.04],
                 [0.01,0.1]]

np.random.seed(42)
vec1_linsep = np.random.multivariate_normal(M1,B1_linsep,size=points)
vec2_linsep = np.random.multivariate_normal(M2,B2_linsep,size=points)

vec1_non_linsep = np.random.multivariate_normal(M1,B1_non_linsep,size=points)
vec2_non_linsep = np.random.multivariate_normal(M2,B2_non_linsep,size=points)

plt.figure(dpi=600)
plt.title('Linsep classes')
plt.scatter(vec1_linsep[:,0],vec1_linsep[:,1],marker='+')
plt.scatter(vec2_linsep[:,0],vec2_linsep[:,1],marker='*')

plt.figure(dpi=600)
plt.title('Non - linsep classes')
plt.scatter(vec1_non_linsep[:,0],vec1_non_linsep[:,1],marker='+')
plt.scatter(vec2_non_linsep[:,0],vec2_non_linsep[:,1],marker='*')

vec_linsep = np.concatenate((vec1_linsep,vec2_linsep),axis=0)
vec_non_linsep = np.concatenate((vec1_linsep,vec2_linsep),axis=0)

labels1 = -np.ones((vec1_linsep.shape[0],1))
labels2 = np.ones((vec2_linsep.shape[0],1))
labels = np.concatenate((labels1,labels2),axis=0)
def SVM_linsep(vec,labels,kernel=np.matmul,C=None):
    K=np.zeros([vec.shape[0],vec.shape[0]])
    test = vec[0,:]
    for i in range(vec.shape[0]):
        for j in range(vec.shape[0]):
            K[i,j]=kernel(vec[i,:],vec[j,:])
    P = cvxopt.matrix(np.outer(labels,labels)*K)
    q = cvxopt.matrix(np.ones(vec.shape[0])*-1)
    A = cvxopt.matrix(labels,(1,vec.shape[0]))
    b = cvxopt.matrix(0.0)
    if(C==None):
        G = cvxopt.matrix(-np.eye(vec.shape[0]))
        h = cvxopt.matrix(np.zeros(vec.shape[0]))
    else:  
        G = cvxopt.matrix(np.vstack((np.diag(np.ones(vec.shape[0]) * -1), np.identity(vec.shape[0]))))
        h = cvxopt.matrix(np.hstack((np.zeros(vec.shape[0]), np.ones(vec.shape[0]) * C)))
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    lbda = np.ravel(solution['x'])
    sv = vec[lbda>1e-5]
    
    if(kernel is not np.matmul):
        w = kernel(vec,vec.T)@(lbda.reshape(-1,1)*labels)
        wn = -(np.max(w[labels==-1])+np.min(w[labels==1]))/2
    else:
        w = sv.T@(lbda[lbda>1e-5].reshape(-1,1)*labels[lbda>1e-5])
        wn = -(np.max((vec@w)[labels==-1])+np.min((vec@w)[labels==1]))/2
    return w, wn, sv,lbda

def plot_results(title,gridx,gridy,mesh,vec1,vec2,support):
    plt.figure(dpi=600)
    plt.title(title)
    plt.contourf(gridx,gridy,mesh,alpha=0.5,cmap='seismic')
    plt.scatter(vec1[:,0],vec1[:,1],marker='+')
    plt.scatter(vec2[:,0],vec2[:,1],marker='*')
    plt.scatter(support[:,0],support[:,1],alpha=0.5,color='blue')
    plt.show()
    
def poly_homogen(a,b):
    return np.matmul(a,b)**2

def euclidean(x,y):
    x_copy = np.array(x)
    y_copy = np.array(y).T
    if len(x_copy.shape)==1:
        x_copy = np.expand_dims(x_copy,axis=0)
    if len(y_copy.shape)==1:
        y_copy = np.expand_dims(y_copy,axis=0)
    return euclidean_distances(x_copy,y_copy)

def rbf(a,b):
    return np.exp(-(euclidean(a, b))/(2*(2**2)))
 

weights, intercept, support, lbda = SVM_linsep(vec_linsep, labels)

gridx,gridy = np.meshgrid(np.linspace(-3,3,100),np.linspace(-3,3,100))
grid = np.hstack((gridx.reshape(-1,1),gridy.reshape(-1,1)))
weights, intercept, support,lbda = SVM_linsep(vec_linsep, labels,np.matmul,10)
mesh = (grid@weights).reshape(100,100)

plot_results('Linear SVM, linsep case', gridx, gridy, mesh, vec1_linsep, vec2_linsep, support)


weights, intercept, support, lbda = SVM_linsep(vec_non_linsep, labels,C = 0.1)

mesh = (grid@weights).reshape(100,100)

plot_results('Linear SVM, non-linsep case, C=0.1', gridx, gridy, mesh, vec1_non_linsep, vec2_non_linsep, support)


weights, intercept, support, lbda = SVM_linsep(vec_non_linsep, labels,C = 1)

mesh = (grid@weights).reshape(100,100)

plot_results('Linear SVM, non-linsep case, C=1', gridx, gridy, mesh, vec1_non_linsep, vec2_non_linsep, support)


weights, intercept, support, lbda = SVM_linsep(vec_non_linsep, labels,C = 10)

mesh = (grid@weights).reshape(100,100)

plot_results('Linear SVM, non-linsep case, C=10', gridx, gridy, mesh, vec1_non_linsep, vec2_non_linsep, support)


weights, intercept, support, lbda = SVM_linsep(vec_non_linsep, labels,rbf,C = 0.1)

mesh = (rbf(grid,vec_non_linsep.T)@(lbda.reshape(-1,1)*labels)).reshape(100,100)

plot_results('Kernel SVM, non-linsep case, C=0.1', gridx, gridy, mesh, vec1_non_linsep, vec2_non_linsep, support)


weights, intercept, support, lbda = SVM_linsep(vec_non_linsep, labels,rbf,C = 1)

mesh = (rbf(grid,vec_non_linsep.T)@(lbda.reshape(-1,1)*labels)).reshape(100,100)

plot_results('Kernel SVM, non-linsep case, C=1', gridx, gridy, mesh, vec1_non_linsep, vec2_non_linsep, support)


weights, intercept, support, lbda = SVM_linsep(vec_non_linsep, labels,rbf,C = 10)

mesh = (rbf(grid,vec_non_linsep.T)@(lbda.reshape(-1,1)*labels)).reshape(100,100)

plot_results('Kernel SVM, non-linsep case, C=10', gridx, gridy, mesh, vec1_non_linsep, vec2_non_linsep, support)


noisy_moons,moon_labels = make_moons(n_samples=100, noise=0.05)
moon_labels = (moon_labels*2-1).reshape(-1,1).astype(np.float64)
plt.figure(dpi=600)
plt.scatter(noisy_moons[:,0],noisy_moons[:,1])

weights, intercept, support, lbda = SVM_linsep(noisy_moons, moon_labels,rbf)

mesh = (rbf(grid,noisy_moons.T)@(lbda.reshape(-1,1)*moon_labels)).reshape(100,100)

plt.figure(dpi=600)
plt.title('make moons example with RBF kernel')
plt.contourf(gridx,gridy,mesh,alpha=0.5,cmap='seismic')
plt.scatter(noisy_moons[(moon_labels==-1).ravel(),0],noisy_moons[(moon_labels==-1).ravel(),1],marker='+')
plt.scatter(noisy_moons[(moon_labels==1).ravel(),0],noisy_moons[(moon_labels==1).ravel(),1],marker='*')
plt.scatter(support[:,0],support[:,1],alpha=0.5,color='blue')

noisy_circles,circle_labels = make_circles(n_samples=100, noise=0.05,factor=.2)
circle_labels = (circle_labels*2-1).reshape(-1,1).astype(np.float64)
plt.figure(dpi=600)
plt.scatter(noisy_circles[:,0],noisy_circles[:,1])

weights, intercept, support, lbda = SVM_linsep(noisy_circles, circle_labels,rbf)

mesh = (rbf(grid,noisy_circles.T)@(lbda.reshape(-1,1)*circle_labels)).reshape(100,100)

plt.figure(dpi=600)
plt.title('make_circles example with RBF kernel')
plt.contourf(gridx,gridy,mesh,alpha=0.5,cmap='seismic')
plt.scatter(noisy_circles[(circle_labels==-1).ravel(),0],noisy_circles[(circle_labels==-1).ravel(),1],marker='+')
plt.scatter(noisy_circles[(circle_labels==1).ravel(),0],noisy_circles[(circle_labels==1).ravel(),1],marker='*')
plt.scatter(support[:,0],support[:,1],alpha=0.5,color='blue')

svc_linear = SVC(kernel='linear')
svc_linear.fit(vec_linsep,labels)
weights=svc_linear.coef_
support = svc_linear.support_vectors_
mesh = (grid@weights.T).reshape(100,100)

plot_results('Sklearn\'s Linear SVM, linsep case', gridx, gridy, mesh, vec1_linsep, vec2_linsep, support)

svc_linear = SVC(kernel='linear',C=0.1)
svc_linear.fit(vec_non_linsep,labels)
weights=svc_linear.coef_
support = svc_linear.support_vectors_
mesh = (grid@weights.T).reshape(100,100)

plot_results('Sklearn\'s Linear SVM, non-linsep case, C=0.1', gridx, gridy, mesh, vec1_non_linsep, vec2_non_linsep, support)


svc_linear = SVC(kernel='linear',C=1)
svc_linear.fit(vec_non_linsep,labels)
weights=svc_linear.coef_
support = svc_linear.support_vectors_
mesh = (grid@weights.T).reshape(100,100)

plot_results('Sklearn\'s Linear SVM, non-linsep case, C=1', gridx, gridy, mesh, vec1_non_linsep, vec2_non_linsep, support)


svc_linear = SVC(kernel='linear',C=10)
svc_linear.fit(vec_non_linsep,labels)
weights=svc_linear.coef_
support = svc_linear.support_vectors_
mesh = (grid@weights.T).reshape(100,100)

plot_results('Sklearn\'s Linear SVM, non-linsep case, C=10', gridx, gridy, mesh, vec1_non_linsep, vec2_non_linsep, support)


svc_kernel = SVC(kernel='rbf',C=0.1)
svc_kernel.fit(vec_non_linsep,labels)
support = svc_kernel.support_vectors_
mesh = svc_kernel.decision_function(np.c_[gridx.ravel(), gridy.ravel()]).reshape(100,100)

plot_results('Sklearn\'s Kernel SVM, non-linsep case, C=0.1', gridx, gridy, mesh, vec1_non_linsep, vec2_non_linsep, support)


svc_kernel = SVC(kernel='rbf',C=1)
svc_kernel.fit(vec_non_linsep,labels)
support = svc_kernel.support_vectors_
mesh = svc_kernel.decision_function(np.c_[gridx.ravel(), gridy.ravel()]).reshape(100,100)

plot_results('Sklearn\'s Kernel SVM, non-linsep case, C=1', gridx, gridy, mesh, vec1_non_linsep, vec2_non_linsep, support)


svc_kernel = SVC(kernel='rbf',C=10)
svc_kernel.fit(vec_non_linsep,labels)
support = svc_kernel.support_vectors_
mesh = svc_kernel.decision_function(np.c_[gridx.ravel(), gridy.ravel()]).reshape(100,100)

plot_results('Sklearn\'s Kernel SVM, non-linsep case, C=10', gridx, gridy, mesh, vec1_non_linsep, vec2_non_linsep, support)



