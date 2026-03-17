#!/usr/bin/env python
# coding: utf-8

# # Function to perform MDA

# In[ ]:


import numpy as np

def MDA(X, m, M):
    #numpy required
    #m - number of projected dimensions
    #M - number of classes
    #assuming equiprior
    #dataset - grouped by class
    # Find within class and between class scatter matrices, and find eigen vectors corresponding to the m largest eigen values of sigma_w_inv X sigma_b
    #find class means
    #prior taken to be equiprobable for the given datasets
    prior = (1/M)*np.ones((M,1))
    k,l = X.shape
    #a is the number of elements within class
    a = l//M
    class_mean = X.reshape(k,M,a).mean(axis = 2)
    print(class_mean.shape)
    anchor = class_mean@prior
    
    #between class scatter
    delta = class_mean - anchor
    sigma_b = delta@np.diag(prior.flatten())@delta.T
    print(sigma_b.shape)

    #number 
    #within class scatter
    prior = prior.flatten()
    sigma_w = np.zeros((k,k))
    for i in range(M):
        X_i = X[:, a*i: a*(i+1)]
        delta = X_i - class_mean[:,i].reshape(-1,1)
        sigma_w = sigma_w + prior[i]*delta@delta.T
        
    print(sigma_w.shape)

    B = np.linalg.inv(sigma_w)@sigma_b
    eig_values, eig_vectors = np.linalg.eig(B)
    sort_indices = np.argsort(eigenvalues)[::-1]
    largest_m_eigenvalues = eig_values[sort_indices[:m]]
    largest_m_eigenvectors = eig_values[:, sort_indices[:m]]

    A = largest_m_eigenvectors.T
    return A


    


