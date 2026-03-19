#!/usr/bin/env python
# coding: utf-8

import numpy as np

#################### Function to perform PCA

def PCA(X,m):
    # X is matrix each column is an observation of length k and there are l observations (assumed to be flattened)
    # find largest m eigen vectors of cov matrix
    l,N = X.shape
    mean = X.mean(axis = 1, keepdims = True)
    delta = X - mean
    #cov matrix is delta delta transpose, and we need to find its eigen values
    #the spectrum values of delta transpose delta is the same
    #so use one of the two based on the smaller dimension
    sigma = delta @ delta.T
    #eigh is for symm matrices and sorts eigen values in ascending order
    eig_values,eig_vectors = np.linalg.eigh(sigma)
    A = eig_vectors[:,-m:]
    return A.T
            
    





####################### Function to perform MDA

def MDA(X, m, M):
    # returns the projection matrix, class means and shared covariance matrix 
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
    anchor = class_mean@prior
    #between class scatter
    delta = class_mean - anchor
    sigma_b = delta@np.diag(prior.flatten())@delta.T

    #number 
    #within class scatter
    prior = prior.flatten()
    sigma_w = np.zeros((k,k))
    for i in range(M):
        X_i = X[:, a*i: a*(i+1)]
        delta = X_i - class_mean[:,i].reshape(-1,1)
        sigma_w = sigma_w + prior[i]*delta@delta.T/a
        
    
    B = np.linalg.inv(sigma_w)@sigma_b
    eig_values, eig_vectors = np.linalg.eig(B)
    sort_indices = np.argsort(eig_values)[::-1]
    largest_m_eigenvalues = eig_values[sort_indices[:m]]
    largest_m_eigenvectors = eig_vectors[:, sort_indices[:m]]

    A = largest_m_eigenvectors.T
    return A, class_mean,sigma_w


    


