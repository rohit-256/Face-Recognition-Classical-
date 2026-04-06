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
    return A.T, mean
            
    





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


    ############################ Function to perform k-NN classification (return k nearest neighbours as well)

def k_NN(X_train, X_test, k):
    #X_train training data l x N_train, first row contains labels
    #X_test testing data l x N_test, first row contains labels
    y_train = X_train[0,:]
    y_test = X_test[0,:]
    X_train = np.delete(X_train, 0, axis=0)
    X_test =  np.delete(X_test, 0, axis=0)
    l,N_test = X_test.shape
    N_train = X_train.shape[1]
    
    distances = np.zeros((N_test,N_train))
    for i in range(N_test):
        delta = X_train - X_test[:,i].reshape(-1,1)
        distances[i,:] = np.sum(delta * delta, axis = 0)
    # get indices of k smallest elements of each row using np.partition

    neighbour_indices = np.argpartition(distances, k, axis = 1)[:,:k]
    #map indices to labels
    neighbours = y_train[neighbour_indices]

    neighbours = neighbours.astype(int)

    nearest = np.array([np.bincount(row).argmax() for row in neighbours])

    return nearest.reshape(1,-1), neighbours
