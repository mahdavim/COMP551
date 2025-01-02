# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 13:00:38 2023

@author: home
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Change normalization
#%%
class GradDescent:
    
    def __init__(self, learning_rate=.001, max_iters=1e4, epsilon=1e-8,
                 btch_sz=32, tol=5, alpha=0.01, beta=0, mn_beta=0.9,
                 record_history=False):
        '''      

        Parameters
        ----------
        learning_rate : float. Model learning rate
            
        max_iters : int, maximum number of iterations to perform gradient descent
        
        epsilon : float, an upper bound for gradient value. If gradient's norm remains
        lower than this for tol iterations, break the training loop
        
        btch_sz : int, size of input batch. change to training size-1 for full batch
        tol : int, number of trials to tolerate a low gradient 
        
        alpha : float, L2 regularization strength
        
        beta : float, L1 regularization strength
        
        mn_beta : float, Beta value for momentum implementation. 0 will result in vanilla gradient descent
        record_history : boolean
        -------
        '''
        
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.record_history = record_history
        self.epsilon = epsilon
        self.btch_sz = btch_sz
        self.alpha = alpha
        self.beta = beta
        self.tol = tol # Tolerance
        self.mn_beta = mn_beta # Momentum term
        if record_history:
            self.w_history = []                 # To store the weight history for visualization
            
    

    def run(self, gradient_fn,moment_fn, x, y, w):
    
        grad = np.inf
        tol  = 0
        btch = 0
        N, D = x.shape
        btch_num = np.floor(N/self.btch_sz)
        prev_mnt = np.ones((D,1))
        #------------
        tmp_ls = []
        #-----------
        for ind in range(int(self.max_iters)):
            if ind%5000 == 0: # print state
                print(f'Iteration: {ind}')
                
            if np.linalg.norm(grad) < self.epsilon: # If grad goes below the epsilon
                if tol >= self.tol: # If the grad had small changes for tol iterations
                    break
                else:
                    tol+=1
            
            if btch == 0:    # Batch initiation
                tmp_ind = np.random.permutation(np.arange(N))

            if btch==btch_num:
                slc_ind = tmp_ind[(btch)*self.btch_sz:]
                x_bt = x[slc_ind, :]
                y_bt = y[slc_ind,:]

                btch = 0 # Reset batch tracking index
            else:
                slc_ind = tmp_ind[(btch)*self.btch_sz:(btch+1)*self.btch_sz]
                x_bt = x[slc_ind, :]
                y_bt = y[slc_ind, :]

                btch += 1
            
            grad = gradient_fn(x_bt, y_bt, w, self.alpha, self.beta) 
   
            prev_mnt = moment_fn(grad, prev_mnt, mn_beta=self.mn_beta)
            w = w - self.learning_rate * prev_mnt
            
            # ------------- store loss -----------
            tmp_y = x @ w
            tmp_ls.append(mean_squared_error(y, tmp_y)) # Store loss in each iteration
            #----------------
            if self.record_history:
                self.w_history.append(w)
                
            if tol>0  and np.linalg.norm(grad) > self.epsilon: # Reset tolerance if gradient goes over epsilon
                tol=0 
            
        return w, tmp_ls


#%%



class LinearReg:
    '''
    The linear regression class. The class has performs training and prediction with 
    least square and gradient descent.
    '''
    
    def __init__(self, add_bias=True, alpha=0):
        self.add_bias = add_bias
        self.alpha = alpha
        
    def fit_ls(self, x, y):
        if x.ndim == 1:
            x = x[:, None]                         # Add a dimension for the features
        N, _ = x.shape
        if self.add_bias:
            x = np.column_stack([x,np.ones(N)])    # Add bias by adding a constant feature of value 1
            _, D = x.shape
        #alternatively: self.w = np.linalg.inv(x.T @ x)@x.T@y
        if self.alpha>0:
            self.w = np.linalg.inv(x.T @ x + self.alpha * np.identity(D))@x.T@y
        else: 
            self.w = np.linalg.lstsq(x, y)[0]          # Return w for the least square difference
        return self  
    
    
    def predict_ls(self, x):
        if self.add_bias:
            N = x.shape[0]
            x = np.column_stack([x,np.ones(N)])

        y_pred = x @ self.w                             # Predict the y values
        return y_pred



    def fit_grd(self, x, y, optimizer, gradient, momentum):
        if x.ndim == 1:
            x = x[:, None]                         # Add a dimension for the features
        N, _ = x.shape
        if self.add_bias:
            x = np.column_stack([x,np.ones(N)])
        
        _, D = x.shape
        #w0 = np.zeros((D,1))
        #w0 = 1000* np.random.randn(D,1) # For momentum testing
        w0 = np.random.randn(D,1)
        #w0 = np.ones((D,1)) * 1000
        self.w_grd, self.tmp_ls = optimizer.run(gradient, momentum,  x, y, w0 )
        
      
            
    def predict_grd(self, x):
        if self.add_bias:
            N = x.shape[0]
            x = np.column_stack([x,np.ones(N)])

        y_pred = x @ self.w_grd               # Predict the y values
        return y_pred
        
#%%