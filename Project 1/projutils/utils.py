# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 11:15:47 2023

@author: Mahdi Mahdavi
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def reg_eval(y, y_pred, prnt=True):
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    if prnt:
        print(f'MSE: {mse}')
        print(f'R2 score: {r2}')
    return mse, r2
    

def gradient( x, y, w, alpha, beta):
    '''
    Function to calculate the gradient. 
    x: input features
    y: outcomes
    w: weights
    alpha: L2 regularization strength
    beta: L1 regularization strength    
    '''
    y_pred =  x @ w 
    dlt_y = y_pred - y
    N, _ = x.shape
    tmp_w = w.T.copy()  

    grad = (.5*np.dot(dlt_y.T, x)/N) 
    grad[:, :-1] += alpha * tmp_w[:, :-1].copy()
    grad[:, :-1] += beta * np.sign(tmp_w[:, :-1])
    # Delta y is inversed to adjust the dimensions for multiplication
    return grad.T # To convert into (D,1) Format


def momntm(grad, prev_mnt, mn_beta=0.9):
    mmnt = mn_beta * prev_mnt + (1-mn_beta) * grad
    return mmnt
    
    



