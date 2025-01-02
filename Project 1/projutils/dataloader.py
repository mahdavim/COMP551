# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 13:01:03 2023

@author: home
"""
#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

class reg_dat: 
    '''
    Regression data class. The class loads the data, calculates the statistical summaries, 
    prints them, and saves them if needed. The split function splits the data into train-test
    parts based on a given ratio. If the 'normalize' method is raised, z-score normalized 
    versions of train and test set are created and stored in the class.
    '''
    def __init__(self, pth):
        self.dat     = pd.read_excel(pth)
        self.n       = self.dat.shape[0]
        self.x_train = None
        self.y_train = None
        self.x_test  = None     
        self.y_test  = None
             
    def stat_sum(self, prnt, save): # Provide statistical summary of the data
        stat_sum = pd.DataFrame(self.dat.describe())
        cat_1 = (self.dat['X8'].value_counts()/self.n) * 100 #Categorical data
        cat_2 = (self.dat['X6'].value_counts()/self.n) * 100
        cat_3 = (self.dat['X4'].value_counts()/self.n) * 100
        cat_4 = (self.dat['X5'].value_counts()/self.n) * 100
        cat_5 = (self.dat['X7'].value_counts()/self.n) * 100
        all_cat = pd.concat([cat_1, cat_2, cat_3, cat_4, cat_5], axis=1)
        if prnt:
            print(stat_sum)
            print('Categorical data:')
            print(all_cat)
        if save: 
            stat_sum.to_excel('statistical_summary.xlsx')
            all_cat.to_excel('categorical_statistical_summary.xlsx')
           
    def split(self, tst_ratio, outcm=9, rnd_stat=100):        
        # Split into train-test sets. Note that the outcomes are Y1 (8), Y2 (9)
        if not( outcm == 9 or outcm ==8):
            print('Warning: You're choosing an independent variable as an outcome!')
            
        x_train, x_test, y_train, y_test = train_test_split(self.dat.iloc[:, 0:8],
                                                               self.dat.iloc[:, outcm],
                                                               test_size=tst_ratio,
                                                               random_state=rnd_stat)
        self.x_train = x_train.values
        self.x_test  = x_test.values
        self.y_train = y_train.values
        self.y_test  = y_test.values

    def dum_coder(self, train, test, cat_dat=[5,7]):
        '''
        inputs:
            train: training set data
            test: test set data
            cat_dat: column index of categorical data
            
        stores:
            self.dum_x_train: training data with one hot encoding
            self.dum_x_test: test data with one hot encoding
        '''
        enc = OneHotEncoder(drop=None, sparse=False)
        
        x_train = np.delete(train, cat_dat, 1)    
        x_test  = np.delete(test, cat_dat, 1)  
        for cat in cat_dat:
            tmp_dat_trn = enc.fit_transform(train[:,cat].reshape(-1,1))
            tmp_dat_tst = enc.fit_transform(test[:,cat].reshape(-1,1))
            x_train = np.concatenate((x_train, tmp_dat_trn), axis=1)
            x_test  = np.concatenate((x_test, tmp_dat_tst), axis=1)
        
        self.dum_x_train = x_train
        self.dum_x_test  = x_test
            
        
    def normalize(self): 
        '''
        Perform z-score normalization
        
        stores:
            self.nx_train: normalized training set
            self.nx_test: normalized test set
        '''
           
        sclr = StandardScaler()
        self.nx_train = sclr.fit_transform(self.x_train)
        self.nx_test  = sclr.transform(self.x_test)
           
    def normalize_mn(self):
        sclr = MinMaxScaler()
        self.nnx_train = sclr.fit_transform(self.x_train)
        self.nnx_test  = sclr.transform(self.x_test)
              