# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')  

#Change project path if you run on another system
proj_pth = 'D:\\Program Files\\OneDrive - McGill University\\Work and studies\\Courses\\Project 1'
os.chdir(proj_pth)
# Local project python files
from projutils.Regression import LinearReg, GradDescent
from projutils.utils import reg_eval, gradient, momntm
from projutils.dataloader import reg_dat

pth = 'D:\\ENB2012_data.xlsx' # Data path. Change on other systems



#%% Reported performances
# 1 Unnormalized input and no regularization

linrg = LinearReg(alpha=0)

rg_dat = reg_dat(pth)
rg_dat.split(tst_ratio=0.2, outcm=8, rnd_stat=100) # Split into train test sets

linrg.fit_ls(rg_dat.x_train, rg_dat.y_train)
    
y_prd = linrg.predict_ls(rg_dat.x_test)
y_prd_train = linrg.predict_ls(rg_dat.x_train)
    
mse_test_ls, r2_test_ls = reg_eval(rg_dat.y_test, y_prd, prnt=False)
mse_train_ls, r2_train_ls = reg_eval(rg_dat.y_train, y_prd_train, prnt=False)

#============== Grad descent
# Warning: unnormalized data will cause the gradient to be unstable since data scales
# are significantly large. In case of too large error, decrease learning rate (e.g., 000005) 
# and add  regularization. Unnormalized data usually results in decreased performance

optmz = GradDescent(learning_rate=.000005, max_iters=1e5, epsilon=1e-4,
             btch_sz=rg_dat.x_train.shape[0]-1, alpha=0.1, beta=0,mn_beta=0, tol=5)

linrg.fit_grd(rg_dat.x_train, rg_dat.y_train[:, None], optmz, gradient, momntm)

y_pred = linrg.predict_grd(rg_dat.x_test)# Predict

y_prd_tr = linrg.predict_grd(rg_dat.x_train)

mse_test_grd, r2_test_grd = reg_eval(rg_dat.y_test, y_pred, prnt=False)
mse_train_grd, r2_train_grd = reg_eval(rg_dat.y_train, y_prd_tr, prnt=False)

#%% 2
# Normalized input, no regularization, and no one hot encoding

#================= LS===============
linrg = LinearReg(alpha=0)

rg_dat = reg_dat(pth)
rg_dat.split(tst_ratio=0.2, outcm=8, rnd_stat=100)
rg_dat.normalize() # Create z-score normalized training and set data

linrg.fit_ls(rg_dat.nx_train, rg_dat.y_train)
    
y_prd = linrg.predict_ls(rg_dat.nx_test)
y_prd_train = linrg.predict_ls(rg_dat.nx_train)
    
mse_test_ls, r2_test_ls = reg_eval(rg_dat.y_test, y_prd, prnt=False)
mse_train_ls, r2_train_ls = reg_eval(rg_dat.y_train, y_prd_train, prnt=False)


#=============== Grad
optmz = GradDescent(learning_rate=.005, max_iters=1e5, epsilon=1e-4,
             btch_sz=rg_dat.nx_train.shape[0]-1, alpha=0, beta=0,mn_beta=0, tol=5)

linrg.fit_grd(rg_dat.nx_train, rg_dat.y_train[:, None], optmz, gradient, momntm)

y_pred = linrg.predict_grd(rg_dat.nx_test)# Predict

y_prd_tr = linrg.predict_grd(rg_dat.nx_train)

mse_test_grd, r2_test_grd = reg_eval(rg_dat.y_test, y_pred, prnt=False)
mse_train_grd, r2_train_grd = reg_eval(rg_dat.y_train, y_prd_tr, prnt=False)

pd.DataFrame([mse_test_ls, r2_test_ls]). to_excel('test_perf_norm_noreg.xlsx')
pd.DataFrame([mse_train_ls, r2_train_ls]). to_excel('train_perf_norm_noreg.xlsx')
pd.DataFrame(linrg.w).to_excel('LS_weight_norm_noreg.xlsx')

pd.DataFrame([mse_test_grd, r2_test_grd]). to_excel('grd_test_perf_norm_noreg.xlsx')
pd.DataFrame([mse_train_grd, r2_train_grd]). to_excel('grd_train_perf_norm_noreg.xlsx')
pd.DataFrame(linrg.w_grd).to_excel('grd_weight_norm_noreg.xlsx')

#%% 3
# Normalized input + L2 regularization + no one hot
rg_dat = reg_dat(pth)
rg_dat.split(tst_ratio=0.2, outcm=8, rnd_stat=100)
rg_dat.normalize()

#================= LS===============
linrg = LinearReg(alpha=0.01)
linrg.fit_ls(rg_dat.nx_train, rg_dat.y_train)
    
y_prd = linrg.predict_ls(rg_dat.nx_test)
y_prd_train = linrg.predict_ls(rg_dat.nx_train)
    
mse_test_ls, r2_test_ls = reg_eval(rg_dat.y_test, y_prd, prnt=False)
mse_train_ls, r2_train_ls = reg_eval(rg_dat.y_train, y_prd_train, prnt=False)

pd.DataFrame([mse_test_ls, r2_test_ls]). to_excel('test_perf_norm_l2reg.xlsx')
pd.DataFrame([mse_train_ls, r2_train_ls]). to_excel('train_perf_norm_l2reg.xlsx')
pd.DataFrame(linrg.w).to_excel('LS_weight_norm_l2reg.xlsx')



#=============== Grad
optmz = GradDescent(learning_rate=.005, max_iters=1e5, epsilon=1e-4,
             btch_sz=rg_dat.nx_train.shape[0]-1, alpha=0.01, beta=0, tol=5,
             mn_beta=0)

linrg.fit_grd(rg_dat.nx_train, rg_dat.y_train[:, None], optmz, gradient, momntm)


y_pred = linrg.predict_grd(rg_dat.nx_test)# Predict

y_prd_tr = linrg.predict_grd(rg_dat.nx_train)

mse_test_grd, r2_test_grd = reg_eval(rg_dat.y_test, y_pred, prnt=False)
mse_train_grd, r2_train_grd = reg_eval(rg_dat.y_train, y_prd_tr, prnt=False)

pd.DataFrame([mse_test_grd, r2_test_grd]). to_excel('grd_test_perf_norm_l2reg.xlsx')
pd.DataFrame([mse_train_grd, r2_train_grd]). to_excel('grd_train_perf_norm_l2reg.xlsx')
pd.DataFrame(linrg.w_grd).to_excel('grd_weight_norm_l2reg.xlsx')

#%% 4
# Normalized input, L2 regularization, and one hot encoding

rg_dat = reg_dat(pth)
rg_dat.split(tst_ratio=0.2, outcm=8, rnd_stat=100)
rg_dat.normalize()
rg_dat.dum_coder(rg_dat.nx_train, rg_dat.nx_test, cat_dat=[5,7]) #store one hot encoded data

#================= LS===============
linrg = LinearReg(alpha=0.01)

linrg.fit_ls(rg_dat.dum_x_train, rg_dat.y_train)
    
y_prd = linrg.predict_ls(rg_dat.dum_x_test)
y_prd_train = linrg.predict_ls(rg_dat.dum_x_train)
    
mse_test_ls, r2_test_ls = reg_eval(rg_dat.y_test, y_prd, prnt=False)
mse_train_ls, r2_train_ls = reg_eval(rg_dat.y_train, y_prd_train, prnt=False)

pd.DataFrame([mse_test_ls, r2_test_ls]). to_excel('test_perf_norm_l2reg_oh.xlsx')
pd.DataFrame([mse_train_ls, r2_train_ls]). to_excel('train_perf_norm_l2reg_oh.xlsx')
pd.DataFrame(linrg.w).to_excel('LS_weight_norm_l2reg_oh.xlsx')


#=============== Grad
optmz = GradDescent(learning_rate=.005, max_iters=1e5, epsilon=1e-4,
             btch_sz=rg_dat.nx_train.shape[0]-1, alpha=0.01, beta=0, tol=5, 
             mn_beta=0)

linrg.fit_grd(rg_dat.dum_x_train, rg_dat.y_train[:, None], optmz, gradient, momntm)

y_pred = linrg.predict_grd(rg_dat.dum_x_test)# Predict

y_prd_tr = linrg.predict_grd(rg_dat.dum_x_train)

mse_test_grd, r2_test_grd = reg_eval(rg_dat.y_test, y_pred, prnt=False)
mse_train_grd, r2_train_grd = reg_eval(rg_dat.y_train, y_prd_tr, prnt=False)

pd.DataFrame([mse_test_grd, r2_test_grd]). to_excel('grd_test_perf_norm_l2reg_oh.xlsx')
pd.DataFrame([mse_train_grd, r2_train_grd]). to_excel('grd_train_perf_norm_l2reg_oh.xlsx')
pd.DataFrame(linrg.w_grd).to_excel('grd_weight_norm_l2reg_oh.xlsx')



#%% 5 Investigate feature importance via L1 regularization
linrg = LinearReg(alpha=0.01)
rg_dat = reg_dat(pth)
rg_dat.split(tst_ratio=0.2, outcm=8, rnd_stat=100)
rg_dat.normalize()


optmz = GradDescent(learning_rate=.005, max_iters=1e4, epsilon=1e-4,
             btch_sz=rg_dat.nx_train.shape[0]-1, alpha=0, beta=0.1, tol=5, 
             mn_beta=0.9)

linrg.fit_grd(rg_dat.nx_train, rg_dat.y_train[:, None], optmz, gradient, momntm)


y_pred = linrg.predict_grd(rg_dat.nx_test)# Predict
y_prd_tr = linrg.predict_grd(rg_dat.nx_train)

mse_test_grd, r2_test_grd = reg_eval(rg_dat.y_test, y_pred, prnt=False)
mse_train_grd, r2_train_grd = reg_eval(rg_dat.y_train, y_prd_tr, prnt=False)

#pd.DataFrame([mse_test_grd, r2_test_grd]). to_excel('grd_test_perf_norm_l2reg.xlsx')
#pd.DataFrame([mse_train_grd, r2_train_grd]). to_excel('grd_train_perf_norm_l2reg.xlsx')
#pd.DataFrame(linrg.w_grd).to_excel('grd_weight_norm_l2reg.xlsx')


fig, ax = plt.subplots()
plt.stem(linrg.w_grd[:-1])
ax.set_xticks(np.arange(0,8))
ax.set_xticklabels((rg_dat.dat.columns[:-2])) 
plt.xlabel('Features')
plt.ylabel('Weight')
plt.savefig('l1 weight', dpi=150)

#%% 6 Use different training sizes and see model performances

# Report MSE as performance marker

trn_pool = np.arange(2,9) * 10

train_ls_los = []
test_ls_los  = []
train_grd_los = []
test_grd_los = []

train_ls_r2 = []
test_ls_r2  = []
train_grd_r2 = []
test_grd_r2 = []
for sz in trn_pool:
    tst_ratio = (100-sz)/100
    
    linrg = LinearReg(alpha=0.01)

    rg_dat = reg_dat(pth)
    rg_dat.split(tst_ratio=tst_ratio, outcm=8, rnd_stat=None)
    rg_dat.normalize()
    rg_dat.dum_coder(rg_dat.nx_train, rg_dat.nx_test, cat_dat=[5,7])

    linrg.fit_ls(rg_dat.dum_x_train, rg_dat.y_train)
        
    y_prd = linrg.predict_ls(rg_dat.dum_x_test)
    y_prd_train = linrg.predict_ls(rg_dat.dum_x_train)
        
    mse_test_ls, r2_test_ls = reg_eval(rg_dat.y_test, y_prd, prnt=False)
    test_ls_los.append(mse_test_ls)
    test_ls_r2.append(r2_test_ls)
    mse_train_ls, r2_train_ls = reg_eval(rg_dat.y_train, y_prd_train, prnt=False)
    train_ls_los.append(mse_train_ls)
    train_ls_r2.append(r2_train_ls)


    #=============== Grad
    optmz = GradDescent(learning_rate=.005, max_iters=1e5, epsilon=1e-3,
                 btch_sz=rg_dat.nx_train.shape[0]-1, alpha=0.01,
                 beta=0, tol=5, mn_beta=0)

    linrg.fit_grd(rg_dat.dum_x_train, rg_dat.y_train[:, None], optmz, gradient, momntm)

    y_pred = linrg.predict_grd(rg_dat.dum_x_test)# Predict

    y_prd_tr = linrg.predict_grd(rg_dat.dum_x_train)

    mse_test_grd, r2_test_grd = reg_eval(rg_dat.y_test, y_pred, prnt=False)
    test_grd_los.append(mse_test_grd)
    test_grd_r2.append(r2_test_grd)
    mse_train_grd, r2_train_grd = reg_eval(rg_dat.y_train, y_prd_tr, prnt=False)
    train_grd_los.append(mse_train_grd)
    train_grd_r2.append(r2_train_grd)










fig, ax = plt.subplots()
plt.plot(train_ls_los, '-o')
plt.plot(train_grd_los, '-o')
ax.set_xticks(np.arange(0,len(trn_pool)))
ax.set_xticklabels(trn_pool) 
ax.spines[['right', 'top']].set_visible(False)
ax.set_ylabel('MSE')
ax.set_xlabel('Training set ratio (%)')
ax.legend(['Ls', 'Grad_desceent'])
plt.savefig('Training_loss_ratio', dpi=150)

fig, ax = plt.subplots()
plt.plot(test_ls_los, '-o')
plt.plot(test_grd_los, '-o')
ax.set_xticks(np.arange(0,len(trn_pool)))
ax.set_xticklabels(trn_pool) 
ax.spines[['right', 'top']].set_visible(False)
ax.set_ylabel('MSE')
ax.set_xlabel('Training set ratio (%)')
ax.legend(['Ls', 'Grad_desceent'])
plt.savefig('Test_loss_ratio', dpi=150)



fig, ax = plt.subplots()
plt.plot(train_ls_r2, '-o')
plt.plot(train_grd_r2, '-o')
ax.set_xticks(np.arange(0,len(trn_pool)))
ax.set_xticklabels(trn_pool) 
ax.spines[['right', 'top']].set_visible(False)
ax.set_ylabel('MSE')
ax.set_xlabel('Training set ratio (%)')
ax.legend(['Ls', 'Grad_desceent'])
plt.savefig('Training_loss_ratio', dpi=150)

fig, ax = plt.subplots()
plt.plot(test_ls_r2, '-o')
plt.plot(test_grd_r2, '-o')
ax.set_xticks(np.arange(0,len(trn_pool)))
ax.set_xticklabels(trn_pool) 
ax.spines[['right', 'top']].set_visible(False)
ax.set_ylabel('MSE')
ax.set_xlabel('Training set ratio (%)')
ax.legend(['Ls', 'Grad_desceent'])
plt.savefig('Test_loss_ratio', dpi=150)


#%% 7 Try different batch sizes + full batch

btch_pool = 2 ** (np.arange(3, 8))
btch_pool = np.hstack((btch_pool, rg_dat.nx_train.shape[0]-1))

train_btch_perf = []
test_btch_perf = []
nitr = 100000
train_al_los = np.zeros((nitr, len(btch_pool)))

for idx, btch in enumerate(btch_pool):
    
    linrg = LinearReg(alpha=0.1)

    rg_dat = reg_dat(pth)
    rg_dat.split(tst_ratio=tst_ratio, outcm=8, rnd_stat=100)
    rg_dat.normalize()
    rg_dat.dum_coder(rg_dat.nx_train, rg_dat.nx_test, cat_dat=[5,7])


    #=============== Grad
    optmz = GradDescent(learning_rate=.001, max_iters=nitr, epsilon=1e-4,
                 btch_sz=btch, alpha=0.01, beta=0, tol=5, mn_beta=0.9)

    linrg.fit_grd(rg_dat.dum_x_train, rg_dat.y_train[:, None],
                  optmz, gradient, momntm)

    y_pred = linrg.predict_grd(rg_dat.dum_x_test)# Predict

    y_prd_tr = linrg.predict_grd(rg_dat.dum_x_train)

    mse_test_grd, _ = reg_eval(rg_dat.y_test, y_pred, prnt=False)
    test_btch_perf.append(mse_test_grd)
    mse_train_grd, _ = reg_eval(rg_dat.y_train, y_prd_tr, prnt=False)
    train_btch_perf.append(mse_train_grd)
    
    train_al_los[:, idx] = linrg.tmp_ls



fig, ax = plt.subplots()
plt.plot(train_al_los)
ax.spines[['right', 'top']].set_visible(False)
ax.set_ylabel('MSE')
ax.set_xlabel('Iteration')
ax.legend(btch_pool)
plt.xlim([0, 3000])
plt.savefig('iteration performance batch sizes train', dpi=150)



fig, ax = plt.subplots()
plt.plot(test_btch_perf, '-o')
ax.set_xticks(np.arange(0,len(btch_pool)))
ax.set_xticklabels(btch_pool)
ax.spines[['right', 'top']].set_visible(False)
ax.set_ylabel('MSE')
ax.set_xlabel('Batch size')

plt.savefig('final performance batch sizes test', dpi=150)

fig, ax = plt.subplots()
plt.plot(train_btch_perf, '-o')
ax.set_xticks(np.arange(0,len(btch_pool)))
ax.set_xticklabels(btch_pool)
ax.spines[['right', 'top']].set_visible(False)
ax.set_ylabel('MSE')
ax.set_xlabel('Batch size')


#%% 8 Perform gradient descent with different learning rates

lr_pool = [0.0001, 0.001, 0.01, 0.1, 1]

train_lr_perf = []
test_lr_perf = []
nitr = 100000
train_lr_los = np.zeros((nitr, len(btch_pool)))

for idx, lr in enumerate(lr_pool):   

    rg_dat = reg_dat(pth)
    rg_dat.split(tst_ratio=tst_ratio, outcm=8, rnd_stat=100)
    rg_dat.normalize()
    rg_dat.dum_coder(rg_dat.nx_train, rg_dat.nx_test, cat_dat=[5,7])

        
    linrg = LinearReg(alpha=0.01)
    #=============== Grad
    optmz = GradDescent(learning_rate=lr, max_iters=nitr, epsilon=1e-4,
                 btch_sz=64, alpha=0.01, beta=0, tol=5, 
                 mn_beta=0)

    linrg.fit_grd(rg_dat.dum_x_train, rg_dat.y_train[:, None], optmz, gradient, momntm)


    y_pred = linrg.predict_grd(rg_dat.dum_x_test)# Predict

    y_prd_tr = linrg.predict_grd(rg_dat.dum_x_train)

    mse_test_grd, _ = reg_eval(rg_dat.y_test, y_pred, prnt=False)
    test_lr_perf.append(mse_test_grd)
    mse_train_grd, _ = reg_eval(rg_dat.y_train, y_prd_tr, prnt=False)
    train_lr_perf.append(mse_train_grd)
    
    train_lr_los[:, idx] = linrg.tmp_ls



fig, ax = plt.subplots()
plt.plot(train_lr_los)
ax.spines[['right', 'top']].set_visible(False)
ax.set_ylabel('MSE')
ax.set_xlabel('Iteration')
ax.legend(lr_pool)
plt.xlim([0, 10000])
plt.savefig('iteration performance learning rates', dpi=150)




fig, ax = plt.subplots()
plt.plot(test_lr_perf, '-o')
ax.set_xticks(np.arange(0,len(lr_pool)))
ax.set_xticklabels(lr_pool)
ax.spines[['right', 'top']].set_visible(False)
ax.set_ylabel('MSE')
ax.set_xlabel('Learning rate')
plt.savefig('final performance learning rates', dpi=150)




#%% 9 Compare analytical linear regression solution with mini-batch stochastic gradient descent
#!!! Alpha zero here

linrg = LinearReg(alpha=0.01)
rg_dat = reg_dat(pth)
rg_dat.split(tst_ratio=0.2, outcm=8, rnd_stat=100)
rg_dat.normalize()
rg_dat.dum_coder(rg_dat.nx_train, rg_dat.nx_test, cat_dat=[5,7])

linrg.fit_ls(rg_dat.dum_x_train, rg_dat.y_train)
    
y_prd = linrg.predict_ls(rg_dat.dum_x_test)
y_prd_train = linrg.predict_ls(rg_dat.dum_x_train)
    
mse_test_ls, r2_test_ls = reg_eval(rg_dat.y_test, y_prd, prnt=False)
mse_train_ls, r2_train_ls = reg_eval(rg_dat.y_train, y_prd_train, prnt=False)



#=============== Grad
optmz = GradDescent(learning_rate=.005, max_iters=1e5, epsilon=1e-4,
             btch_sz=64, alpha=0.01, beta=0, tol=5, mn_beta=0)

linrg.fit_grd(rg_dat.dum_x_train, rg_dat.y_train[:, None], optmz, gradient, momntm)


y_pred = linrg.predict_grd(rg_dat.dum_x_test)# Predict

y_prd_tr = linrg.predict_grd(rg_dat.dum_x_train)

mse_test_grd, r2_test_grd = reg_eval(rg_dat.y_test, y_pred, prnt=False)
mse_train_grd, r2_train_grd = reg_eval(rg_dat.y_train, y_prd_tr, prnt=False)

pd.DataFrame([mse_test_ls, r2_test_ls]). to_excel('test_perf_ls_noreg.xlsx')
pd.DataFrame([mse_train_ls, r2_train_ls]). to_excel('train_perf_ls_noreg.xlsx')
pd.DataFrame(linrg.w).to_excel('LS_weight_ls_noreg.xlsx')

pd.DataFrame([mse_test_grd, r2_test_grd]). to_excel('minigrd_test_perf_noreg.xlsx')
pd.DataFrame([mse_train_grd, r2_train_grd]). to_excel('minigrd_train_perf_noreg.xlsx')
pd.DataFrame(linrg.w_grd).to_excel('minigrd_weight_noreg.xlsx')


#%%    Compare with momentum and without momentum models
# Note! Change weight initialization in the Regression file to 1000* random initialization in 

linrg_m = LinearReg(alpha=0)

rg_dat = reg_dat(pth)
rg_dat.split(tst_ratio=0.2, outcm=9, rnd_stat=125)
rg_dat.normalize()


optmz = GradDescent(learning_rate=.001, max_iters=20000, epsilon=1e-4,
             btch_sz=4, alpha=0.1, beta=0,mn_beta = 0.9, tol=5)

linrg_m.fit_grd(rg_dat.nx_train, rg_dat.y_train[:, None], optmz, gradient, momntm)


y_pred = linrg.predict_grd(rg_dat.nx_test)# Predict

y_prd_tr = linrg.predict_grd(rg_dat.nx_train)

mse_test_grd, r2_test_grd = reg_eval(rg_dat.y_test, y_pred, prnt=False)
mse_train_grd, r2_train_grd = reg_eval(rg_dat.y_train, y_prd_tr, prnt=False)


linrg_v = LinearReg(alpha=0)


optmz = GradDescent(learning_rate=.001, max_iters=20000, epsilon=1e-4,
             btch_sz=4, alpha=0.1, beta=0,mn_beta = 0, tol=5)

linrg_v.fit_grd(rg_dat.nx_train, rg_dat.y_train[:, None], optmz, gradient, momntm)


y_pred = linrg.predict_grd(rg_dat.nx_test) # Predict

y_prd_tr = linrg.predict_grd(rg_dat.nx_train)

mse_test_grd, r2_test_grd = reg_eval(rg_dat.y_test, y_pred, prnt=False)
mse_train_grd, r2_train_grd = reg_eval(rg_dat.y_train, y_prd_tr, prnt=False)

plt.plot(linrg_m.tmp_ls)
plt.plot(linrg_v.tmp_ls)
