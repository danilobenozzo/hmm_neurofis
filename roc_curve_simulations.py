# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 12:35:14 2018

@author: danilo
"""

"""Simulation ROC curve
"""

import numpy as np
from sklearn import metrics
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LinearRegression, BayesianRidge, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pdb

from sklearn.metrics.pairwise import rbf_kernel

from lightning.classification import CDClassifier
from lightning.classification import KernelSVC

from plotting_functions import scatter_plot

np.random.seed(0)

class SparseNonlinearClassifier(CDClassifier):

    def __init__(self, gamma=1e-2, C=1, alpha=1):
        self.gamma = gamma
        super(SparseNonlinearClassifier, self).__init__(C=C,
                                                        alpha=alpha,
                                                        loss='log',#"squared_hinge",
                                                        penalty="l1")

    def fit(self, X, y):
        K = rbf_kernel(X, gamma=self.gamma)
        self.X_train_ = X
        super(SparseNonlinearClassifier, self).fit(K, y)
        return self

    def decision_function(self, X):
        K = rbf_kernel(X, self.X_train_, gamma=self.gamma)
        return super(SparseNonlinearClassifier, self).decision_function(K)


def feature_engineering(Xs, block_normalisation=False):
    feature_space = []
    for X in Xs:
        if block_normalisation :
            print "Block-normalization"
            X = row_normalise(X)#grand_normalise(X)#feature_scaling(X)
        feature_space += [X, np.power(X, 2), np.power(X, 3), np.sign(X) * np.sqrt(np.abs(X))]
        # Feature engineering: all possible products between the original feature values:
        #feature_space.append(np.array([np.multiply.outer(X[i], X[i])[np.triu_indices(X.shape[1], 1)] for i in range(X.shape[0])]))
    return feature_space

def compute_roc_curve(x0_axis, x0_freq, x1_axis, x1_freq, bin_size):
    """Compute the ROC curve from the frequency densities of the two classes
    """
    fpr = []
    tpr = []
    min_tot = np.min([np.min(x0_axis), np.min(x1_axis)]) - bin_size
    max_tot = np.max([np.max(x0_axis), np.max(x1_axis)]) + bin_size
    
    fpr += [np.sum(x0_freq[x0_axis>th_i]) for i_th, th_i in enumerate(np.arange(min_tot,max_tot,bin_size))]
    tpr += [np.sum(x1_freq[x1_axis>th_i]) for i_th, th_i in enumerate(np.arange(min_tot,max_tot,bin_size))]
    
    return fpr, tpr

    
def compute_prob_density(x, bin_size=1, plotting=False):
    """Compute the frequency density
    """
    x_hist = np.arange(x.min(), x.max()+bin_size, bin_size)
    y_bar = []
    y_bar += [np.sum(np.logical_and(x>=bin_i, x<bin_i+bin_size)) for i_bin, bin_i in enumerate(x_hist)]
    y_bar = np.array(y_bar)/float(len(x))
    
    if plotting:
        #import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(x_hist, y_bar, '.-')
        plt.xlabel('spk/s')
        plt.ylabel('%trial')
        plt.show()
    
    return x_hist, y_bar


def compute_spike_density_across_trial(trials, tws=[0,1000], hist_bin=50, step=None, n_bin=None):
    """Receive trials and compute spk/s single trial
    """
    if hist_bin is None and not(n_bin is None):
        compute_hist_bin = True
    elif n_bin is None and not(hist_bin is None):
        compute_hist_bin = False
        if step is None:
            step = hist_bin/4
    else:
        print "Bad input, default values: hist_bin=50, step=50/4"
        compute_hist_bin = False
        hist_bin = 50
        step = hist_bin/4
    try:
        if len(tws[0])>=2 and len(tws)==len(trials):
            print "Size tws coherent"
        else:
            print "Error size tws"
    except TypeError:
        tws_new = []
        tws_new += [tws for _ in trials]
        tws = tws_new
    
    y_bar = []
    x_hist = []
    for i_trial, trial_i in enumerate(trials):
        if compute_hist_bin:
            hist_bin = (tws[i_trial][1]-tws[i_trial][0]+n_bin) / (n_bin/4 + 1)
            step = hist_bin/4
            if step == 0: step = hist_bin
        
        x_hist.append(np.arange(tws[i_trial][0], tws[i_trial][1]-hist_bin+1, step))
        n_event = []
        n_event += [np.sum(np.logical_and(trial_i>=bin_i, trial_i<bin_i+hist_bin)) for bin_i in x_hist[-1]]
        n_event = np.array(n_event, dtype=int)
        y_bar.append(n_event*1000/float(hist_bin))
    
    return x_hist, y_bar


def roc_index_filtering(X, label, class_label, tws=[0,1000], sliding_window=500, step=None, n_bin=None, n_perm=100, plotting=False):
    """input: X: recorded trials of one neuron
              label: trial label
              class_label: [label class 0, label class 1]
              sliding_window: time window length in which roc curve is computed
              hist_bin: bin size to compute the roc curve given a sliding window
              step: in moving the sliding window
       output: auc: [len(x_hist)] only significant auc (pvalue .5), 0 otherwise 
    """
    try:
        if len(tws[0])>=2 and len(tws)==len(X):
            print "Size tws coherent, single tws"
        else:
            print "Error size tws"
    except TypeError:
        print "Global tws"
        tws_new = []
        tws_new += [tws for _ in X]
        tws = tws_new
    shuffle = np.concatenate(([False], np.repeat(True, n_perm)), axis=0)
    #print 'Shuffle: ', shuffle
    auc = []
    auc += [[] for _ in shuffle]
    nTrial = len(label)
    
    for i_perm, perm_i in enumerate(shuffle):
        index = range(nTrial)
        if perm_i:
            np.random.shuffle(index)
        label_toUse = []
        label_toUse += [label[i] for i in index]
        #print "index", index
        #print "label", label
        #print "labeltoUse", label_toUse
        #class separation
        X0 = []
        X1 = []
        tws0 = []
        tws1 = []
        for i_label, label_i in enumerate(label_toUse):
            if label_i in class_label[0]:
                X0 += [X[i_label]]
                tws0  += [tws[i_label]]
            elif label_i in class_label[1]:
                X1 += [X[i_label]]
                tws1 += [tws[i_label]]
            else:
                print "Unknown label"
        
        #pdb.set_trace()
        #if not(perm_i):
            #from load_data_connectivity import scatter_plot
            #scatter_plot(X0, tws=tws, hist_bin=hist_bin, step=step, options=['null', 'null', 0])
            #scatter_plot(X1, tws=tws, hist_bin=hist_bin, step=step, options=['null', 'null', 1])
        #sliding_window, step
        
        x_time0, y_spk0 = compute_spike_density_across_trial(X0, tws=tws0, hist_bin=sliding_window, step=step, n_bin=n_bin)
        x_time1, y_spk1 = compute_spike_density_across_trial(X1, tws=tws1, hist_bin=sliding_window, step=step, n_bin=n_bin)
        
        max_bin0, max_xtime0 = longhest_xtime(x_time0)
        max_bin1, max_xtime1 = longhest_xtime(x_time1)
                
        if max_bin0 > max_bin1:
            max_bin = max_bin1
        else:
            max_bin = max_bin0
        
        for i_bin in range(max_bin):
            y_spk_bin = []
            y_spk_bin += [y_spk0_i[i_bin] for y_spk0_i in y_spk0 if i_bin < len(y_spk0_i)] 
            y_spk_bin = np.array(y_spk_bin, int)
            x_hist0, y_bar0 = compute_prob_density(y_spk_bin, bin_size=1, plotting=plotting)
            y_spk_bin = []
            y_spk_bin += [y_spk1_i[i_bin] for y_spk1_i in y_spk1 if i_bin < len(y_spk1_i)]             
            y_spk_bin = np.array(y_spk_bin, int)
            x_hist1, y_bar1 = compute_prob_density(y_spk_bin, bin_size=1, plotting=plotting)
            
            fpr, tpr = compute_roc_curve(x_hist0, y_bar0, x_hist1, y_bar1, bin_size=1)
            auc_tmp = metrics.auc(fpr, tpr, reorder=True)
            if auc_tmp < .5:
                auc_tmp = 1 - auc_tmp
            auc[i_perm].append(auc_tmp)
            if plotting:
                plt.figure()
                plt.plot(fpr, tpr, '.r')
                plt.plot([0,1], [0,1], '-k')
                plt.plot([0,1], [1,1], '-k')
                plt.title('auc:%.4f' % auc[i_perm, i_bin])
        
    #at least one significant auc across windows = good neuron
    #pdb.set_trace()
    nbin = len(auc[0]) #n bin of permt=False
    pvalue = np.zeros(nbin)
    #auc = np.array(auc, dtype=float)
    auc_fixBin = []
    auc_fixBin += [auc_i[:nbin] for auc_i in auc if len(auc_i)>=nbin]
    auc_fixBin = np.array(auc_fixBin, dtype=float)
    n_perm = auc_fixBin.shape[0] - 1
    for i_bin, auc_i in enumerate(auc_fixBin.T):
        pvalue[i_bin] = (np.sum(auc_i[1:] >= auc_i[0])+1) / (float(n_perm)+1)
    
    return auc[0], pvalue
    
def longhest_xtime(x_time):
    
    max_bin = 0
    for x_time_i in x_time:
        if len(x_time_i) > max_bin:
            max_bin = len(x_time_i)
            max_xtime = x_time_i
    
    return max_bin, max_xtime


def feature_scaling(A):
    """Feature scaling according to wikipedia x-x_min / x_max-x_min 
    """  
    A = (A - A.min()) / (A.max() - A.min())
    return A

def grand_normalise(A):
    """Normalise (z-scoring) array A.
    """
    A = A - A.mean()
    A = np.nan_to_num(A / A.std())
    return A

def row_normalise(A):
    """Normalize along row array A
    """    
    A = column_normalise(A.T) 
    return A.T

def column_normalise(A):
    """NOrmalise along column array A
    """
    A = A - A.mean(0)
    A = np.nan_to_num(A / A.std(0))    
    return A


if __name__=='__main__':
    
    np.random.seed(np.random.randint(100))
    #pwd_save = ''
    #filename_dataset = '' % (pwd_save)
    
    #roc curve from the probability density
    mu0 = 0.5
    std0 = np.arange(.01,1,.05)
    n0 = 300
    mu1 = 1
    std1 = .2#np.arange(.01,.4,.01)
    n1 = 300
    
    shift = range(0,1)#np.arange(0,1,.01)
    plot_flag = False
    
    auc = np.zeros(len(std0))
    auc_sk = np.zeros(len(std0))
    
    for i_sh, sh_i in enumerate(std0):
        
        print "Data generation."    
        #x0 = np.random.beta(2,5,n0)
        x0 = np.random.normal(mu0, sh_i, n0)
        #x0 = np.concatenate([np.random.normal(mu0, std0, n0/2), np.random.normal(mu0+1, std0, n0/2)])
        #x0 = np.random.uniform(0,2,n0)  
        x1 = np.random.normal(mu1, std1, n1)
        #x1 = np.random.beta(2,2,n1) + sh_i
        
        print "Frequency distribution."
        bin_size = .01
        x0_axis, x0_freq = compute_prob_density(x0, bin_size)
        x1_axis, x1_freq = compute_prob_density(x1, bin_size)
        
        if plot_flag:
            plt.figure()
            plt.plot(x0_axis, x0_freq, '.r')
            plt.plot(x1_axis, x1_freq, '*b')
            plt.show()
            
        print "Compute ROC curve."
        fpr, tpr = compute_roc_curve(x0_axis, x0_freq, x1_axis, x1_freq, bin_size)
        auc[i_sh] = metrics.auc(fpr, tpr, reorder=True)
        
        if plot_flag:
            plt.figure()
            plt.plot(fpr, tpr, '.-k')
        
        #roc curve from binary classification algorith
        print "Classification problem."    
        X = np.concatenate([x0, x1])
        X = X.reshape(-1,1)
        
        feature_eng_label = False
        if feature_eng_label:
            print "Feature Engineering."
            X_feat_eng = []
            X_feat_eng += [feature_engineering(X[i_trial], block_normalisation=False) for i_trial in range(n0+n1)]
            X = np.array(X_feat_eng, dtype=float)
        
        X = metrics.pairwise.rbf_kernel(X)
        y_true = np.concatenate([np.repeat(0,n0), np.repeat(1,n1)])
        
        skf = StratifiedKFold(y_true, n_folds=5)
        clf = LogisticRegression(C=1.0, penalty='l2', random_state=0)
        #clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        #clf = SparseNonlinearClassifier(gamma=0.1, alpha=1./0.05)
        
        y_pred = np.zeros(n0+n1)
        pred_prob = np.zeros([n0+n1,2])
        for train_index, test_index in skf:
            clf.fit(X[train_index], y_true[train_index])
            y_pred[test_index] = clf.predict(X[test_index])
            pred_prob[test_index] = clf.predict_proba(X[test_index])
        
        print "Compute curve and auc."    
        auc_sk[i_sh] = metrics.roc_auc_score(y_true, pred_prob[:,1])
        fpr_sk, tpr_sk, thresholds = metrics.roc_curve(y_true, pred_prob[:,1])#, pos_label=1)
    
        if plot_flag:
            plt.plot(fpr_sk, tpr_sk, '.-g')
            plt.text(0.1,1,'auc=%s' % auc[i_sh])
            plt.text(0.1,0.9,'auc_sk=%s' % auc_sk[i_sh])
            plt.xlabel('fpr')
            plt.ylabel('tpr')
            plt.show()
        
    plt.figure()
    plt.plot(std0,auc)
    plt.plot(std0,auc_sk)
    plt.legend(['std', 'clf'])
    plt.show()
    