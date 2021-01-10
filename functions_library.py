# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 14:50:06 2018
Container of elementary functions
@author: danilo
"""

import numpy as np
import pdb

def trial_random_permutation(X):
    """X:[n_cell][n_trial] for each cell random permute the trial order
    """
    X_perm = []
    for X_i in X:
        X_perm_i = []
        X_perm_i += [X_i[trial_i] for trial_i in np.random.permutation(len(X_i))]
        X_perm.append(X_perm_i)
    
    return X_perm

def crazyshuffle(arr):
    """Random permutation for each row
    """
    x, y = arr.shape
    rows = np.indices((x,y))[0]
    cols = [np.random.permutation(y) for _ in range(x)]
    return arr[rows, cols]

def adjust_multinomial_dist(X):
    """Modify X (1dim array) in order to contain all integer from its min (zero) and max
       [0,1,2,3,2,3,1] is correct, [1,2,5,3,5] is not correct
    """
    unique_sample = np.unique(X)
    unique_sample_corr = np.arange(len(unique_sample))
    wrong_sample = unique_sample - unique_sample_corr
    i = 0
    while i < len(unique_sample):
        if wrong_sample[i] > 0:
            X[X==unique_sample[i]] = unique_sample_corr[i]
        i += 1
    return X, unique_sample

def set_range(X, up_b):
    """
    """
    X = adjust_multinomial_dist(X)
    X = np.rint(X*up_b/float(X.max()))
    return np.array(X, dtype=int)

def compute_stable_state(state_prob, Z=None, p_min=.8, length_min=50, bin_size=1):
    """Find states with prob>=p_min in at least length_min consecutive ms windows
    """
    n_obs, n_components = state_prob.shape
    length_min_bin = length_min/bin_size #bins in length_min
    ############################# old version
    if Z is None:
        state_stable = state_prob >= p_min #bool variable telling where condition on the p_min is satisfied
    else:
    ############################# new version
        state_stable = np.zeros([n_obs, n_components], dtype=np.bool)
        for i_comp in range(n_components):
            state_stable[Z==i_comp,i_comp] = state_prob[Z==i_comp,i_comp] >= p_min #true where the condition is satisfied but based on the predicted state path
    #############################
    state_stable = np.concatenate((np.zeros([1, n_components], dtype=bool), state_stable))
    state_stable = np.concatenate((state_stable, np.zeros([1, n_components], dtype=bool)))
    
    for i in range(n_components):
        state_diff = np.diff(state_stable[:,i])
        j = 0
        while j<n_obs:
            while j<n_obs and state_diff[j]!=1: 
                j += 1
            if j<n_obs and state_diff[j]==1:
                n_true = 1
                j_true = j+1
                while j_true<n_obs and state_diff[j_true]==0:
                    n_true += 1
                    j_true += 1
                if n_true < length_min_bin:
                    state_stable[j+1:j_true+1,i] = False
                j = j_true + 1
    state_stable = state_stable[1:-1,:]
    return state_stable

def summingto1(l):
    """Normalize the list to summing 1
    """
    return l/l.sum()

def firing_rate(X, tws, bin_size, normalize=True, sum_over_trial=True):
    """Compute the firing rate of a given set of trial, knowing the time window and the bin size
       y_fir_rate [nbin, ntrial]
    """
    x_axis = list(np.arange(tws[0],tws[1],bin_size))
    n_trial = len(X)
    #print "n_trial:", n_trial 
    y_fir_rate = []
    for i_bin, bin_i in enumerate(x_axis):
        n_event = []
        n_event += [np.sum(np.logical_and(X[i_trial]>=bin_i, X[i_trial]<bin_i+bin_size)) for i_trial in range(n_trial)]
        if normalize:
            y_fir_rate.append( (np.sum(n_event)/float(n_trial)) * (1000/float(bin_size)) )
        else:
            if sum_over_trial:
                y_fir_rate.append(np.sum(n_event))
            else:
                y_fir_rate.append(n_event)
    #print "len(y_fir_rate):", len(y_fir_rate) 
    return y_fir_rate, x_axis
               
def firing_rate_tws(X, tws, bin_size, normalize=True, sum_over_trial=True):
    """Compute the firing rate of a given set of trial of a specific cell, knowing the time window and the bin size
       time window is trial specific
       y_fir_rate [ntrial, nbin]
       NB there is no correspondance of time between trials so summing time bins over trial does not make sense
    """
    #print "n_trial:", n_trial 
    y_fir_rate = []
    x_axis = []
    for i_trial, X_i in enumerate(X): #for each trial
        x_axis_trial = list(np.arange(tws[i_trial][0], tws[i_trial][1], bin_size))
        n_event = []
        n_event += [np.sum(np.logical_and(X_i>=bin_i, X_i<bin_i+bin_size)) for bin_i in x_axis_trial]
        y_fir_rate.append(n_event)
        x_axis.append(x_axis_trial)
    #print "len(y_fir_rate):", len(y_fir_rate) 
    return y_fir_rate, x_axis

def make_observation_vector(X, sum_over_cell=False):
    """Build the observation vector fo rhmm analysis
    """
    n_cell = len(X)
    n_bin = len(X[0])
    n_trial = len(X[0][0])
    if sum_over_cell: #make sense only if nbins is the same for each trial
        X_fit = np.sum(np.array(X, dtype=int) > 0, axis=0) #sum over cells
        X_fit = np.ravel(X_fit, order='F') #vertical stacking of each trial      
        length = np.repeat(n_bin, n_trial)
    else:
        X_array = np.array(X, dtype=int)
        X_array = np.array(X_array > 0, dtype=int)#if more than 1 spike in the selected bin, count only one
        X_fit = np.zeros([n_bin, n_trial], dtype=int) - 1
        for i_trial in range(n_trial):
            for i_bin in range(n_bin):
                if np.sum(X_array[:,i_bin,i_trial]) == 1:#only one spike, position of that cell
                    X_fit[i_bin,i_trial] = np.where(X_array[:,i_bin,i_trial])[0]
                if np.sum(X_array[:,i_bin,i_trial]) >= 1:#more than one spike, random selection of one cell
                    tmp = np.where(X_array[:,i_bin,i_trial])[0]
                    X_fit[i_bin,i_trial] = tmp[np.random.randint(len(tmp))]
        X_fit = np.ravel(X_fit, order='F')
        X_fit += 1
        print "Spoken cells: ", np.unique(X_fit)
        length = np.repeat(n_bin, n_trial)
        
    X_fit, cell_label = adjust_multinomial_dist(X_fit)#cell_label contains labels of spoken cells (if not sum_over_cell)
    X_fit = X_fit.reshape(-1,1)
    
    return X_fit, length

def make_observation_vector_tws(X, sum_over_cell=False):
    """Build the observation vector fo rhmm analysis, to use if number of observations is different over trials
    """
    n_cell = len(X)
    n_trial = len(X[0])
    if sum_over_cell: #make sense only if nbins is the same for each trial
        n_bin = len(X[0][0])
        X_fit = np.sum(np.array(X, dtype=int) > 0, axis=0) #sum over cells
        X_fit = np.ravel(X_fit, order='F') #vertical stacking of each trial      
        length = np.repeat(n_bin, n_trial)
    else:
        X_fit = []
        length = []
        for i_trial in range(n_trial):
            X_array = []
            X_array += [X[i_cell][i_trial] for i_cell in range(n_cell)]
            n_bin = len(X_array[0])
            X_array = np.array(X_array, dtype=int)
            #print 'size X_array: ', np.shape(X_array)
            X_array = np.array(X_array > 0, dtype=int)#if more than 1 spike in the selected bin, count only one
            X_fit_trial = np.zeros(n_bin, dtype=int) - 1
            for i_bin in range(n_bin):
                if np.sum(X_array[:,i_bin]) == 1:#only one spike, position of that cell
                    X_fit_trial[i_bin] = np.where(X_array[:,i_bin])[0]
                if np.sum(X_array[:,i_bin]) > 1:#more than one spike, random selection of one cell
                    tmp = np.where(X_array[:,i_bin])[0]
                    #print '#spk %d in bin %d' % (len(tmp), i_bin)
                    X_fit_trial[i_bin] = tmp[np.random.randint(len(tmp))]
            X_fit += [X_fit_trial+1]
            length += [n_bin]
        X_fit = np.hstack(X_fit) #1D array
        print "Spoken cells: ", np.unique(X_fit)
        length = np.hstack(length)
        
    X_fit, cell_label = adjust_multinomial_dist(X_fit)#cell_label contains labels of spoken cells (if not sum_over_cell)
    X_fit = X_fit.reshape(-1,1)
    
    return X_fit, length
    
def trial_selection(data, tws, cls, selected_trial, trial_start, trial_end, task_response, second_filter, selected_task_resp, selected_task_resp_optional, updown_s2, updown_s2_event):
    """
    """
    print "selected task resp: ", selected_task_resp
    #Selecting trials
    start_i = []
    start_i += [trial_start[i] - 1 for i in cls] #starting from 0
    start_ = np.max(start_i)
    end_i = []
    end_i += [trial_end[i] for i in cls]
    end_ = np.min(end_i)
        
    X = []
    tws_ = []
    index_trial = []
    label = []
    n_trial = np.sum(selected_trial[start_:end_])
    print "n_trial:", n_trial
    for i_cls, cls_i in enumerate(cls):
        n_trial_before = np.sum(selected_trial[start_i[i_cls]:start_]) #start_i derives from cls -> i_cls as index
        #print "n_trial_before:", n_trial_before
        index_trial_ = []
        index_trial_ += [i_trial for i_trial in np.arange(n_trial_before, n_trial_before+n_trial) if (task_response[cls_i][i_trial] in selected_task_resp and second_filter[cls_i][i_trial] in selected_task_resp_optional and updown_s2[cls_i][i_trial] in updown_s2_event)]
        index_trial.append(index_trial_)        
        X_i = []
        X_i += [data[cls_i][i_trial] for i_trial in index_trial_] #cls_i is used to access to data since all units are there (we only need units in the cls)
        X.append(X_i)
        print "len(X_i):", len(X_i)
        #print "len(X):", len(X)
        del X_i
    #pdb.set_trace()
    for i_trial in range(len(index_trial_)):
        label_vector = []
        label_vector += [task_response[cls_i][index_trial[i_cls][i_trial]] for i_cls, cls_i in enumerate(cls)]
        #assert(len(np.unique(label_vector)) == 1)
        assert(label_vector[i] == label_vector[i-1] for i in range(1,len(cls)))
        
    label += [task_response[cls[0]][i_trial] for i_trial in index_trial[0]] #cls_i is used to access to data since all units are there (we only need units in the cls)
    #label = np.array(label, dtype=int)
    print "len(label):", len(label)
    tws_ += [tws[cls[0]][i_trial] for i_trial in index_trial[0]]  #same tws across neurons in the same session
    return X, tws_, label, index_trial