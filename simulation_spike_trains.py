# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:25:52 2018

@author: danilo
"""

import numpy as np
import os
from matplotlib import cm, pyplot as plt
from scipy.io import loadmat, savemat
from hmmlearn import hmm
from sklearn.cross_validation import KFold
from sklearn.metrics import r2_score

from hmm_neurofis import summingto1, hmmtrain_matlab, hmmdecode_viterbi_matlab

def squeeze_stable_state(m):
    """for each fold (1st dimension of m) keep only the max in each state, compute the mean of stable time points per time state window
    """
    squeeze_m = np.zeros([m.shape[0], m.shape[2]])
    for i in range(m.shape[0]):
        #print "-----------------"
        #print "m:", m[i]
        pos_max = np.argmax(m[i],1)
        #print "pos max: ", pos_max
        tmp = np.zeros(m.shape[2])
        for j, j_max in enumerate(pos_max):
            #print "_____"
            #print "m[...]", m[i,j,j_max]
            tmp[j_max] += m[i,j,j_max]
            #print "tmp[j_max]: ", tmp[j_max]
        squeeze_m[i] = tmp
    #print squeeze_m
    return squeeze_m.mean(0)

def stability_matrix(state_prob, Z, n_bin, n_trial, bin_size=1, time_state=np.array([1,1,1])):
    """
    """
    from functions_library import compute_stable_state 
    n_components = state_prob.shape[1]
    p_min = 0.8 #min probabilty of being a state
    length_min = 50 #minimum length in ms to be a state a stable state
    state_stable = compute_stable_state(state_prob, Z, p_min, length_min, bin_size=bin_size*1000) #bin_size in ms
    sum_state_stable = np.zeros([len(time_state), n_components])
    time_end = np.array((np.cumsum(time_state/bin_size)), dtype=int)
    n_bin_per_state = time_state/bin_size
    #print n_bin_per_state
    for j, j_end in enumerate(time_end):
        tmp = 0
        #print "j_end: ", j_end
        for i in range(n_trial):        
            tmp += np.sum(state_stable[i*n_bin + (j_end-time_end[0]) : i*n_bin + j_end , :],0) 
            #print tmp
        tmp = tmp / (n_trial*n_bin_per_state[j])
        sum_state_stable[j,:] = tmp
        #print "time window %s: %s" % (j, tmp)
    
    return sum_state_stable.T
    

def score_firing_rate(fir_rates, fir_rates_est):
    """score between true and estimated firing rates
    """
    index_true = np.argsort(np.mean(fir_rates,1))
    index_est = np.argsort(np.mean(fir_rates_est,1))
    y_true = []
    y_pred = []
    for i, i_true in enumerate(index_true):
        y_true.append(fir_rates[i_true])
        y_pred.append(fir_rates_est[index_est[i]])
    y_true = np.hstack(y_true)
    y_pred = np.hstack(y_pred)
    return r2_score(y_true, y_pred)#np.mean((y_true-y_pred)**2)

def repeat_hmm_cv_simulation(X_fit, par_rep, n_cell, n_bin, n_trial, n_components=4, n_folds=5, n_iter=500, tol=1e-05, matlab_=False, eng=None, bin_size=1, time_state=np.array([1,1,1]), options=None, path_matlab=None):
    """
    """
    n_rep = len(par_rep) 
    n_symbols = len(np.unique(X_fit))
    symbols = np.unique(X_fit)
    score_rep = np.zeros(n_rep) #BIC or loglikelihood
    last_score = -np.inf
    
    for i_rep in range(n_rep):
        random_state = par_rep[i_rep]
        init_params = 'e'
        startprob_prior = np.concatenate([[1], np.zeros(n_components-1)])
        emissionprob_prior = np.concatenate([np.ones([n_components,1]), np.zeros([n_components, n_symbols-1])], axis=1) + np.abs(np.random.randn(n_components, n_symbols)*.2)
        transmat_prior = np.identity(n_components) + np.abs(np.random.randn(n_components, n_components)*.2)
        for i in range(n_components):
            emissionprob_prior[i] = summingto1(emissionprob_prior[i])
            transmat_prior[i] = summingto1(transmat_prior[i])
        
        if n_folds == 1:
            kf = [(range(n_trial), range(n_trial))]
        else:
            kf = KFold(n_trial, n_folds=n_folds, shuffle=True, random_state=random_state)

        if not matlab_: 
            model_ = hmm.MultinomialHMM(n_components=n_components, n_iter=n_iter, random_state=random_state, startprob_prior=startprob_prior, transmat_prior=transmat_prior, tol=tol, init_params=init_params, algorithm='viterbi')        
        else:
            model_ = None
        
        logprob_pertrial = []
        logprob_pertrial += [[] for _ in range(n_folds)]
        
        logprob_ = np.zeros(n_folds)
        bic_ = np.zeros(n_folds)
        score_fr_ = np.zeros(n_folds)
        sum_stable_state_ = np.zeros([n_folds, n_components, time_state.shape[0]])
        Z_ = np.zeros(n_trial*n_bin) - 1
        state_prob_ = np.zeros([n_trial*n_bin, n_components])
        for i_fold, (train_index, test_index) in enumerate(kf):
            ######## fit
            trial_train_index = []
            trial_train_index += [range(i_trial*n_bin,(i_trial+1)*n_bin) for i_trial in train_index]
            if not matlab_:
                #------------- hmmlearn
                trial_train_index = np.hstack(trial_train_index)
                length = np.repeat(n_bin, len(train_index))
                model_.fit(X_fit[trial_train_index], length)
                pred_fr = model_.emissionprob_[:,1:] / bin_size
            else:
                #----------- hmmtrain
                X_fit_matrix = []
                X_fit_matrix += [X_fit[i] for i in trial_train_index]
                X_fit_matrix = np.array(X_fit_matrix)
                print "Spoken cell i_fold: ", np.unique(X_fit_matrix)
                transmat, emissionprob = hmmtrain_matlab(X_fit_matrix, transmat_prior, emissionprob_prior, symbols, tol, n_iter, eng, path_matlab)
                pred_fr = emissionprob[:,1:] / bin_size
                trial_train_index = np.hstack(trial_train_index)
            ######## predict
            trial_test_index = []
            trial_test_index += [range(i_trial*n_bin,(i_trial+1)*n_bin) for i_trial in test_index]
            print "rep:%d --- fold:%d" % (i_rep, i_fold)
            #print "Train trial: ", train_index
            #print "Test trial: ", test_index
            if not matlab_:
                #------------hmmlearn
                #trial_test_index = np.hstack(trial_test_index)
                #length_test = np.repeat(n_bin, len(test_index))
                #X_predict = X_fit[trial_test_index]
                #Z_[trial_test_index] = model_.predict(X_predict, length_test)
                #logprob_[i_fold], state_prob_[trial_test_index,:] = model_.score_samples(X_predict, length_test)
                ####### One trial per time
                for i, i_trial in enumerate(test_index):
                    Z_[trial_test_index[i]] = model_.predict(X_fit[trial_test_index[i]])
                    tmp, state_prob_[trial_test_index[i],:] = model_.score_samples(X_fit[trial_test_index[i]])
                    logprob_pertrial[i_fold] += [tmp]
                logprob_[i_fold] = np.mean(logprob_pertrial[i_fold])
                trial_test_index = np.hstack(trial_test_index)
            else:
                #-----------hmmtrain
                for i, i_trial in enumerate(test_index):
                    state_prob_[trial_test_index[i],:], tmp, Z_[trial_test_index[i]] = hmmdecode_viterbi_matlab(X_fit[trial_test_index[i]], transmat, emissionprob, symbols, eng, path_matlab)
                    logprob_pertrial[i_fold] += [tmp]
                    #Z_[trial_test_index[i]] = hmmviterbi_matlab(X_fit[trial_test_index[i]], transmat, emissionprob, symbols, eng)
                logprob_[i_fold] = np.mean(logprob_pertrial[i_fold])       
                trial_test_index = np.hstack(trial_test_index)
            print "Still to test: ", np.sum(Z_ == -1)
            print "check train/test: ", np.unique(np.diff(np.sort(np.concatenate([trial_test_index,trial_train_index]))))            
            #score_fr_[i_fold] = score_firing_rate(options, pred_fr)
            bic_[i_fold] = logprob_[i_fold] - ((n_components**2 + n_components*(n_symbols-2)) / 2 * np.log(n_bin*len(test_index)))
            sum_stable_state_[i_fold,:,:] = stability_matrix(state_prob_[trial_test_index,:], Z_[trial_test_index], n_bin, len(test_index), bin_size, time_state)
                
        score_rep[i_rep] = np.mean(bic_)
        print "Score BIC: ", score_rep[i_rep]
        print "Score firing rate: ", score_fr_.mean()
        print "Stable state: ", squeeze_stable_state(sum_stable_state_)*100
        if score_rep[i_rep] > last_score: #change variable to mean according to the score measure
            if matlab_:
                Z_ = Z_ - 1#matlab start from 1
            Z = Z_
            logprob = logprob_
            state_prob = state_prob_
            bic = bic_
            model = model_
            sum_stable_state = sum_stable_state_
            score_fr = score_fr_
            last_score = score_rep[i_rep]
    
    print "-----------------------------------"
    print "Mean BIC: ", last_score
    print "Mean score firing rate: ", np.mean(score_fr)
    print "std score firing rate: ", np.std(score_fr)
    print "logLik: ", logprob
    print "BIC: ", bic
    #print "stable state sum: ", sum_stable_state
    
    return model, Z, state_prob, sum_stable_state, logprob, bic, score_rep, last_score

if __name__=='__main__':
    
    pwd = '%s/' % os.getcwd()
    N = 5 #number of neurans = spike trains
    n_trial = 80
    generate_data = True
    
    if generate_data:
        plot_raster = False
        step = 3
        T = [.15, .1, .15, .35, .2, .15, .25, .1, .2, .35] #total time duration (s) per segment
        T_l = np.cumsum(T) - T
        f_tot = range(step, step*(len(T)+1), step) #upper bound uniform distr
        f_l = np.concatenate(([f_tot[0]],np.diff(f_tot))) #size uniform distribution
        S_tot = []
        S_tot += [([]) for _ in range(N)]
        for i, s_i in enumerate(S_tot):
            s_i += [([]) for _ in range(n_trial)]
            S_tot[i] = s_i
        
        #generate firing rates for eachs state, this will be the same in each trial
        fir_rates = []
        for i, f_i in enumerate(f_tot):
            fir_rates += [f_i * np.random.rand(N)]#[f_l[i] * np.random.rand(N) + f_i - f_l[i]]
        print "Firing rates: ", fir_rates
        
        print "Staring simulations"
        for i_trial in range(n_trial):
            print "trial ", i_trial
            fir_rates_i = []
            for i, f_i in enumerate(f_tot):
                print "spk/s ", f_i
                #f = 0.1 * np.random.randn(N) + f_i #normaly distributed around f_i 
                f = fir_rates[i] #f_i * np.random.rand(N) #f_l[i] * np.random.rand(N) + f_i - f_l[i] #firing rates -- uniformly distributed between f_i-f_l[i] and f_i spk/s            
                #fir_rates_i += [f]            
                Nspk = int(np.ceil(f.max() * T[i] * 2))
                ISIs = -np.log(np.random.rand(Nspk, N)) / f
                S = np.cumsum(ISIs,0)
                for i_n in range(N):
                    S_n = []
                    for i_time in range(Nspk):
                        if S[i_time, i_n] < T[i]:
                            S_n.append(S[i_time, i_n]+T_l[i])
                    S_tot[i_n][i_trial] += S_n
                
                if plot_raster:                
                    S[S>1] = np.NaN
                    plt.figure(i)
                    Neu = np.repeat(np.arange(N).reshape([1,N]), Nspk, axis=0) + 1
                    plt.plot(S, Neu, '.k')
                    plt.ylim([0,N+1])
                    plt.xlabel('time [s]')
                    plt.ylabel('#neurons')
                    plt.xlim([0,T[i]])
                    plt.show()
            
            #fir_rates += [fir_rates_i] #in case each trial has its own firing rate
        
        for i, s_i in enumerate(S_tot):
            for j, s_ij in enumerate(s_i):
                S_tot[i][j] = np.array(s_ij, dtype=float)
        
        print "Saving simulated file"
        filename_save = '%ssimulated_data_%dneurons_%dtrials.mat' % (pwd, N, n_trial)
        print "Saving %s" % filename_save
        savemat(filename_save, {'X': S_tot,
                                'spkf': f_tot,
                                'firing_rates': fir_rates,
                                'time_state': T,
                                'n_trial': n_trial,
                                'n_cell': N
        })
    
    ###############################3
    print "Opening file and run hhm analysis"
    #N = 10
    #n_trial = 100
    filename_open = '%ssimulated_data_%dneurons_%dtrials.mat' % (pwd, N, n_trial)
    data = loadmat(filename_open, squeeze_me=True)
    X = data['X']
    spkf = data['spkf']
    fir_rates = data['firing_rates']
    time_state = data['time_state']
    t_start = np.cumsum(time_state) - time_state
    t_end = np.cumsum(time_state)
    tws = [t_start[0], t_end[-1]]
    bin_size = 0.005 #s
    
    print "Computing firing rate for each unit"
    from functions_library import firing_rate
    sum_over_trial = False
    X_fir_rate = []
    for X_i in X:
        #print "len(X_i):", len(X_i)
        X_tmp, x_axis = firing_rate(X_i, tws, bin_size, normalize=False, sum_over_trial=sum_over_trial)
        X_fir_rate.append(X_tmp)
        #print "len(X_tmp):", len(X_tmp)
        #print "len(X_fir_rate):", len(X_fir_rate)
        del X_tmp
        
    print "HMM training"
    sum_over_cell = False
    n_cell = len(X_fir_rate)
    n_bin = len(X_fir_rate[0])
    n_trial = len(X_fir_rate[0][0])
    ######### initialization
    
    ######### make sample from a multinomial distribution
    from functions_library import make_observation_vector
    X_fit_array, lenght = make_observation_vector(X_fir_rate, sum_over_cell=sum_over_cell)
    
    ######### fit and prediction
    n_folds = 1
    n_rep = 5
    n_components = 3
    par_rep = np.random.randint(100, size=n_rep) #parameter to repeat
    matlab_ = True
    eng = None
    if matlab_:
        import matlab.engine
        eng = matlab.engine.start_matlab()
        eng.edit('hmm_matlab')
    
    model, Z, state_prob, sum_stable_state, logprob, bic, score_rep, last_score = repeat_hmm_cv_simulation(X_fit_array, par_rep, n_cell, n_bin, n_trial, n_components=n_components, n_folds=n_folds, matlab_=matlab_, eng=eng, bin_size=bin_size, time_state=time_state, options=fir_rates, path_matlab=pwd)          

    if matlab_:
        eng.quit()
    
    sum_stable_state_squeeze = squeeze_stable_state(sum_stable_state)    
    print "-----------------------------"    
    print "Max score across repetitions: ", last_score 
    print "Stable state: ", sum_stable_state_squeeze*100
    
    ######## about state stability
    from functions_library import compute_stable_state 
    from plotting_functions import plot_data_states
    p_min = 0.8 #min probabilty of being a state
    length_min = 50 #minimum length in ms to be a state a stable state
    state_stable = compute_stable_state(state_prob, Z, p_min, length_min, bin_size=bin_size*1000) #bin_size in ms
    
    to_plot = True
    if not sum_over_trial and to_plot:
        for i_trial in range(10):
            X_toplot = []
            X_toplot += [xn[i_trial] for xn in X]
            selected_trial = range(i_trial*n_bin, (i_trial+1)*n_bin)
            plot_data_states(np.array(x_axis, dtype=float), X_toplot, Z[selected_trial], state_prob[selected_trial], state_stable[selected_trial], logprob, tws, separation=np.cumsum(T), text=np.array(np.mean(fir_rates,1), dtype=int))#[list(np.sum(X_fir_rate, axis=0))]
        
    print "Saving for matlab"
    filename_save = '%ssimulated_data_%dneurons_%dtrials_hmmlearn_newBIC_output.mat' % (pwd, N, n_trial)
    print "Saving %s" % filename_save
    savemat(filename_save, {'seq': X_fit_array,
                            'X': X,
                            'Z': Z,
                            'state_prob': state_prob,
                            'state_stable': state_stable,
                            'logprob': logprob,
                            'n_trial': n_trial,
                            'n_bin': n_bin,
                            'n_cell': n_cell,
                            'tws': tws,
                            'bin_size': bin_size,
                            'fir_rates': fir_rates
    })
    
    ##############