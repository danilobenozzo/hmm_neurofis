# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 15:19:13 2018
Plotting function container

@author: danilo
"""
import numpy as np
from matplotlib import cm, pyplot as plt
#import elephant as ele
#import neo as n
#import quantities as pq
import scipy.signal as signal
#import pdb

def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)
    return ax

def figure_layout(x_in, y_in):
    
    fontsize_main = 15./4.*y_in
    fontsize = 12./4.*y_in
    linewidth_main = 1.5/4.*y_in
    linewidth = .8/6.*y_in
    markersize_main = 10./4.*y_in
    markersize = 8./4.*y_in
    return fontsize_main, fontsize, linewidth_main, linewidth, markersize_main, markersize

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def plot_state_firing(emission_firing, x_in=4, y_in=5):
    """
    """
    N, M = emission_firing.shape # number of states, ncells
    
    fontsize_main, fontsize, linewidth_main, linewidth, markersize_main, markersize = figure_layout(x_in, 2.5)
    fig, ax = plt.subplots(1, N, figsize=(x_in, y_in))
    colours = cm.Set2(np.linspace(0, 1, 8))[1:1+N] #cm.Set2(np.linspace(0, 1, N))
    y_pos = np.arange(M)[1:]
    max_spks = np.round(np.max(emission_firing[:, 1:]), -1)
    
    for i_state, colour in enumerate(colours):
        #plt.subplot(1, N, i_state+1)
        ax[i_state].barh(y_pos, emission_firing[i_state, 1:], color=colour)
        if i_state == 0:
            ax[i_state].set_yticks(y_pos)
        else:
            ax[i_state].set_yticks([])
        ax[i_state].set_xlim([0,max_spks])
        ax[i_state].tick_params(labelsize=fontsize_main)

    return fig, ax
    
def plot_data_states(x, X, Z, state_prob, state_stable, logprob=-np.inf, tws=[-400, 400], x_in=4, y_in=5, separation=None, text=None):
    """
    """
    N = state_prob.shape[1]
    M = 1 #number of subplot
    fontsize_main, fontsize, linewidth_main, linewidth, markersize_main, markersize = figure_layout(x_in, 2.5)#y_in)
    #plt.figure(figsize=(20,8))
#    for i_x, x_i in enumerate(X):
#        plt.subplot(M+1,1,i_x+1)
#        plt.plot(x, x_i, 'k')
#        plt.ylabel('#spikes', fontsize=fontsize)
#        #plt.ylabel('spikes cell; %d' % i_x)
    
#    plt.subplot(M+1,1,1)
    fig, ax1 = plt.subplots(figsize=(x_in, y_in))#(10,7)

    ax2 = ax1.twinx()
    colours = cm.Set2(np.linspace(0, 1, 8))[1:1+N] #cm.Set2(np.linspace(0, 1, N))#rainbow(np.linspace(0, 1, N))
    #max_prob = np.max(state_prob, axis=1)
    max_prob = []
    max_prob += [state_prob[i_z, z_i] for i_z, z_i in enumerate(Z)]
    max_prob = np.array(max_prob, dtype=np.float)
    for i_state, colour in enumerate(colours):
        #pos_i = Z==i_state
        #ax2.plot(x[state_stable[:,i_state]], np.zeros(state_stable[:,i_state].sum()), '|', markersize=20, color=colour)#*1.1 #commented 05/06 
        #ax2.plot(x[state_stable[:,i_state]], max_prob[state_stable[:,i_state]], 'ko', markersize=7)
        #ax2.plot(x[pos_i], np.ones(pos_i.sum())*1.1, 's', markersize=7, color=colour)
        ax2.plot(x, state_prob[:,i_state], '-', linewidth=linewidth_main, color=colour, alpha=0.7)
        #mport matplotlib.transforms as mtransforms
        #trans = mtransforms.blended_transform_factory(ax2.transData, ax2.transAxes)
        ax2.fill_between(x, 0, state_prob[:,i_state], where = state_stable[:,i_state], facecolor=colour, alpha=0.5)
    
    plt.xlim(tws)
    ax2.yaxis.tick_right()
    ax2.set_yticks([])
    ax2.set_ylim([0,1.1])
    ax2.tick_params(labelsize=fontsize_main)
    
    for i_cell, cell_i in enumerate(X):
        cell_i = np.reshape(cell_i, (np.size(cell_i),1))
        cell_to_plot = cell_i[np.logical_and(cell_i>=tws[0], cell_i<tws[1])]
        digitize_event = np.digitize(cell_to_plot, x) -1 
        ax1.plot(x[digitize_event], np.repeat(i_cell+1, len(cell_to_plot)), '|k', markersize=markersize, alpha=1.0, zorder=10)#'dk', markersize=20) #'Xk', markersize=5
 
    ax1.set_ylim([0,len(X)+1])   
    ax1.set_yticks(range(1,len(X)+1))
    ax1.tick_params(labelsize=fontsize_main)
    
    #ax1.set_xlabel('Time [ms]', fontsize=fontsize_main)
    #ax1.set_ylabel('Neurons', fontsize=fontsize_main)
    #ax2.set_ylabel(r'$P(S_t|X_t)$', fontsize=fontsize_main)
    #ax2.ylim([0,1.1])
    #fig.tight_layout()
    
#    max_prob = np.max(state_prob, axis=1)    
#    #pos_max = np.argmax(state_prob, axis=1)
#    plt.subplot(M+1,1,M+1)
#    for i_state, state_i in enumerate(range(N)):
#        pos_i = Z==state_i
#        plt.plot(x[state_stable[:,i_state]], max_prob[state_stable[:,i_state]], 'ko', markersize=5)
#        plt.plot(x[pos_i], max_prob[pos_i], '.', markersize=3)
#    
#    plt.xlim([tws[0]-10, tws[1]+10])      
#    plt.xlabel('time[ms]', fontsize=fontsize)
#    plt.ylabel('prob states', fontsize=fontsize)
    #plt.plot(x, X, '-k')
    #plt.scatter(x, X, c=Z, s=50, cmap=discrete_cmap(N, 'brg'))
    #plt.colorbar(ticks=range(N))
    if not(separation is None):
        n_sep = len(separation)
        for i_sep, sep_i in enumerate(np.concatenate(([tws[0]],separation[:-1]))):
#            for i_fr, fr_i in enumerate(text[i_sep]):
#                plt.text(sep_i, 1+(i_fr*.02), np.around(fr_i, decimals=1), fontsize=8)
            ax1.text(sep_i, len(X)+1.1, text[i_sep], fontsize=fontsize)#1.02 in ax2
        separation = np.reshape(np.array(separation, dtype=float),[1,n_sep])
        x_sep = np.repeat(separation,2,0)
        y_sep = np.vstack([np.zeros([1,n_sep]), np.ones([1,n_sep])])
        ax2.plot(x_sep, y_sep, 'k--', linewidth=linewidth, alpha=0.7)
    #plt.show()
    return fig, ax1, ax2, fontsize
    
def scatter_plot(trials, tws=[0,1000], hist_bin=50, step=None, smooth=None, options=['null', 'null', 'null'], to_save=None):
    """Receive trials to plot with one trial per row
    """
    plt.figure()
    
    plt.subplot(211)    
    plt.title('file: %s, unit: %s, task:%s' % (options[0], options[1], options[2]), fontsize=10)
    
    for i_trial in range(len(trials)):
        trial_to_plot = trials[i_trial][np.logical_and(trials[i_trial]>=tws[0], trials[i_trial]<tws[1])]
        plt.plot(trial_to_plot, np.repeat(i_trial+1, len(trial_to_plot)), '.r', markersize=2)
        
    plt.xlim([tws[0]-10, tws[1]+10])    
    plt.ylim([0,len(trials)+1])
    plt.xlabel('time[ms]')
    plt.ylabel('#trial')
    
    plt.subplot(212)
    if step is None:
        step = hist_bin
    x_hist = np.arange(tws[0], tws[1]-hist_bin+1, step)
    y_bar = []
    for i_bin, bin_i in enumerate(x_hist):
        n_event = []
        n_event += [np.sum(np.logical_and(trials[i_trial]>=bin_i, trials[i_trial]<bin_i+hist_bin)) for i_trial in range(len(trials))]
        y_bar.append( (np.sum(n_event)/float(len(trials))) * (1000/float(hist_bin)) )
    
    plt.plot(x_hist, y_bar, '.-')
    
    #pdb.set_trace()
    if smooth:
        spiketrains = []
        spiketrains += [n.SpikeTrain(trial_i[np.logical_and(trial_i >= tws[0], trial_i < tws[1])] * pq.ms, t_start = tws[0] * pq.ms, t_stop = tws[1] * pq.ms) for trial_i in trials]
        X_ep = []
        X_ep += [trial_i[np.logical_and(trial_i >= tws[0], trial_i < tws[1])] for trial_i in trials]
        X_ep_tot = np.sort(np.hstack(X_ep)) / 1000. #seconds
        x_hist = np.arange(tws[0], tws[1]-hist_bin+1, hist_bin)
        kern = ele.statistics.sskernel(X_ep_tot, tin=x_hist/1000.)#optimal fixed kernel bandwidth (seconds)
        binsize = hist_bin / 1000.
        y_bar_mean = ele.statistics.time_histogram(spiketrains, binsize=binsize * pq.s, output='mean') #spk histogram mean spk
        kernel, norm, m_idx = ele.statistics.make_kernel('GAU', sigma=kern['optw']*pq.s, sampling_period=binsize*pq.s) #gaussian kernel 
        y_bar_smooth = norm * signal.fftconvolve(y_bar_mean, kernel.reshape((-1,1)), mode='same')
        
        plt.plot(x_hist, y_bar_smooth, '-')
    
    plt.xlim([tws[0]-10, tws[1]+10])    
    plt.xlabel('time[ms]')
    plt.ylabel('spk/s')
    plt.show()
    
    if not(to_save is None):
        filename_figure = '%s_unit%s_%s' % (options[0][:-2], options[1], options[2])
        plt.savefig('%s%s.png' % (to_save, filename_figure), dpi=300)
        plt.close()

def scatter_plot_2conditions(trials1, trials2, tws=[0,1000], hist_bin=50, step=None, smooth=None, options=['null', 'null', 'null'], to_save=None):
    """Receive trials to plot with one trial per row for 2 conditions (histogram plotted together)
    """
    plt.figure()
    
    text_label = ['Sred>Sblu', 'Sred<Sblu']#['S2>S1', 'S2<S1']#['bias+', 'bias-']
    cmap = plt.get_cmap("Set1")
    plt.subplot(311)    
    plt.title('file: %s, unit: %s, task:%s' % (options[0], options[1], options[2]), fontsize=10)
    
    for i_plot, trials in enumerate([trials1, trials2]):
        plt.subplot(3, 1, i_plot+1)
        for i_trial in range(len(trials)):
            trial_to_plot = trials[i_trial][np.logical_and(trials[i_trial]>=tws[0], trials[i_trial]<tws[1])]
            plt.plot(trial_to_plot, np.repeat(i_trial+1, len(trial_to_plot)), '|', color=cmap.colors[i_plot+2], markersize=2)
            
        plt.xlim([tws[0]-10, tws[1]+10])    
        plt.ylim([0,len(trials)+1])
        #plt.xlabel('time[ms]')
        plt.ylabel('#trial')
        
        plt.subplot(3, 1, 3)
        if step is None:
            step = hist_bin
        x_hist = np.arange(tws[0], tws[1]-hist_bin+1, step)
        y_bar = []
        for i_bin, bin_i in enumerate(x_hist):
            n_event = []
            n_event += [np.sum(np.logical_and(trials[i_trial]>=bin_i, trials[i_trial]<bin_i+hist_bin)) for i_trial in range(len(trials))]
            y_bar.append( (np.sum(n_event)/float(len(trials))) * (1000/float(hist_bin)) )
        
        plt.plot(x_hist, y_bar, '.-', color=cmap.colors[i_plot+2], label=text_label[i_plot])
        
        #pdb.set_trace()
        if smooth:
            spiketrains = []
            spiketrains += [n.SpikeTrain(trial_i[np.logical_and(trial_i >= tws[0], trial_i < tws[1])] * pq.ms, t_start = tws[0] * pq.ms, t_stop = tws[1] * pq.ms) for trial_i in trials]
            X_ep = []
            X_ep += [trial_i[np.logical_and(trial_i >= tws[0], trial_i < tws[1])] for trial_i in trials]
            X_ep_tot = np.sort(np.hstack(X_ep)) / 1000. #seconds
            x_hist = np.arange(tws[0], tws[1]-hist_bin+1, hist_bin)
            kern = ele.statistics.sskernel(X_ep_tot, tin=x_hist/1000.)#optimal fixed kernel bandwidth (seconds)
            binsize = hist_bin / 1000.
            y_bar_mean = ele.statistics.time_histogram(spiketrains, binsize=binsize * pq.s, output='mean') #spk histogram mean spk
            kernel, norm, m_idx = ele.statistics.make_kernel('GAU', sigma=kern['optw']*pq.s, sampling_period=binsize*pq.s) #gaussian kernel 
            y_bar_smooth = norm * signal.fftconvolve(y_bar_mean, kernel.reshape((-1,1)), mode='same')
            
            plt.plot(x_hist, y_bar_smooth, '-', color=cmap.colors[i_plot+2])
        
    plt.xlim([tws[0]-10, tws[1]+10])    
    plt.xlabel('time[ms]')
    plt.ylabel('spk/s')
    plt.show()    
    plt.legend()
    
    if not(to_save is None):
        filename_figure = '%s_unit%s_%s' % (options[0][:-2], options[1], options[2])
        plt.savefig('%s%s.png' % (to_save, filename_figure), dpi=300)
        plt.close()