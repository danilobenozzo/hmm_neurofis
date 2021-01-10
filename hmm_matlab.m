function hmm_matlab(task, filename_input, filename_output)
%hmm_matlab
%   ...
load(filename_input); %matlab_functions/input_matlab
%filename_output = 'matlab_functions/output_matlab.mat';
if task == 0
    if iscell(X)
        Xnew = cell(size(X));
        for i=1:size(Xnew,2)
            Xnew{i} = X{i}';
        end
    else
        Xnew = X;
    end
    [n_components, n_symbols] = size(emissionprob_prior);
    pseudo_transmat = abs(randn(n_components, n_components)) * 0.01;
    pseudo_transmat = pseudo_transmat ./ repmat(sum(pseudo_transmat,2),1,n_components);
    pseudo_emissionmat = abs(randn(n_components, n_symbols)) * 0.01;
    pseudo_emissionmat = pseudo_emissionmat ./ repmat(sum(pseudo_emissionmat,2),1,n_symbols);
    [transmat,emissionprob] = hmmtrain(Xnew,transmat_prior,emissionprob_prior,'Symbols',symbols,'Tolerance',tol,'Maxiterations',n_iter,'Pseudoemissions',pseudo_emissionmat,'Pseudotransitions',pseudo_transmat);
    save(filename_output, 'transmat', 'emissionprob', '-v6')
    return
elseif task == 1
    [state_prob,logprob] = hmmdecode(X',transmat,emissionprob,'Symbols',symbols);
    Z = hmmviterbi(X',double(transmat),emissionprob,'Symbols',symbols);
    save(filename_output, 'state_prob', 'logprob', 'Z', '-v6')
    
    to_plot = false;
    tws = [-400,400];
    bin_size = 1;
    n_cell = length(symbols)-1;
    if to_plot
        x_axis = tws(1):bin_size:tws(2)-bin_size;
        seq_toplot = double(X);
        seq_toplot(seq_toplot==0) = NaN;
        figure()
        hold on
        [ax, h1, h2] = plotyy(x_axis, seq_toplot, x_axis, state_prob);
        h1.Marker = '.';
        h1.MarkerSize = 10;
        h1.LineStyle = 'none';
        h1.Color = 'k';
        ax(2).YLim = [0,1.1];
        ax(2).YTick = [0:0.5:1];
        ax(2).YColor = 'k';
        ax(1).YLim = [0 n_cell+1];
        ax(1).YTick = [0:n_cell];
        ax(1).YColor = 'k';
        xlim(tws)
        xlabel('time[s]')
        ylabel(ax(1),'#cell')
        ylabel(ax(2),'prob state')
        title('Poisson spike trains','fontweight','normal')
        set(ax,'fontsize',15,'color','none');
        hold off
    end  
    return
elseif task == 2
    Z = hmmviterbi(X',transmat,emissionprob,'Symbols',symbols);
    save(filename_output, 'Z', '-v6')
    return
else return
    
end
end

