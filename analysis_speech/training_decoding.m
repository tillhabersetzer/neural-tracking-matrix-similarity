function training_decoding(settings, subject, condition, run_nested_cv)
%--------------------------------------------------------------------------
% Till Habersetzer, 27.01.2025
% Communication Acoustics, CvO University Oldenburg
% till.habersetzer@uol.de 
%
%   Description:
%   This function implements a complete subject-level training and testing
%   pipeline for a neural decoding model. It utilizes the mTRF-Toolbox to 
%   construct linear backward models (decoders) that reconstruct continuous 
%   stimulus features from multichannel M/EEG data. 
%
%   The pipeline handles data from single or multiple pooled sessions. It
%   features a robust two-pronged evaluation strategy:
%   A) A nested cross-validation procedure to rigorously evaluate within-corpus
%      (audiobook) performance without data leakage.
%   B) A full-dataset cross-validation and training phase to build the model 
%      for out-of-corpus (OLSA) testing.
%
%   Key Steps:
%   1.  **Load Data:** Loads preprocessed, epoched neural and audio data for
%       the specified subject, condition and sensor type.
%   2.  **Normalize Data:** Applies max-absolute scaling to the audio envelope 
%       to standardize inputs. Applies global z-score normalization to neural 
%       data to preserve the relative power and spatial topography across 
%       channels.
%   3.  **Within-Corpus Evaluation (Nested CV):** Randomly partitions the data 
%       into X folds (e.g. 5 folds if 80/20 splits). An inner loop tunes 
%       the regularization parameter (lambda) strictly on the training set, 
%       while the outer loop evaluates the model on the completely held-out
%       test set. This prevents data leakage and ensures un-inflated correlation values.
%   4.  **Optimize Final Lambda:** Performs a standard cross-validation procedure 
%       across 100% of the audiobook trials to identify a single, globally 
%       optimal regularization parameter for the final model.
%   5.  **Train Final Model:** Trains the final backward model on the 
%       entire 100% of the audiobook data using the optimal lambda. 
%       This maximizes the model's robustness for subsequent application to 
%       independent datasets (e.g., OLSA).
%   6.  **Interpret Model Weights:** Computes a standard forward encoding model 
%       and transforms the backward model weights into a forward representation 
%       to allow for neurophysiological interpretation.
%   7.  **Save Results:** Saves the trained model, nested CV metrics, full CV 
%       metrics, forward models, and key parameters into a single `.mat` file.
%
%   Syntax:
%   training_decoding(settings, subject, condition)
%
%   Inputs:
%   settings (struct):       Configuration struct containing all necessary
%                            paths, filenames, and parameters for the decoding
%                            analysis (e.g., mTRF time lags, lambda range).
%   subject (string/char):   Subject identifier (e.g., 'sub-01').
%   condition (string/char): Specifies the dataset to use. Can be a single
%                            session ('ses-01', 'ses-02') or pooled across all
%                            sessions ('ses-pooled').
%   run_nested_cv (logical): [Optional] Flag to execute the nested cross-validation
%                            block. Defaults to false if omitted.
%
%   Outputs:
%   This function does not return variables to the workspace. It saves a
%   `.mat` file to the directory specified in `settings.path2decoding_training`.
%   The file contains a `results` struct with the following fields:
%   -   cv (struct):         Full cross-validation results from `mTRFcrossval` 
%                            on 100% of the data.
%   -   lambda (double):     The globally optimal regularization parameter.
%   -   n_trials (double):   Total number of audiobook trials used.
%   -   model (struct):      The final backward TRF model trained on 100% of 
%                            the data, containing weights (`w`), bias (`b`), etc.
%   -   stats_test (struct): Results from the nested cross-validation, containing 
%                            un-inflated prediction statistics (`sorted` and 
%                            `shuffled`), fold assignments (`train_trials_folds`, 
%                            `test_trials_folds`), and inner-loop lambdas.
%   -   fmodel (struct):     Forward representations for interpretation, containing 
%                            the standard forward model (`fmodel_orig`) and the 
%                            transformed backward model (`fmodel_transformed`).
%--------------------------------------------------------------------------

% Default to false if the 4th argument is not provided
if nargin < 4
    run_nested_cv = false;
end

%% Script settings 
%--------------------------------------------------------------------------

% Get selected sensors
sensors4analysis = settings.analysis.sensors;
n_sens           = length(sensors4analysis);

% Apply zscoring of all trials
apply_normalization = settings.decoding.apply_normalization;

% Sanity check plots
plot_results = false;

% Computational parameters
bpfreq           = settings.decoding.bpfreq;
decoder_intgrwin = sort(settings.decoding.decoder.intgrwin,'ascend');

%% Loop over sensors
%--------------------------------------------------------------------------

for sens_idx = 1:n_sens % Loop over sensors / channels

    sensor = sensors4analysis{sens_idx};
 
    %% Import audiobook data 
    %----------------------------------------------------------------------
    switch condition
        case {'ses-01','ses-02'}
            session                = condition;
            % Audiobook
            fname                  = sprintf(settings.fnames.preprocessed_audiobooks,subject,session,sensor,settings.helper.formatFreqBand(bpfreq));  
            data                   = importdata(fullfile(settings.path2decoding_preprocessing(subject),fname));   
            epochs_audio_audiobook = data.epochs_audio;
            epochs_neuro_audiobook = data.epochs_neuro;
            n_trials               = length(epochs_audio_audiobook);
            clear data

        case {'ses-03','ses-pooled'} % after first run it is renamed to ses-pooled
            condition              = 'ses-pooled'; % rename condition
            n_ses                  = 2;
            epochs_audio_audiobook = cell(1,n_ses);
            epochs_neuro_audiobook = cell(1,n_ses);

            for ses_idx = 1:2
                session                         = sprintf('ses-0%d',ses_idx);
                % Audiobook
                fname                           = sprintf(settings.fnames.preprocessed_audiobooks,subject,session,sensor,settings.helper.formatFreqBand(bpfreq));  
                data                            = importdata(fullfile(settings.path2decoding_preprocessing(subject),fname));   
                epochs_audio_audiobook{ses_idx} = data.epochs_audio;
                epochs_neuro_audiobook{ses_idx} = data.epochs_neuro;
                clear data
            end

            % Concatenate data over both sessions for z-score and
            % further analysis
            %----------------------------------------------------------
            % Neuro
            cfg                    = [];
            cfg.keepsampleinfo     = 'no'; 
            epochs_neuro_audiobook = ft_appenddata(cfg, epochs_neuro_audiobook{:});
            % Audio
            epochs_audio_audiobook = [epochs_audio_audiobook{:}];
            n_trials               = length(epochs_audio_audiobook);

        otherwise
            error('%s: Unexpected condition requested (condition)!',condition)
    end
    clear data
    fprintf("Audiobook data from %s/%s/%s is loaded.\n",subject,condition,sensor);

    %% Apply normalization: z-scoring & amplitude scaling
    %------------------------------------------------------------------
    % Z-scoring is applied using a single mean and standard deviation
    % calculated from all data points across all trials combined. This ensures
    % a consistent normalization across the entire dataset.
    
    % For neuro data, this global approach preserves the relative amplitude
    % differences (i.e., the topography) between channels, which would be
    % lost if each channel were z-scored independently.
    % For the audio data, the relative power across features is preserved.
    
    % Note: Magnetometers and gradiometers are normalized separately before 
    % being combined for the analysis, due to their different physical units 
    % and scales.

    if apply_normalization
        % Z-score
        %--------
        % only on neuro data
        epochs_neuro_audiobook = apply_zscore_epochs(epochs_neuro_audiobook,sensor);
        % epochs_audio_audiobook = apply_zscore_epochs(epochs_audio_audiobook,'audio');

        % Max-abs-scaling
        %----------------
        % only on audio data
        % epochs_neuro_audiobook = apply_scaling_epochs(epochs_neuro_audiobook,sensor);
        epochs_audio_audiobook = apply_scaling_epochs(epochs_audio_audiobook,'audio');

        fprintf('Normalization to neuro and audio data applied.\n')
    end

    %% Train & Test decoder on Audiobooks (Nested Cross-Validation)
    %----------------------------------------------------------------------
    % - Only used for validation of model on audiobooks!
    % - Not evaluated on OLSA at all!

    % lambda_exp = -6:2:6;
    lambda_exp = settings.decoding.decoder.lambda_exp;
    lambdas    = 10.^(lambda_exp );

    % Check the function argument flag
    if run_nested_cv
        fprintf('Starting Nested Cross-Validation for Audiobooks...\n');

        % Extract the exact train and test indices for each fold
        %----------------------------------------------------------------------
        % Determine the number of folds based on the requested split
        % (e.g., an 80% training portion means a 20% test portion -> 5 folds)
        n_folds = round(1 / (1 - settings.decoding.decoder.n_portion_training));
    
        % Randomly shuffle all trial indices
        rng("shuffle")
        randomized_trials = randperm(n_trials);
    
        % Assign each of the shuffled trials to one of the folds.
        % This 'mod' trick deals the trials out sequentially (1,2,3,4,5,1,2,3...), 
        % ensuring the folds are as perfectly equal in size as mathematically possible.
        fold_ids = mod(0:n_trials-1, n_folds) + 1;
        
        % Initialize cell arrays to store the assignments
        train_idx = cell(n_folds, 1);
        test_idx  = cell(n_folds, 1);
       
        for fold_idx = 1:n_folds
            % The test trials are the ones assigned to the current fold 'k'
            test_idx{fold_idx} = randomized_trials(fold_ids == fold_idx);
            
            % The training trials are simply all the rest
            train_idx{fold_idx} = randomized_trials(fold_ids ~= fold_idx);
        end
    
        % Perform Nested Cross-Validation
        %--------------------------------
        opt_lambda   = nan(n_folds,1); % Store optimal lambdas
        r_sorted     = cell(n_folds, 1); % Correlation value for each test trial
        err_sorted   = cell(n_folds, 1); % Error value for each test trial
        r_shuffled   = cell(n_folds, 1); % Number of shuffled trials depends on possible permutations
        err_shuffled = cell(n_folds, 1); % Number of shuffled trials depends on possible permutations
    
        for fold_idx = 1:n_folds
    
            % Inner Cross-validation
            %-----------------------
            cv = mTRFcrossval(epochs_audio_audiobook(train_idx{fold_idx}), ... % stimulus
                              epochs_neuro_audiobook.trial(train_idx{fold_idx}), ... % response
                              epochs_neuro_audiobook.fsample, ... % sampling rate (Hz)
                              settings.decoding.decoder.drct, ... % direction
                              decoder_intgrwin(1), ... % minimum time lag (ms)
                              decoder_intgrwin(2), ... % maximum time lag (ms)
                              lambdas, ... % regularization values
                              'dim', settings.decoding.decoder.dim,... % work along the rows, observations in columns
                              'zeropad', settings.decoding.decoder.zeropad, ... % zero-pad the outer rows of the design matrix or delete them
                              'fast', settings.decoding.decoder.fast, ... % use the fast cross-validation method (requires more memory)
                              'corr', settings.decoding.decoder.corr_metric, ... % Spearman's rank correlation coefficient
                              'type', settings.decoding.decoder.type); % use all lags simultaneously to fit a multi-lag model
    
            % Get optimal  regularization hyperparameter
            [~, idx]             = max(mean(cv.r, 1));
            lambda               = lambdas(idx);
            opt_lambda(fold_idx) = lambda; 
    
            % Training & Testing
            %-------------------
            model = mTRFtrain(epochs_audio_audiobook(train_idx{fold_idx}), ...
                              epochs_neuro_audiobook.trial(train_idx{fold_idx}), ...
                              epochs_neuro_audiobook.fsample, ...
                              settings.decoding.decoder.drct, ... 
                              decoder_intgrwin(1), ...
                              decoder_intgrwin(2), ...
                              lambda, ...
                              'dim', settings.decoding.decoder.dim, ...
                              'zeropad', settings.decoding.decoder.zeropad, ...
                              'type', settings.decoding.decoder.type);
    
             [pred_test, stats_sorted] = mTRFpredict(epochs_audio_audiobook(test_idx{fold_idx}), ...
                                                     epochs_neuro_audiobook.trial(test_idx{fold_idx}), ...
                                                     model, ...
                                                     'zeropad', settings.decoding.decoder.zeropad, ...
                                                     'dim', settings.decoding.decoder.dim, ...
                                                     'corr', settings.decoding.decoder.corr_metric);
    
             % Store
             r_sorted{fold_idx}   = stats_sorted.r(:);
             err_sorted{fold_idx} = stats_sorted.err(:);
    
             % Correlation shuffled data
             %--------------------------
             pred_test  = cellfun(@transpose, pred_test, 'UniformOutput', false)';
             circ_shift = false; % trials not circularly shifted 
        
             stats_shuffled = compute_null_distribution_single_trials(epochs_audio_audiobook(test_idx{fold_idx}), ...
                                                                      pred_test, ...
                                                                      settings.decoding.decoder.dim, ...
                                                                      settings.decoding.decoder.corr_metric, ...
                                                                      circ_shift, ...
                                                                      []); % fs_audio
    
             % Clear
             r_shuffled{fold_idx}   = stats_shuffled.r(:);
             err_shuffled{fold_idx} = stats_shuffled.err(:);
    
             clear cv pred_test stats_sorted stats_shuffled model lambda
    
             fprintf('Outer Nested Cross-Validation Fold %i/%i for Audiobooks completed.\n', fold_idx, n_folds)
        end
        fprintf('Nested Cross-Validation for Audiobooks completed.\n')
    
        % Collect Results
        %----------------
        % Accumulate data across folds
        stats_test                    = struct();
        stats_test.sorted.r           = vertcat(r_sorted{:})';
        stats_test.sorted.err         = vertcat(err_sorted{:})';
        stats_test.shuffled.r         = vertcat(r_shuffled{:})';
        stats_test.shuffled.err       = vertcat(err_shuffled{:})';
        stats_test.train_trials_folds = train_idx;
        stats_test.test_trials_folds  = test_idx;
        stats_test.lambda             = opt_lambda';
    else
        % Skip the block and initialize empty struct so saving doesn't crash
        fprintf('Skipping Nested Cross-Validation for Audiobooks (flag disabled).\n');
        stats_test = struct();
    end % nested cv
    
    %% Train Decoder for OLSA
    %----------------------------------------------------------------------
   
    % Cross-validation for optimization
    %----------------------------------
    % To optimize the decoders ability to predict stimulus features from new 
    % MEG/EEG data, we tune the regularization parameter using an efficient 
    % leave-one-out cross-validation (CV) procedure.
    
    % Run fast cross-validation on all data
    cv = mTRFcrossval(epochs_audio_audiobook, ... % stimulus
                      epochs_neuro_audiobook.trial, ... % response
                      epochs_neuro_audiobook.fsample, ... % sampling rate (Hz)
                      settings.decoding.decoder.drct, ... % direction
                      decoder_intgrwin(1), ... % minimum time lag (ms)
                      decoder_intgrwin(2), ... % maximum time lag (ms)
                      lambdas, ... % regularization values
                      'dim',settings.decoding.decoder.dim,... % work along the rows, observations in columns
                      'zeropad',settings.decoding.decoder.zeropad, ... % zero-pad the outer rows of the design matrix or delete them
                      'fast',settings.decoding.decoder.fast, ... % use the fast cross-validation method (requires more memory)
                      'corr', settings.decoding.decoder.corr_metric, ... % Spearman's rank correlation coefficient
                      'type', settings.decoding.decoder.type); % use all lags simultaneously to fit a multi-lag model
    
    if plot_results
        % Plot CV accuracy 
        %-----------------
        nlambda = length(lambdas);
        nfold   = n_trials_train;
        figure
        subplot(1,2,1), errorbar(1:numel(lambdas),mean(cv.r),std(cv.r)/sqrt(nfold-1),'linewidth',2)
        set(gca,'xtick',1:nlambda,'xticklabel',lambda_exp), xlim([0,numel(lambdas)+1]), axis square, grid on
        title('CV Accuracy'), xlabel('Regularization (1\times10^\lambda)'), ylabel('Correlation')
        % Plot CV error
        %--------------
        subplot(1,2,2), errorbar(1:numel(lambdas),mean(cv.err),std(cv.err)/sqrt(nfold-1),'linewidth',2)
        set(gca,'xtick',1:nlambda,'xticklabel',lambda_exp), xlim([0,numel(lambdas)+1]), axis square, grid on
        title('CV Error'), xlabel('Regularization (1\times10^\lambda)'), ylabel('MSE')
    end
    
    % Get optimal  regularization hyperparameter
    % (nfold-by-nlambda-by-yvar) == (20x7lambdas)
    [~, idx] = max(mean(cv.r, 1));
    lambda   = lambdas(idx);
    
    % Train backword model on training data
    %--------------------------------------   
    model = mTRFtrain(epochs_audio_audiobook, ...
                      epochs_neuro_audiobook.trial, ...
                      epochs_neuro_audiobook.fsample, ...
                      settings.decoding.decoder.drct, ... 
                      decoder_intgrwin(1), ...
                      decoder_intgrwin(2), ...
                      lambda, ...
                      'dim',settings.decoding.decoder.dim, ...
                      'zeropad',settings.decoding.decoder.zeropad, ...
                      'type', settings.decoding.decoder.type);


    % Add forward model
    %----------------------------------------------------------------------
    fmodel_orig = mTRFtrain(epochs_audio_audiobook, ...
                            epochs_neuro_audiobook.trial, ...
                            epochs_neuro_audiobook.fsample, ...
                            1, ... % direction
                            decoder_intgrwin(1), ...
                            decoder_intgrwin(2), ...
                            lambda, ...
                            'dim',settings.decoding.decoder.dim, ...
                            'zeropad',settings.decoding.decoder.zeropad, ...
                            'type', settings.decoding.decoder.type);


    % Transform backward into forward model: Interpreting model weights
    %------------------------------------------------------------------
    fmodel_transformed = mTRFtransform(model, ...
                                       epochs_neuro_audiobook.trial, ...
                                       'dim',settings.decoding.decoder.dim, ...
                                       'zeropad',settings.decoding.decoder.zeropad);

    % Save forward models
    %--------------------
    fmodel                    = struct;
    fmodel.fmodel_orig        = fmodel_orig;
    fmodel.fmodel_transformed = fmodel_transformed;
    fmodel.label              = epochs_neuro_audiobook.label;
    if isfield(epochs_neuro_audiobook,'grad')
        fmodel.grad = epochs_neuro_audiobook.grad;
    end
    if isfield(epochs_neuro_audiobook,'elec')
        fmodel.elec = epochs_neuro_audiobook.elec;
    end

    clear fmodel_orig fmodel_transformed
    
    %% Save results
    %----------------------------------------------------------------------
    dir2save = settings.path2decoding_training(subject);
    if ~exist(dir2save,'dir')
        mkdir(dir2save)
    end
    fname = sprintf(settings.fnames.training_decoder,subject,condition,sensor,settings.helper.formatFreqBand(bpfreq),settings.helper.formatIntgrWin(decoder_intgrwin));

    results            = struct();
    results.cv         = cv; % cross-validation
    results.lambda     = lambda; % regularization value
    results.n_trials   = n_trials; % all trials
    results.model      = model; % trained model
    results.stats_test = stats_test; % test statistics on audiobook data
    results.fmodel     = fmodel;

    save(fullfile(dir2save,fname),'results','-v7.3'); 
    fprintf("\n%s from %s saved.\n",fname,subject)

end % Loop over sensors

end % end of function