function prediction_concatenated_sentences_olsa_similarity_decoding(settings, subject, condition)
%--------------------------------------------------------------------------
% Till Habersetzer, 21.04.2026
% Communication Acoustics, CvO University Oldenburg
% till.habersetzer@uol.de
% 
%   Description:
%   This function tests the generalization of a pre-trained neural decoding
%   model (backward TRF). It loads a model trained on continuous audiobook
%   data and applies it to M/EEG data from the OLSA sentence-in-noise task.
%
%   It performs a comprehensive, multi-baseline performance analysis,
%   calculating prediction accuracy separately for each signal-to-noise
%   ratio (SNR) present in the OLSA data.
%
%   Key Steps:
%   1.  Data Loading: Loads the preprocessed OLSA dataset and the pre-trained
%       audiobook TRF model. Handles single-session or pooled-session data.
%   2.  Normalization & Onset Removal & Prediction: 
%       Applies optional removal of onset response to audio and neuro data. 
%       Applies optional z-scoring to neural data and max-absolute-scaling 
%       to audio data. Uses the loaded model to predict speech envelopes for 
%       all OLSA trials.
%   3.  SNR-based Evaluation: For each unique SNR, it computes four distinct
%       performance metrics:
%       -   Standard Performance: Correlation between true and predicted
%           envelopes, calculated on cumulatively concatenated trials to
%           create a performance growth curve.
%       -   Shuffled Baseline: A null distribution from correlating
%           mismatched (deranged) true and predicted trials.
%       -   Time-Shifted Shuffled Baseline: A stricter null distribution
%           using mismatched trials that are also randomly time-shifted.
%       -   Mean Sentence Baseline: Correlation of predictions against the
%           average OLSA sentence envelope to test for stimulus specificity.
%   4.  Save Results: Saves all computed statistics, organized by SNR, into a
%       single .mat file.
%
%   Syntax:
%   prediction_concatenated_sentences_decoding(settings, subject, condition)
%
%   Inputs:
%   settings (struct):      A structure with all necessary paths and parameters.
%   subject (char/string):  The subject identifier (e.g., 'sub-01').
%   condition (char/string):The session to process ('ses-01', 'ses-02') or
%                           pool ('ses-pooled').
%
%   Outputs:
%   This function does not return any variables to the workspace but saves
%   a .mat file to the directory specified in 'settings.path2decoding'.
%   The saved file contains a 'results' struct with the computed
%   'stats_olsa' and 'event_description'.
%
%--------------------------------------------------------------------------

%% Script settings 
%--------------------------------------------------------------------------

% Get selected sensors
sensors4analysis = settings.analysis.sensors;
n_sens           = length(sensors4analysis);

% Apply zscoring of all trials
apply_normalization = settings.decoding.apply_normalization;

% Computational parameters
bpfreq           = settings.decoding.bpfreq;
decoder_intgrwin = sort(settings.decoding.decoder.intgrwin,'ascend');

% Remove onset (Load preprocessed data without OLSA sentence onsets)
remove_onset = settings.remove_onset;

%% Loop over sensors
%--------------------------------------------------------------------------

for sens_idx = 1:n_sens % Loop over sensors / channels

    sensor = sensors4analysis{sens_idx};
     
    %% Import olsa data
    %----------------------------------------------------------------------
    switch condition
        case {'ses-01','ses-02'}
            session                     = condition;
            % Olsa
            fname                       = sprintf(settings.fnames.preprocessed_olsa,subject,session,sensor,settings.helper.formatFreqBand(bpfreq));
            data                        = importdata(fullfile(settings.path2decoding_preprocessing(subject),fname));   
            epochs_audio_olsa           = data.epochs_audio;
            epochs_neuro_olsa           = data.epochs_neuro;
            epochs_audio_detection_olsa = data.epochs_audio_detection;
            event_description           = data.event_description;
            clear data

            % Audiobook - only to check for matching channels
            fname                        = sprintf(settings.fnames.preprocessed_audiobooks,subject,session,sensor,settings.helper.formatFreqBand(bpfreq));  
            data                         = importdata(fullfile(settings.path2decoding_preprocessing(subject),fname));   
            epochs_neuro_audiobook_label = data.epochs_neuro.label;
            clear data

        case {'ses-03','ses-pooled'}
            condition                   = 'ses-pooled'; % rename condition
            n_ses                       = 2; 
            epochs_audio_olsa           = cell(1,n_ses);
            epochs_neuro_olsa           = cell(1,n_ses);
            epochs_audio_detection_olsa = cell(1,n_ses);
            event_descriptions          = cell(1,n_ses);

            for ses_idx = 1:2
                session = sprintf('ses-0%d',ses_idx);
       
                % Olsa
                fname                                = sprintf(settings.fnames.preprocessed_olsa,subject,session,sensor,settings.helper.formatFreqBand(bpfreq));
                data                                 = importdata(fullfile(settings.path2decoding_preprocessing(subject),fname));   
                epochs_audio_olsa{ses_idx}           = data.epochs_audio;
                epochs_neuro_olsa{ses_idx}           = data.epochs_neuro;
                epochs_audio_detection_olsa{ses_idx} = data.epochs_audio_detection;
                event_descriptions{ses_idx}          = data.event_description;
                clear data 
                
                % Audiobook - only to check for matching channels
                fname = sprintf(settings.fnames.preprocessed_audiobooks,subject,session,sensor,settings.helper.formatFreqBand(bpfreq));  
                data  = importdata(fullfile(settings.path2decoding_preprocessing(subject),fname));  
                if ses_idx==2
                    [chanidx1,chanidx2] = match_str(epochs_neuro_audiobook_label,data.epochs_neuro.label);
                    if ~isequal(chanidx1,chanidx2)
                        error('Channel labels have different order across sessions!')
                    end
                    clear chanidx1 chanidx2
                end
                epochs_neuro_audiobook_label = data.epochs_neuro.label;
                clear data
            end

            % Concatenate data over both sessions for z-score and
            % further analysis
            %----------------------------------------------------------
            % Neuro
            cfg                = [];
            cfg.keepsampleinfo = 'no'; 
            epochs_neuro_olsa  = ft_appenddata(cfg, epochs_neuro_olsa{:});
            % Audio
            epochs_audio_olsa           = [epochs_audio_olsa{:}];
            epochs_audio_detection_olsa = [epochs_audio_detection_olsa{:}];

            % Event_description
            event_description                = event_descriptions{1};
            event_description.snr_data       = vertcat(event_descriptions{1}.snr_data,event_descriptions{2}.snr_data);
            event_description.intelli_data   = vertcat(event_descriptions{1}.intelli_data,event_descriptions{2}.intelli_data);
            event_description.playlist_data  = vertcat(event_descriptions{1}.playlist_data,event_descriptions{2}.playlist_data);
            event_description.mapping_epochs = vertcat(event_descriptions{1}.mapping_epochs,event_descriptions{2}.mapping_epochs+event_descriptions{1}.mapping_epochs(end)); % add offset
            event_description.mapping_labels = vertcat(event_descriptions{1}.mapping_labels',event_descriptions{2}.mapping_labels');

            % Sanity check: snr + intelligibility
            if ~isequal(event_description.snr_data,epochs_neuro_olsa.trialinfo(:,3)); error('%s: Unexpected snr_data!', subject); end
            if ~isequal(event_description.intelli_data,epochs_neuro_olsa.trialinfo(:,4)); error('%s: Unexpected intelli_data!', subject); end

        otherwise
            error('%s: Unexpected condition requested (condition)!',condition)
    end
    clear data
    fprintf("Audiobook and olsa data from %s/%s/%s is loaded.\n",subject,condition,sensor);

    %% Import trained model and audio files
    %----------------------------------------------------------------------
    % Trained model
    fname = sprintf(settings.fnames.training_decoder,subject,condition,sensor,settings.helper.formatFreqBand(bpfreq),settings.helper.formatIntgrWin(decoder_intgrwin));
    data  = importdata(fullfile(settings.path2decoding_training(subject),fname));   
    model = data.model;
    clear data

    % Audio files
    fname                 = sprintf(settings.fnames.audio_olsa, settings.helper.formatFreqBand(bpfreq));     
    audio_envelopes       = importdata(fullfile(settings.path2derivatives,'stimuli',fname));
    audio_field_names_all = fieldnames(audio_envelopes);
    audio_field_names     = audio_field_names_all(startsWith(audio_field_names_all, 'envelope'));
    fs                    = audio_envelopes.fs;

    % '\d+$' is a regular expression pattern that means "ends with one or more digits"
    match_idx         = ~cellfun(@isempty, regexp(audio_field_names, '\d+$'));
    fnames_audiofiles = audio_field_names(match_idx);
    % Only extract numbers as filenames
    fnames_audiofiles = regexp(fnames_audiofiles, '\d+$', 'match', 'once');
    n_sentences       = length(fnames_audiofiles);

    %% Optional: Remove onset response
    %----------------------------------------------------------------------
    if remove_onset
        % Remove first 500ms (hard-coded)
        onset_tmax = settings.onset_tmax;
        onset_idxs = find(epochs_neuro_olsa.time{1} <= onset_tmax);
        start_idx  = onset_idxs(end) +1;

        % Correct Olsa neuro and audio epochs
        %------------------------------------
        n_trials  = length(epochs_neuro_olsa.trial);
        for trl_idx = 1:n_trials
            epochs_neuro_olsa.trial{trl_idx} = epochs_neuro_olsa.trial{trl_idx}(:,start_idx:end);
            epochs_neuro_olsa.time{trl_idx}  = epochs_neuro_olsa.time{trl_idx}(:,start_idx:end);

            epochs_audio_olsa{trl_idx}           = epochs_audio_olsa{trl_idx}(start_idx:end);
            epochs_audio_detection_olsa{trl_idx} = epochs_audio_detection_olsa{trl_idx}(start_idx:end);
        end

        % Correct each audiofile
        %-----------------------
        for f_idx = 1:length(audio_field_names_all)
            field_name = audio_field_names_all{f_idx};
            % Apply for all fields except sampling frequency
            if ~strcmp(field_name, 'fs')
                data_field                   = audio_envelopes.(field_name);
                audio_envelopes.(field_name) = data_field(start_idx:end);
            end
            clear data_field
        end
        clear field_name data_field
    end

    %% Apply normalization
    %----------------------------------------------------------------------
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
        %------------------------------------------------------------------
        % only on neuro data
        epochs_neuro_olsa = apply_zscore_epochs(epochs_neuro_olsa,sensor);
        % epochs_audio_olsa = apply_zscore_epochs(epochs_audio_olsa,'audio');
        % mean_envelope = (mean_envelope-mean(mean_envelope,'all'))/std(mean_envelope,0,'all');

        % Normalise each audio file
        % for f_idx = 1:numel(audio_field_names)
        %     field_name                   = audio_field_names{f_idx};
        %     sentence_audio               = audio_envelopes.(field_name);
        %     sentence_audio               = (sentence_audio-mean(sentence_audio,'all'))./std(sentence_audio,0,'all');
        %     audio_envelopes.(field_name) = sentence_audio;
        % end


        % Max-abs-scaling
        %------------------------------------------------------------------
        % only on audio data
        % epochs_neuro_olsa = apply_scaling_epochs(epochs_neuro_olsa,sensor);
        epochs_audio_olsa = apply_scaling_epochs(epochs_audio_olsa,'audio');

        % Normalise each audio file
        for f_idx = 1:numel(audio_field_names)
            field_name                   = audio_field_names{f_idx};
            sentence_audio               = audio_envelopes.(field_name);
            sentence_audio               = sentence_audio ./ max(abs(sentence_audio), [], 2);
            audio_envelopes.(field_name) = sentence_audio;
        end

        fprintf('Normalization to neuro and audio data applied.\n')
    end

    %% Compute predication scores - Preparations
    %----------------------------------------------------------------------
    
    % Check if channel order is identical in olsa and audiobook meg recordings
    %--------------------------------------------------------------------------
    [chanidx1,chanidx2] = match_str(epochs_neuro_audiobook_label,epochs_neuro_olsa.label);
    if ~isequal(chanidx1,chanidx2)
        error('Channel labels have different order!')
    end
    clear chanidx1 chanidx2

    % Compute prediction for each sentence
    %----------------------------------------------------------------------
    % Includes speech, but also post stimulus timewindow, cause it is 
    % needed for temporal integration of decoder
    [pred_olsa,~] = mTRFpredict(epochs_audio_olsa, ...
                                epochs_neuro_olsa.trial, ...
                                model,...
                                'zeropad', settings.decoding.decoder.zeropad, ...
                                'dim', settings.decoding.decoder.dim, ...
                                'corr', settings.decoding.decoder.corr_metric); 

    % Transpose olsa predictions for conformity
    pred_olsa = cellfun(@transpose, pred_olsa, 'UniformOutput', false)';

    % Prepare snrs to loop over it
    %----------------------------------------------------------------------
    
    % Group according to SNR  
    snr_data         = event_description.snr_data;
    [n_trials, snrs] = groupcounts(snr_data);
    [snrs, sort_idx] = sort(snrs,'ascend');
    n_trials         = n_trials(sort_idx);
    n_snrs           = length(snrs);
    % Intelligibilities
    intelli_data     = event_description.intelli_data;
    [~, intellis]    = groupcounts(intelli_data);
    intellis         = sort(intellis,'ascend');
    n_intellis       = length(intellis);

    if n_snrs ~= n_intellis
        error('Number of snrs (%d) and intelligibilities (%d) are not the same!', n_snrs, n_intellis)
    end

    % Compute maximum number of samples for shift (minimum duration of cropped sentence)
    all_fields        = fieldnames(audio_envelopes);
    fields2rem        = all_fields(cellfun(@isempty, regexp(all_fields, 'audio_detection_\d{5}'))); % fields for removal
    filtered_struct   = rmfield(audio_envelopes, fields2rem); % filter out fields
    all_sums          = structfun(@(x) sum(x(:)), filtered_struct);
    max_shift_samples = min(all_sums);
    clear all_fields fields2rem filtered_struct all_sums

    %% Compute predication scores - Loop over snrs
    %----------------------------------------------------------------------

    % Initialize data
    %----------------
    n_perms                      = 1000; % Number of permutations for null distribution
    % max_shift_samples            = fs; % shift up to 1s
    % shift_samples                = 0:max_shift_samples-1;
    shift_max                    = floor(max_shift_samples / 2); % Symmetric around 0
    shift_samples                = -shift_max:1:shift_max;
    n_shifts                     = length(shift_samples);

    % 1.)
    stats_sorted                 = struct();
    stats_sorted.r               = nan(n_snrs,1);
    stats_sorted.err             = nan(n_snrs,1);
    % 2.)
    stats_sorted_circshift.r     = struct();
    stats_sorted_circshift.r     = nan(n_snrs,  n_perms);
    stats_sorted_circshift.err   = nan(n_snrs,n_perms);
    % 3.)
    stats_shuffled               = struct();
    stats_shuffled.r             = nan(n_snrs, n_perms);
    stats_shuffled.err           = nan(n_snrs, n_perms);
    % 4.)
    stats_shuffled_circshift     = struct();
    stats_shuffled_circshift.r   = nan(n_snrs, n_perms);
    stats_shuffled_circshift.err = nan(n_snrs, n_perms);
    % 5.)
    stats_sentences              = struct();
    stats_sentences.fnames       = fnames_audiofiles;
    stats_sentences.r            = nan(n_snrs, n_sentences);
    stats_sentences.err          = nan(n_snrs, n_sentences);
    % 6.)
    stats_mean_sentence          = struct();
    stats_mean_sentence.r        = nan(n_snrs,1);
    stats_mean_sentence.err      = nan(n_snrs,1);
    % 7.)
    stats_shift                  = struct();
    stats_shift.shift_samples    = shift_samples;
    stats_shift.shift_time       = shift_samples/audio_envelopes.fs;
    stats_shift.r                = nan(n_snrs, n_shifts);
    stats_shift.err              = nan(n_snrs, n_shifts);

    for snr_idx = 1:n_snrs

        % Get subselection of trials for each snr
        %------------------------------------------------------------------
        snr          = snrs(snr_idx);
        idx4snr      = snr_data==snr;
        n_trials_snr = sum(idx4snr);
        if ~ismember(n_trials_snr,[40,80,100,200]) 
            error('Unexpected number of sentences for %s (%d)!',subject, n_trials_snr)
        end

        % Subselect snr trials and apply speech detection window
        epochs_audio = cellfun(@(x,y) x(:,logical(y)), epochs_audio_olsa(idx4snr), epochs_audio_detection_olsa(idx4snr), 'UniformOutput', false);
        pred_audio   = cellfun(@(x,y) x(:,logical(y)), pred_olsa(idx4snr), epochs_audio_detection_olsa(idx4snr), 'UniformOutput', false);

        % 1.) Correlations on matched true sentences
        %------------------------------------------------------------------
        epochs_concat = [epochs_audio{:}];
        pred_concat   = [pred_audio{:}];
        
        [r, err] = mTRFevaluate(epochs_concat, ...
                                pred_concat,...
                                'error', 'mse', ...
                                'dim',settings.decoding.decoder.dim,...
                                'corr',settings.decoding.decoder.corr_metric);
      
        stats_sorted.r(snr_idx, 1)   = r;
        stats_sorted.err(snr_idx, 1) = err;
        clear epochs_concat pred_concat r err

        % 2.) Null distribution of randomly shifted sorted sentences
        %------------------------------------------------------------------
        circ_shift                 = true;
        stats_sorted_circshift_snr = compute_statistic_concatenated_trials(epochs_audio, ...
                                                                           pred_audio, ...
                                                                           n_perms, ...
                                                                           settings.decoding.decoder.dim, ...
                                                                           settings.decoding.decoder.corr_metric, ...
                                                                           circ_shift, ...
                                                                           fs, ... % shift will be at least 1s
                                                                           'audio'); % shift audio envelopes

        stats_sorted_circshift.r(snr_idx,:)   = stats_sorted_circshift_snr.r;
        stats_sorted_circshift.err(snr_idx,:) = stats_sorted_circshift_snr.err;
        clear stats_sorted_circshift_snr

        % 3.) Correlation shuffled sentences
        %------------------------------------------------------------------
        circ_shift         = false; % sentences not circularly shifted 
        stats_shuffled_snr = compute_statistic_concatenated_trials_deranged(epochs_audio, ...
                                                                            pred_audio, ...
                                                                            n_perms, ...
                                                                            settings.decoding.decoder.dim, ...
                                                                            settings.decoding.decoder.corr_metric, ...
                                                                            circ_shift, ...
                                                                            [], ...
                                                                            'audio'); % shift audio envelopes); 

        stats_shuffled.r(snr_idx,:)   = stats_shuffled_snr.r;
        stats_shuffled.err(snr_idx,:) = stats_shuffled_snr.err;
        clear stats_shuffled_snr     

        % 4.) Correlation shuffled and circshifted sentences
        %------------------------------------------------------------------
        circ_shift                   = true; 
        stats_shuffled_circshift_snr = compute_statistic_concatenated_trials_deranged(epochs_audio, ...
                                                                                      pred_audio, ...
                                                                                      n_perms, ...
                                                                                      settings.decoding.decoder.dim, ...
                                                                                      settings.decoding.decoder.corr_metric, ...
                                                                                      circ_shift, ...
                                                                                      fs); % shift will be at least 1s

        stats_shuffled_circshift.r(snr_idx,:)   = stats_shuffled_circshift_snr.r;
        stats_shuffled_circshift.err(snr_idx,:) = stats_shuffled_circshift_snr.err;
        clear stats_shuffled_circshift_snr     

        % 5.) Correlation single sentences
        %------------------------------------------------------------------
        for f_idx = 1:n_sentences
            fname = fnames_audiofiles{f_idx};

            sentence_envelope  = audio_envelopes.(sprintf('envelope_%s',fname));
            envelope_idx       = logical(audio_envelopes.(sprintf('audio_detection_%s',fname)));
    
            sentence_envelopes = repmat({sentence_envelope(envelope_idx)},1,n_trials_snr);
            lengths_audio      = cellfun(@length, sentence_envelopes);
            lengths_pred       = cellfun(@length, pred_audio);
            segment_lengths    = min(lengths_audio, lengths_pred); % minimum length of both trials
            segment_lengths    = num2cell(segment_lengths);
    
            % Concatenate sentences and apply minimum duration
            epochs_cropped     = cellfun(@(x,y) x(:,1:y), sentence_envelopes, segment_lengths, 'UniformOutput', false);
            pred_cropped       = cellfun(@(x,y) x(:,1:y), pred_audio, segment_lengths, 'UniformOutput', false);
           
            epochs_concat      = [epochs_cropped{:}];
            pred_concat        = [pred_cropped{:}];
    
            [r, err] = mTRFevaluate(epochs_concat, ...
                                    pred_concat,...
                                    'error', 'mse', ...
                                    'dim',settings.decoding.decoder.dim,...
                                    'corr',settings.decoding.decoder.corr_metric);
    
            stats_sentences.r(snr_idx, f_idx)   = r;
            stats_sentences.err(snr_idx, f_idx) = err;
            clear sentence_envelope envelope_idx sentence_envelopes epochs_cropped pred_cropped
            clear epochs_concat pred_concat r err
        end

        % 6.) Correlation mean sentence
        %------------------------------------------------------------------
        sentence_envelope  = audio_envelopes.envelope_avg;
        envelope_idx       = logical(audio_envelopes.audio_detection_avg);

        sentence_envelopes = repmat({sentence_envelope(envelope_idx)},1,n_trials_snr);
        lengths_audio      = cellfun(@length, sentence_envelopes);
        lengths_pred       = cellfun(@length, pred_audio);
        segment_lengths    = min(lengths_audio, lengths_pred); % minimum length of both trials
        segment_lengths    = num2cell(segment_lengths);

        % Concatenate sentences and apply minimum duration
        epochs_cropped     = cellfun(@(x,y) x(:,1:y), sentence_envelopes, segment_lengths, 'UniformOutput', false);
        pred_cropped       = cellfun(@(x,y) x(:,1:y), pred_audio, segment_lengths, 'UniformOutput', false);
       
        epochs_concat      = [epochs_cropped{:}];
        pred_concat        = [pred_cropped{:}];

        [r, err] = mTRFevaluate(epochs_concat, ...
                                pred_concat,...
                                'error', 'mse', ...
                                'dim',settings.decoding.decoder.dim,...
                                'corr',settings.decoding.decoder.corr_metric);

        stats_mean_sentence.r(snr_idx,1)   = r;
        stats_mean_sentence.err(snr_idx,1) = err;
        clear sentence_envelope envelope_idx sentence_envelopes epochs_cropped pred_cropped
        clear epochs_concat pred_concat r err

        % 7.) Correlation shifted sentences
        %------------------------------------------------------------------
        for s_idx = 1:n_shifts
            shift = shift_samples(s_idx);

            % Apply circshift (in samples)
            epochs_audio_shifted = cellfun(@(x) circshift(x, shift), epochs_audio, 'UniformOutput', false);

            epochs_concat = [epochs_audio_shifted{:}];
            pred_concat   = [pred_audio{:}];

            [r, err] = mTRFevaluate(epochs_concat, ...
                                    pred_concat,...
                                    'error', 'mse', ...
                                    'dim', settings.decoding.decoder.dim,...
                                    'corr', settings.decoding.decoder.corr_metric);

            stats_shift.r(snr_idx, s_idx)   = r;
            stats_shift.err(snr_idx, s_idx) = err;
            clear epochs_concat pred_concat r err epochs_audio_shifted
        end

    end % Loop over snrs

    % Store computed statistics
    %--------------------------
    stats_olsa                    = struct;
    stats_olsa.sorted             = stats_sorted;
    stats_olsa.sorted_circshift   = stats_sorted_circshift;
    stats_olsa.shuffled           = stats_shuffled;
    stats_olsa.shuffled_circshift = stats_shuffled_circshift;
    stats_olsa.single_sentences   = stats_sentences;
    stats_olsa.mean_sentence      = stats_mean_sentence;
    stats_olsa.shift              = stats_shift;
    stats_olsa.snrs               = snrs;
    stats_olsa.intelligibilities  = intellis;
    stats_olsa.n_trials_per_snr   = n_trials;

    %% Save results
    %----------------------------------------------------------------------
    dir2save = settings.path2decoding_testing(subject);
    if ~exist(dir2save,'dir')
        mkdir(dir2save)
    end

    fname = sprintf(settings.fnames.prediction_decoder_olsa_sim,subject,condition,sensor,settings.helper.formatFreqBand(bpfreq),settings.helper.formatIntgrWin(decoder_intgrwin),'concat_sentences');
    if remove_onset
        fname = strrep(fname, 'decoding', 'onset-removed_decoding');
    end

    results                   = struct();
    results.event_description = event_description;
    results.stats_olsa        = stats_olsa;
    save(fullfile(dir2save,fname),'results','-v7.3'); 
    fprintf("\n%s from %s saved.\n",fname,subject)

end % Loop over sensors 

end % end of function