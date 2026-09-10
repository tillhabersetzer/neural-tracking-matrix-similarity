%--------------------------------------------------------------------------
% Till Habersetzer, 05.07.2025
% Communication Acoustics, CvO University Oldenburg
% till.habersetzer@uol.de 
%
% This script analyzes and visualizes the performance of a pre-trained
% neural decoder. It loads preprocessed Olsa (neuro and audio) data
% for a specific subject, sensor, and session. It then uses the
% trained model (mTRF) to reconstruct the audio envelope from the
% neural data. The main output is a series of visualizations
% comparing the original and reconstructed audio, primarily to
% evaluate decoder accuracy (correlation) across different SNRs.
%--------------------------------------------------------------------------

%% Initial setup
%--------------------------------------------------------------------------
% Clears workspace, closes figures, and loads the main settings file

close all;
clearvars;
clc;

% Import main settings from an external file
settings_decoding;

%% Script settings 
%--------------------------------------------------------------------------

% Select subject
subject = 'sub-10';

% Select sensor
sensor = 'meg'; % 'meg', 'eeg', 'ear_eeg'

% Select condition
condition = 'ses-pooled'; % 'ses-01', 'ses-02', 'ses-pooled'

% Apply zscoring of all trials
apply_normalization = settings.decoding.apply_normalization;

% Computational parameters - decide for frequency band and integration
% window
bpfreq           = [0.5, 8];
decoder_intgrwin = [0, 400];

% Add Required Toolboxes to MATLAB Path 
fprintf('Adding toolboxes to path...\n');
addpath(fullfile(settings.path2project, 'analysis', 'helper_functions'));

% Initialize mTRF Toolboxes
addpath(genpath(settings.path2mtrftoolbox));

%% Import data
%--------------------------------------------------------------------------

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

% Import olsa data envelopes
%--------------------------------------------------------------------------

fname                        = sprintf(settings.fnames.audio_olsa, settings.helper.formatFreqBand(bpfreq));     
audio_envelopes              = importdata(fullfile(settings.path2derivatives,'stimuli',fname));
envelope_avg                 = audio_envelopes.envelope_avg;
envelope_avg_audio_detection = logical(audio_envelopes.audio_detection_avg);

% Extract envelopes
%------------------
all_fields      = fieldnames(audio_envelopes);
env_idx         = contains(all_fields, "envelope", "IgnoreCase", true) & ~contains(all_fields, "avg", "IgnoreCase", true);
envelopes_names = all_fields(env_idx);

n_sentences               = length(envelopes_names);
envelopes                 = cell(1, n_sentences);
envelopes_audio_detection = cell(1, n_sentences);

% Match envelopes with detection window and crop
for trl_idx = 1:n_sentences
    fname_envelope                     = envelopes_names{trl_idx};
    base_names                         = strsplit(fname_envelope,'_');
    envelopes{trl_idx}                 = audio_envelopes.(sprintf('envelope_%s',base_names{2}));
    envelopes_audio_detection{trl_idx} = logical(audio_envelopes.(sprintf('audio_detection_%s',base_names{2})));
    clear fname_envelope base_names 
end

fs_audio = audio_envelopes.fs;
clear audio_envelopes

% Import sentences
%-----------------
envelopes_sentence = cell(n_sentences,1);
for trl_idx = 1:n_sentences
    envelopes_sentence{trl_idx} = fileread(fullfile(settings.path2bids,'stimuli','olsa','sentences',[envelopes_names{trl_idx}(10:end),'.txt']));
end
fprintf("Olsa envelopes loaded.\n");

%% Import trained model 
%--------------------------------------------------------------------------
% Trained model
fname = sprintf(settings.fnames.training_decoder,subject,condition,sensor,settings.helper.formatFreqBand(bpfreq),settings.helper.formatIntgrWin(decoder_intgrwin));
data  = importdata(fullfile(settings.path2decoding_training(subject),fname));   
model = data.model;
clear data

%% Apply normalization
%--------------------------------------------------------------------------

if apply_normalization
    % Z-score
    %--------
    % only on neuro data
    epochs_neuro_olsa = apply_zscore_epochs(epochs_neuro_olsa,sensor);
    % epochs_audio_olsa = apply_zscore_epochs(epochs_audio_olsa,'audio');
   
    % Max-abs-scaling
    %----------------
    % only on audio data
    % epochs_neuro_olsa = apply_scaling_epochs(epochs_neuro_olsa,sensor);
    epochs_audio_olsa = apply_scaling_epochs(epochs_audio_olsa,'audio');
    for s_idx = 1:n_sentences
        envelopes{s_idx} = envelopes{s_idx} ./ max(abs(envelopes{s_idx}), [], 2);
    end
    envelope_avg = envelope_avg ./ max(abs(envelope_avg), [], 2);

    fprintf('Normalization to neuro and audio data applied.\n')
end

%% Compute predication scores - Preparations
%--------------------------------------------------------------------------

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

% Simplify intelligibilities
intellis(intellis<2)  = 0;
intellis(intellis>98) = 100;

if n_snrs ~= n_intellis
    error('Number of snrs (%d) and intelligibilities (%d) are not the same!', n_snrs, n_intellis)
end

fs = epochs_neuro_olsa.fsample;

%% Visualize correlation for concatenated sentences including monotonicity score (original data)
%--------------------------------------------------------------------------

% Save original and predicted concatenated data
pred_concat_all  = cell(1,n_snrs);
audio_concat_all = cell(1,n_snrs);
time_vec_all     = cell(1,n_snrs);
r_all            = zeros(1,n_snrs);

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

        % Ensure all cell entries are 1xN row vectors for consistent indexing
        pred_audio   = cellfun(@(x) x(:).', pred_audio, 'UniformOutput', false);
        epochs_audio = cellfun(@(x) x(:).', epochs_audio,    'UniformOutput', false);
        % This ensures the cell array is a 1xN row, not an Nx1 column.
        pred_audio   = pred_audio(:).';
        epochs_audio = epochs_audio(:).';
        % Concatenate data
        pred_concat   = [pred_audio{:}];
        epochs_concat = [epochs_audio{:}];

        [r, ~] = mTRFevaluate(epochs_concat, ...
                              pred_concat,...
                              'dim',settings.decoding.decoder.dim,...
                              'corr',settings.decoding.decoder.corr_metric);

        % Save data
        r_all(snr_idx)            = r;
        pred_concat_all{snr_idx}  = pred_concat;
        audio_concat_all{snr_idx} = epochs_concat;
        time_vec_all{snr_idx}     = (0:length(epochs_concat)-1) / fs;

end

% Visualize concatenated data
%----------------------------
breaktime = 7;
tmax      = 10;

% --- Define Font Sizes ---
fontSizes          = struct();
fontSizes.Title    = 24; % sgtitle
fontSizes.Label    = 24; % xlabel, ylabel
fontSizes.Axis     = 20; % Ticks
fontSizes.Legend   = 20;
fontSizes.Textbox  = 20; % Correlation text

figure('Name', 'Concatenated data', 'NumberTitle', 'off', 'color', 'white');

% --- Define which SNRs to plot ---
% snr_indices_to_plot = 1:2:n_snrs; % e.g., [1, 3, 5, ...]
% snr_indices_to_plot = 2:2:n_snrs; % e.g., [2, 4, 6, ...]
% snr_indices_to_plot = [1,5]; % e.g., [2, 4, 6, ...]
snr_indices_to_plot = [2,6]; 
num_plots           = length(snr_indices_to_plot);
tl                  = tiledlayout(num_plots, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
% tl                  = tiledlayout(num_plots, 1);

for i = 1:num_plots
    % Get the actual snr_idx we want to plot for this iteration
    snr_idx = snr_indices_to_plot(i);

    % Select the next tile for plotting
    ax = nexttile; 
    hold(ax, 'on'); % Use hold(ax, 'on')
    
    % --- Get and Normalize Data ---
    time_orig  = time_vec_all{snr_idx};
    audio_orig = audio_concat_all{snr_idx} / max(abs(audio_concat_all{snr_idx}));
    pred_orig  = pred_concat_all{snr_idx} / max(abs(pred_concat_all{snr_idx}));
    
    % Get original end time
    T_end = time_orig(end);
    
    % --- Define Segments ---
    % 1. First 5 seconds
    idx1   = find(time_orig <= breaktime);
    time1  = time_orig(idx1);
    audio1 = audio_orig(idx1);
    pred1  = pred_orig(idx1);
    
    % 2. Last seconds
    T_start2 = T_end - (tmax-breaktime);
    idx2     = find(time_orig >= T_start2);
    
    time2_orig = time_orig(idx2);
    audio2     = audio_orig(idx2);
    pred2      = pred_orig(idx2);
    
    % --- Remap Time Axis for Segment 2 ---
    % This shifts [T_start2, T_end] to [8, 10]
    time2_remap = (time2_orig - T_start2) + breaktime; 
    
    % --- Plot Segments ---
    % Plot segment 1 and store handles for the legend
    p1 = plot(ax, time1, audio1, 'LineWidth', 2);
    p2 = plot(ax, time1, pred1, '-.', 'LineWidth', 2);
    
    % Plot segment 2, linking colors and turning off legend visibility
    plot(ax, time2_remap, audio2, 'LineWidth', 2, 'Color', p1.Color, 'HandleVisibility', 'off');
    plot(ax, time2_remap, pred2, '-.', 'LineWidth', 2, 'Color', p2.Color, 'HandleVisibility', 'off');
    
    hold(ax, 'off');
    
    % --- Add Legend (First Plot Only) ---
    if i == 1 % Check if this is the first iteration
        lgd = legend([p1, p2], {'original', 'reconstructed'}, 'FontSize', fontSizes.Legend);
        % lgd.Layout.Tile = 'east';
        lgd.Location = 'northeast';
    end

    % --- Set Axis Limits ---
    xlim([0, tmax]) % The new, remapped axis is 10s long
    ylim([-1, 1])

    % --- Add Grid and Box ---
    grid(ax, 'on');
    grid(ax, 'minor');
    box(ax, 'on');
    
    % --- Add Break Visuals ---
    yl = ylim; % Get y-limits
    
    % Define break region
    break_center = breaktime;
    break_width  = 0.1; % Width of the white patch
    x_patch = [break_center, break_center + break_width, break_center + break_width, break_center];
    y_patch = [yl(1), yl(1), yl(2), yl(2)];
    
    % Draw a white, opaque patch to "erase" the plot lines under the break
    patch(ax, x_patch, y_patch, 'w', 'EdgeColor', 'none', 'HandleVisibility', 'off');
    
    % Add two black lines to frame the white "gap"
    line(ax, [x_patch(1), x_patch(1)], yl, 'Color', 'k', 'LineWidth', 0.5, 'HandleVisibility', 'off');
    line(ax, [x_patch(2), x_patch(2)], yl, 'Color', 'k', 'LineWidth', 0.5, 'HandleVisibility', 'off');
    
    % --- Set custom tick labels ---
    
    % Create ticks for the first segment (0 to breaktime)
    xticks_pos_1   = 0:1:breaktime;
    xtick_labels_1 = arrayfun(@num2str, xticks_pos_1, 'UniformOutput', false);
    
    % Create ticks for the second segment (after break)
    % Find the first integer label (e.g., ceil(90.5) = 91)
    first_orig_label = ceil(T_start2); 
    
    % Find the time delta in original time (e.g., 91 - 90.5 = 0.5)
    orig_delta = first_orig_label - T_start2;
    
    % Find the remapped position for that label (e.g., 7 + 0.5 = 7.5)
    first_remap_pos = breaktime + orig_delta;
    
    % Create the remapped positions (e.g., 7.5, 8.5, 9.5...)
    xticks_pos_2 = first_remap_pos:1:(tmax);
    
    % Create the corresponding original labels (e.g., 91, 92, 93...)
    orig_labels    = first_orig_label:1:(first_orig_label + length(xticks_pos_2) - 1);
    xtick_labels_2 = arrayfun(@(x) sprintf('%.0f', x), orig_labels, 'UniformOutput', false);

    % 3. Combine tick positions and labels
    xticks_position = [xticks_pos_1, xticks_pos_2];
    xtick_labels    = [xtick_labels_1, xtick_labels_2];

    % --- Apply labels ONLY to the bottom plot ---
    xticks(ax ,xticks_position);
    if i == num_plots % Check if this is the last iteration
        xticklabels(ax, xtick_labels);
    else
        % Hide labels for upper plots
        xticklabels(ax, {}); 
        xticklabels(ax, xtick_labels);
    end
    
    % --- Add Correlation Text Box ---
    str = sprintf('SNR \\approx %.1f dB (Intelligibility: %.0f %%) / r_{%s} \\approx %.02f', ...
                  snrs(snr_idx), intellis(snr_idx), settings.decoding.decoder.corr_metric, r_all(snr_idx));

    % Place text relative to the [0, 10] x-axis and [-1, 1] y-axis
    text(ax, 0.1, 1.05, str, ...
         'VerticalAlignment', 'top', ...
         'HorizontalAlignment', 'left', ...
         'FontWeight', 'normal', ...
         'BackgroundColor', 'white', ...
         'EdgeColor', 'black', ...
         'Interpreter', 'tex', ...
         'FontSize', fontSizes.Textbox); % Set textbox font size

    % --- Set Axis Font Size ---
    ax.FontSize = fontSizes.Axis; % Sets tick label font size
end

% --- Add Shared Labels for the Entire Layout ---
% This adds the x-label to the bottom plot
xlabel(tl, 't / s', 'FontSize', fontSizes.Label, 'FontWeight', 'bold'); 
% This adds a single, shared y-label to the left
ylabel(tl, 'normalized amplitudes', 'FontSize', fontSizes.Label, 'FontWeight', 'bold');

% sgtitle(tl, sprintf('Subject: %s / Condition: %s / Sensor: %s', subject, condition, sensor), 'FontSize', fontSizes.Title, 'FontWeight', 'bold'); % Add the super-title

%% Compute similarity between Olsa envelopes for ranking
%--------------------------------------------------------------------------

% Compute correlations between all pairs of sentences
[X, Y]    = ndgrid(1:n_sentences, 1:n_sentences);
all_pairs = [X(:), Y(:)];
all_pairs = all_pairs(all_pairs(:,1) ~= all_pairs(:,2), :); 
n_pairs   = size(all_pairs, 1);

% Define the interval for progress updates (every 10%)
update_step = max(1, floor(n_pairs / 10));

sim_vals_sentcs = nan(n_sentences);
% Compute Correlations
for p_idx = 1:n_pairs

    % Get correct envelopes for correlation
    idx1 = all_pairs(p_idx, 1);
    idx2 = all_pairs(p_idx, 2);

    envelope1 = envelopes{idx1}(logical(envelopes_audio_detection{idx1}));
    envelope2 = envelopes{idx2}(logical(envelopes_audio_detection{idx2}));

    min_len   = min(length(envelope1), length(envelope2));
    envelope1 = envelopes{idx1}(1:min_len);
    envelope2 = envelopes{idx2}(1:min_len);
  
    % Perform the correlation computation
    [r, ~] = mTRFevaluate(envelope1, envelope2, 'dim', 2, 'corr', settings.decoding.decoder.corr_metric, 'error', 'mse');
    sim_vals_sentcs(idx1, idx2) = apply_fisher_z_transform(r);

    if mod(p_idx, update_step) == 0 || p_idx == n_pairs
        fprintf('\nProcessed %d / %d correlations.', p_idx, n_pairs);
    end
end
clear idx1 idx2 envelope1 envelope2 r min_len

% Marginalize distribution - mean values for each sentencee
% Inverse transform the mean to r-space
mean_sim_ss_marg = apply_inverse_fisher_z_transform(mean(sim_vals_sentcs, 2, 'omitnan'));

% Sort values
[~, sort_idx] = sort(mean_sim_ss_marg, 'descend');

% Plot data
%----------
figure('Name', sprintf('Similarity metric (%s)', settings.decoding.decoder.corr_metric), 'Color', 'w');
hold on;
plot(mean_sim_ss_marg(sort_idx), 'x', 'Color', [0.2 0.4 0.6], 'LineWidth', 2, 'MarkerSize', 8);
hold off;

ylabel(sprintf('Similarity metric (%s)', settings.decoding.decoder.corr_metric));
xlabel('Sentences');
title('Sorted similarity values');
grid on;
grid minor;
axis square;
box on;

%% Visualize correlation for concatenated sentences including monotonicity score (same single sentence)
%--------------------------------------------------------------------------

% Select sentence condition
% sentence2plot = 'mean-sentence';
sentence2plot = 'single-sentence';

% If single-sentence, select plotting number
% sentce2plot_idx = 7; % select which number to plot based on ranking (1: highest)

switch sentence2plot
    case 'mean-sentence'
        sentence_envelope = envelope_avg;
        envelope_idx      = envelope_avg_audio_detection;
    case 'single-sentence'
        sentence_idx      = find(contains(envelopes_sentence, 'Ulrich gibt fünf alte Tassen')); % Select fixed sentence
        % sentence_idx      = sort_idx(sentce2plot_idx);
        sentence_envelope = envelopes{sentence_idx};
        envelope_idx      = envelopes_audio_detection{sentence_idx};
end

% Save original and predicted concatenated data
pred_concat_all  = cell(1,n_snrs);
audio_concat_all = cell(1,n_snrs);
time_vec_all     = cell(1,n_snrs);
r_all            = zeros(1,n_snrs);
sentence_starts  = cell(1,n_snrs);

color1 = [0.2 0.4 0.6]; % ~ blue
color2 = [0.8 0.1 0.1]; % ~ red

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
        pred_audio      = cellfun(@(x,y) x(:,logical(y)), pred_olsa(idx4snr), epochs_audio_detection_olsa(idx4snr), 'UniformOutput', false); 
        epochs_audio    = repmat({sentence_envelope(envelope_idx)},1 ,n_trials_snr);
        lengths_audio   = cellfun(@length, epochs_audio);
        lengths_pred    = cellfun(@length, pred_audio);
        segment_lengths = min(lengths_audio, lengths_pred); % minimum length of both trials
        segment_lengths = num2cell(segment_lengths);
        % Concatenate sentences and apply minimum duration
        pred_audio      = cellfun(@(x,y) x(:,1:y), pred_audio, segment_lengths, 'UniformOutput', false);
        epochs_audio    = cellfun(@(x,y) x(:,1:y), epochs_audio, segment_lengths, 'UniformOutput', false);
        % This ensures the cell array is a 1xN row, not an Nx1 column.
        pred_audio   = pred_audio(:).';
        epochs_audio = epochs_audio(:).';
        % Concatenate data
        pred_concat   = [pred_audio{:}];
        epochs_concat = [epochs_audio{:}];

        [r, ~] = mTRFevaluate(epochs_concat, ...
                              pred_concat,...
                              'dim',settings.decoding.decoder.dim,...
                              'corr',settings.decoding.decoder.corr_metric);

        % Save data
        r_all(snr_idx)            = r;
        pred_concat_all{snr_idx}  = pred_concat;
        audio_concat_all{snr_idx} = epochs_concat;
        time_vec_all{snr_idx}     = (0:length(epochs_concat)-1) / fs;
        % Remember start index of each sentence
        lens                      = [segment_lengths{:}];
        sentence_starts{snr_idx}  = [1, cumsum(lens(1:end-1)) + 1];
end

% Visualize concatenated data
%----------------------------
breaktime = 7;
tmax      = 10;
% axis limits
ylimits = [-1, 1.2];

% --- Define Font Sizes ---
fontSizes          = struct();
fontSizes.Title    = 25; % sgtitle
fontSizes.Label    = 25; % xlabel, ylabel
fontSizes.Axis     = 25; % Ticks
fontSizes.Legend   = 25;
fontSizes.Textbox  = 25; % Correlation text

fig = figure('Name', 'Concatenated data', 'NumberTitle', 'off', 'color', 'white');

% --- Define which SNRs to plot ---
% snr_indices_to_plot = 1:2:n_snrs; % e.g., [1, 3, 5, ...]
% snr_indices_to_plot = 2:2:n_snrs; % e.g., [2, 4, 6, ...]
% snr_indices_to_plot = [1,5]; % e.g., [2, 4, 6, ...]
snr_indices_to_plot = [1,6]; 
num_plots           = length(snr_indices_to_plot);
tl                  = tiledlayout(num_plots, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
% tl                  = tiledlayout(num_plots, 1);

for i = 1:num_plots
    % Get the actual snr_idx we want to plot for this iteration
    snr_idx = snr_indices_to_plot(i);

    % Select the next tile for plotting
    ax = nexttile; 
    hold(ax, 'on'); % Use hold(ax, 'on')
    ax.TickLabelInterpreter = 'latex';
    
    % --- Get and Normalize Data ---
    time_orig  = time_vec_all{snr_idx};
    audio_orig = audio_concat_all{snr_idx} ./ max(abs(audio_concat_all{snr_idx}));
    pred_orig  = pred_concat_all{snr_idx} ./ max(abs(pred_concat_all{snr_idx}));
    
    % Get original end time
    T_end = time_orig(end);
    
    % --- Define Segments ---
    % 1. First 7 seconds
    idx1   = find(time_orig <= breaktime);
    time1  = time_orig(idx1);
    audio1 = audio_orig(idx1);
    pred1  = pred_orig(idx1);
    
    % 2. Last seconds
    T_start2 = T_end - (tmax-breaktime);
    idx2     = find(time_orig >= T_start2);
    
    time2_orig = time_orig(idx2);
    audio2     = audio_orig(idx2);
    pred2      = pred_orig(idx2);
    
    % --- Remap Time Axis for Segment 2 ---
    % This shifts [T_start2, T_end] to [8, 10]
    time2_remap = (time2_orig - T_start2) + breaktime; 
    
    % --- Plot Segments ---
    % Plot segment 1 and store handles for the legend
    p1 = plot(ax, time1, audio1, 'LineWidth', 2, 'Color', color1);
    p2 = plot(ax, time1, pred1, '-.', 'LineWidth', 2, 'Color', color2);
    
    % Plot segment 2, linking colors and turning off legend visibility
    plot(ax, time2_remap, audio2, 'LineWidth', 2, 'Color', p1.Color, 'HandleVisibility', 'off');
    plot(ax, time2_remap, pred2, '-.', 'LineWidth', 2, 'Color', p2.Color, 'HandleVisibility', 'off');

    % --- Add Sentence Boundary Lines ---
    % Convert sentence start indices to time (seconds)
    sentence_times = (sentence_starts{snr_idx} - 1) / fs;
    
    % Find boundaries in Segment 1 (0 to breaktime)
    idx_seg1 = sentence_times(sentence_times > 0 & sentence_times < breaktime);
    for t_boundary = idx_seg1
        line(ax, [t_boundary, t_boundary], ylimits, 'Color', [0.3 0.3 0.3], 'LineStyle', ':', 'LineWidth', 3, 'HandleVisibility', 'off');
    end
    
    % Find boundaries in Segment 2 (T_start2 to T_end)
    idx_seg2 = sentence_times(sentence_times > T_start2 & sentence_times < T_end);
    for t_boundary = idx_seg2
        % Remap the boundary time to the [breaktime, tmax] range
        t_remapped = (t_boundary - T_start2) + breaktime;
        line(ax, [t_remapped, t_remapped], ylimits, 'Color', [0.3 0.3 0.3], 'LineStyle', ':', 'LineWidth', 3, 'HandleVisibility', 'off');
    end
    
    hold(ax, 'off');
    
    % --- Add Legend (First Plot Only) ---
    if i == 1 % Check if this is the first iteration
        lgd = legend([p1, p2], {'Selected envelopes', 'Reconstructed envelopes'}, 'FontSize', fontSizes.Legend, 'Interpreter', 'latex');
        % lgd.Layout.Tile = 'east';
        lgd.Location = 'northeast';
    end

    % --- Set Axis Limits ---
    xlim([0, tmax]) % The new, remapped axis is 10s long
    ylim(ylimits)

    % --- Add Grid and Box ---
    grid(ax, 'on');
    grid(ax, 'minor');
    box(ax, 'on');
    
    % --- Add Break Visuals ---
    % yl = ylim; % Get y-limits
    
    % Define break region
    break_center = breaktime;
    break_width  = 0.1; % Width of the white patch
    x_patch = [break_center, break_center + break_width, break_center + break_width, break_center];
    y_patch = [ylimits(1), ylimits(1), ylimits(2), ylimits(2)];
    
    % Draw a white, opaque patch to "erase" the plot lines under the break
    patch(ax, x_patch, y_patch, 'w', 'EdgeColor', 'none', 'HandleVisibility', 'off');
    
    % Add two black lines to frame the white "gap"
    line(ax, [x_patch(1), x_patch(1)], ylimits, 'Color', 'k', 'LineWidth', 0.5, 'HandleVisibility', 'off');
    line(ax, [x_patch(2), x_patch(2)], ylimits, 'Color', 'k', 'LineWidth', 0.5, 'HandleVisibility', 'off');
    
    % --- Set custom tick labels ---
    
    % Create ticks for the first segment (0 to breaktime)
    xticks_pos_1   = 0:1:breaktime;
    xtick_labels_1 = arrayfun(@num2str, xticks_pos_1, 'UniformOutput', false);
    
    % Create ticks for the second segment (after break)
    % Find the first integer label (e.g., ceil(90.5) = 91)
    first_orig_label = ceil(T_start2); 
    
    % Find the time delta in original time (e.g., 91 - 90.5 = 0.5)
    orig_delta = first_orig_label - T_start2;
    
    % Find the remapped position for that label (e.g., 7 + 0.5 = 7.5)
    first_remap_pos = breaktime + orig_delta;
    
    % Create the remapped positions (e.g., 7.5, 8.5, 9.5...)
    xticks_pos_2 = first_remap_pos:1:(tmax);
    
    % Create the corresponding original labels (e.g., 91, 92, 93...)
    orig_labels    = first_orig_label:1:(first_orig_label + length(xticks_pos_2) - 1);
    xtick_labels_2 = arrayfun(@(x) sprintf('%.0f', x), orig_labels, 'UniformOutput', false);

    % 3. Combine tick positions and labels
    if i==1
        xticks_position = [xticks_pos_1, xticks_pos_2];
        xtick_labels    = [xtick_labels_1, xtick_labels_2];
    else % Manual tweak to avoid crash of xtick-labels
        xticks_position = [xticks_pos_1(1:end-1), xticks_pos_2]; 
        xtick_labels    = [xtick_labels_1(1:end-1), xtick_labels_2];
    end
    % --- Apply labels ONLY to the bottom plot ---
    xticks(ax ,xticks_position);
    if i == num_plots % Check if this is the last iteration
        xticklabels(ax, xtick_labels);
    else
        % Hide labels for upper plots
        xticklabels(ax, {}); 
        xticklabels(ax, xtick_labels);
    end
    
    % --- Add Correlation Text Box ---
    str = sprintf('SNR = %.1f~dB (Intelligibility: %.0f~\\%%) / $r_{%s} \\approx %.2f$', ...
              snrs(snr_idx), intellis(snr_idx), ...
              settings.decoding.decoder.corr_metric, r_all(snr_idx));

    % Place text relative to the [0, 10] x-axis and [-1, 1] y-axis
    text(ax, 0.1, ylimits(2)*1.05, str, ...
         'VerticalAlignment', 'top', ...
         'HorizontalAlignment', 'left', ...
         'FontWeight', 'normal', ...
         'BackgroundColor', 'white', ...
         'EdgeColor', 'black', ...
         'Interpreter', 'latex', ...
         'FontSize', fontSizes.Textbox); % Set textbox font size

    % --- Set Axis Font Size ---
    ax.FontSize = fontSizes.Axis; % Sets tick label font size
end

% --- Add Shared Labels for the Entire Layout ---
% This adds the x-label to the bottom plot
xlabel(tl, 'Time / s', 'FontSize', fontSizes.Label, 'FontWeight', 'bold', 'Interpreter', 'latex'); 
% This adds a single, shared y-label to the left
ylabel(tl, 'Normalized amplitudes', 'FontSize', fontSizes.Label, 'FontWeight', 'bold', 'Interpreter', 'latex');

% sgtitle(tl, sprintf('Subject: %s / Condition: %s / Sensor: %s', subject, condition, sensor), 'FontSize', fontSizes.Title, 'FontWeight', 'bold'); % Add the super-title

% Plot correlation values over SNR/Intelligibility
%-------------------------------------------------
figure('Name', sprintf('Correlation coefficients across SNR (%s)', envelopes_sentence{sentence_idx}), 'Color', 'w');

% Update plot with MarkerSize and LineWidth
plot(intellis, r_all, 'x', 'Color', [0.2 0.4 0.6], ...
     'MarkerSize', 12, 'LineWidth', 2.5);

xlabel('Intelligibility / %', 'FontSize', fontSizes.Label, 'FontWeight', 'bold');
ylabel(sprintf("r_{%s}", settings.decoding.decoder.corr_metric), 'FontSize', fontSizes.Label, 'FontWeight', 'bold');
% Ticks and Labels
xticks(intellis);
xticklabels(string(intellis));
% Assign and format axes
ax                   = gca; 
ax.FontSize          = fontSizes.Axis; 
ax.XTickLabelRotation = 0;
box(ax, 'on');
grid(ax, 'on');
grid(ax, 'minor');

% Maintain the square aspect ratio
axis square;

%% Show single concatenated curves with axis
%--------------------------------------------------------------------------

snr_idx    = 6;
time_orig  = time_vec_all{snr_idx};
audio_orig = audio_concat_all{snr_idx} / max(abs(audio_concat_all{snr_idx}));
pred_orig  = pred_concat_all{snr_idx} / max(abs(pred_concat_all{snr_idx}));

figure('Color', 'white');
% plot(time_orig, audio_orig, 'LineWidth', 2, 'Color', [0 0.4470 0.7410]);
plot(time_orig, pred_orig, 'LineWidth', 2, 'Color', [0 0.4470 0.7410]);
xlim([0,5])
% Turn off the axes of the main plot
axis off;

% --- Draw Arrows and Text using 'annotation' ---

% X-axis arrow (thicker)
annotation('arrow', [0.1, 0.4], [0.1, 0.1], 'LineWidth', 2); 
% Y-axis arrow (thicker)
annotation('arrow', [0.1, 0.1], [0.1, 0.4], 'LineWidth', 2);

% X-axis label (horizontal)
annotation('textbox', [0.2, 0.0, 0.1, 0.1], ...
             'String', 'time / s', ...
             'EdgeColor', 'none', ...
             'FontSize', 14, ...
             'HorizontalAlignment', 'center'); % Align with arrow

% Y-axis label (vertical)
annotation('textbox', [0.1, 0.1, 0.2, 0.2], ... % Adjusted position
             'String', 'amplitude', ...
             'EdgeColor', 'none', ...
             'FontSize', 14, ...
             'Rotation', 90, ... % Add rotation
             'HorizontalAlignment', 'center', ... % Center along its new axis
             'VerticalAlignment', 'bottom');

%% Plot mixture (speech + noise)
%--------------------------------------------------------------------------
% Additionally import an example stimulus from that subject & session
% condition must be ses-01 or ses-02

% Select run and trial
run            = 'run-02';
trial          = 'trial-01';
condition2plot = 'ses-01';

fname                    = sprintf('%s_%s_task-olsa_%s_%s_stim.wav',subject,condition2plot,run,trial);  
[audio_mixture, fs_orig] = audioread(fullfile(settings.path2bids,'stimuli','olsa',subject,condition2plot,fname));
audio_mixture            = audio_mixture(:,1); % take only first channel
audio_mixture            = audio_mixture/max(abs(audio_mixture));
time_vec                 = (0:length(audio_mixture)-1) / fs_orig;

figure('Color', 'white');
plot(time_vec, audio_mixture, 'LineWidth', 2, 'Color', [0 0.4470 0.7410]);
xlim([0,1])
% Turn off the axes of the main plot
axis off;

% --- Draw Arrows and Text using 'annotation' ---

% X-axis arrow (thicker)
annotation('arrow', [0.1, 0.4], [0.1, 0.1], 'LineWidth', 2); 
% Y-axis arrow (thicker)
annotation('arrow', [0.1, 0.1], [0.1, 0.4], 'LineWidth', 2);

% X-axis label (horizontal)
annotation('textbox', [0.2, 0.0, 0.1, 0.1], ...
             'String', 'time / s', ...
             'EdgeColor', 'none', ...
             'FontSize', 14, ...
             'HorizontalAlignment', 'center'); % Align with arrow

% Y-axis label (vertical)
annotation('textbox', [0.1, 0.1, 0.2, 0.2], ... % Adjusted position
             'String', 'amplitude', ...
             'EdgeColor', 'none', ...
             'FontSize', 14, ...
             'Rotation', 90, ... % Add rotation
             'HorizontalAlignment', 'center', ... % Center along its new axis
             'VerticalAlignment', 'bottom');

