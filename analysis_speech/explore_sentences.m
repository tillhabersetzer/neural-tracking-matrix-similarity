%--------------------------------------------------------------------------
% Till Habersetzer, 17.04.2026
% Communication Acoustics, CvO University Oldenburg
% till.habersetzer@uol.de 
%
% Description:
%   This script provides a comprehensive analysis of speech stimulus 
%   characteristics, focusing on amplitude envelope similarity and 
%   temporal modulation profiles. 
%
%   Part 1: Similarity Metrics
%   Calculates similarity (Spearman, Pearson, or RMSE) between sentence 
%   envelopes and a grand-average template. Includes Fisher Z-transforms,
%   normality testing (Shapiro-Wilk), and cross-correlation via circular 
%   shifting or zero-padding.
%
%   Part 2: Modulation Spectroscopy
%   Extracts the rhythmic profile of speech stimuli using the AMToolbox 
%   (AMT) and Gabor-based modulation spectrograms. Includes a custom 
%   Voice Activity Detection (VAD) to isolate active speech and 
%   automated chunking for long-form audio (audiobooks).
%
% Requirements:
%   - mTRF Toolbox: For 'mtrf' correlation logic.
%   - AMToolbox (AMT): For auditory modulation processing.
%   - Project Helper Functions: apply_fisher_z_transform, swtest, etc.
%   - plot_modspecgram_TH: Modified AMT function for data extraction.
%
% Key Parameters:
%   bpfreq              - Bandpass filter range for envelope extraction.
%   corr_metric         - Choice of 'Spearman', 'Pearson', or 'Rmse'.
%   mfmax               - Max modulation frequency for spectrum analysis.
%   max_duration_sec    - Chunking threshold for long audio files (VAD).
%
% Visualizations:
%   1. Sorted similarity values and Gaussian distribution fits.
%   2. Histograms of sentence-to-sentence vs. sentence-to-mean similarity.
%   3. Time-domain comparisons of high-similarity envelope pairs.
%   4. Mean similarity as a function of circular and linear time-shifts.
%   5. Grand average 1D modulation spectrum with syllabic rate indicators.
%--------------------------------------------------------------------------

close all
clearvars
clc 

%% Import main settings 
%--------------------------------------------------------------------------
settings_decoding

% Bandpass filter frequency
%--------------------------
% bpfreq = [0.5,4];
bpfreq = [0.5,8];
% bpfreq = [0.5,30];
% bpfreq = [4,8];

% Apply normalization of audio before correlation
%------------------------------------------------
apply_normalization = true;

% Choose Correlation metric
%--------------------------
corr_metric = 'Spearman';
% corr_metric = 'Pearson';
% corr_metric = 'Rmse';

% Choose logic to compute similarity
%-----------------------------------
% Both do the same (just a sanity check!)
% corr_logic = 'matlab';
corr_logic = 'mtrf';

% Add paths
%--------------------------------------------------------------------------
addpath(genpath(settings.path2mtrftoolbox));
addpath(fullfile(settings.path2project,'analysis','helper_functions'))

%% Import envelopes
%--------------------------------------------------------------------------

% Extact mean envelope
%--------------------------------------------------------------------------
fname                  = sprintf(settings.fnames.audio_olsa, settings.helper.formatFreqBand(bpfreq));     
audio_envelopes        = importdata(fullfile(settings.path2derivatives,'stimuli',fname));
envelope_avg           = audio_envelopes.envelope_avg;
envelope_avg_detection = audio_envelopes.audio_detection_avg;
envelope_avg_cropped   = envelope_avg(envelope_avg_detection);
length_avg             = length(envelope_avg_cropped);
fs                     = audio_envelopes.fs;

% Extract envelopes
%--------------------------------------------------------------------------
all_fields        = fieldnames(audio_envelopes);
env_idx           = contains(all_fields, "envelope", "IgnoreCase", true) & ~contains(all_fields, "avg", "IgnoreCase", true);
envelopes_names   = all_fields(env_idx);

n_sentences       = length(envelopes_names);
envelopes         = cell(1,n_sentences);
envelopes_cropped = cell(1,n_sentences);

% Match envelopes with detection window and crop
for trl_idx = 1:n_sentences
    fname_envelope             = envelopes_names{trl_idx};
    base_names                 = strsplit(fname_envelope,'_');
    envelopes{trl_idx}         = audio_envelopes.(sprintf('envelope_%s',base_names{2}));
    audio_detection            = audio_envelopes.(sprintf('audio_detection_%s',base_names{2}));
    envelopes_cropped{trl_idx} = envelopes{trl_idx}(logical(audio_detection));
    clear fname_envelope base_names audio_detection
end

lengths_envelopes = cellfun(@length, envelopes_cropped);
fs_audio          = audio_envelopes.fs;
clear audio_envelopes

% Import sentences
%-----------------
envelopes_sentence = cell(n_sentences,1);
for trl_idx = 1:n_sentences
    envelopes_sentence{trl_idx} = fileread(fullfile(settings.path2bids,'stimuli','olsa','sentences',[envelopes_names{trl_idx}(10:end),'.txt']));
end

%% Optional: Apply normalization
%--------------------------------------------------------------------------

if apply_normalization
    % Z-score
    %--------
    % envelopes_cropped    = apply_zscore_epochs(envelopes_cropped,'audio');
    % envelope_avg_cropped = (envelope_avg_cropped-mean(envelope_avg_cropped,'all'))/std(envelope_avg_cropped,0,'all');

    % Max-abs-scaling- > does not lead to change in correlation
    %----------------
    % envelopes_cropped    = apply_scaling_epochs(envelopes_cropped,'audio');
    for s_idx = 1:n_sentences
        envelopes_cropped{s_idx} = envelopes_cropped{s_idx} ./ max(abs(envelopes_cropped{s_idx}), [], 2);
    end
    envelope_avg_cropped = envelope_avg_cropped ./ max(abs(envelope_avg_cropped), [], 2);

    fprintf('Normalization to neuro and audio data applied.\n')
end

%% Compute correlation values
%--------------------------------------------------------------------------

% Compute correlations between all pairs of sentences
%--------------------------------------------------------------------------
[X, Y]    = ndgrid(1:n_sentences, 1:n_sentences);
all_pairs = [X(:), Y(:)];
all_pairs = all_pairs(all_pairs(:,1) ~= all_pairs(:,2), :); 
n_pairs   = size(all_pairs, 1);

% Calculate lengths
min_lengths = min(lengths_envelopes(all_pairs(:,1)),lengths_envelopes(all_pairs(:,2)));

% Define the interval for progress updates (every 10%)
update_step = max(1, floor(n_pairs / 10));

sim_vals_sentcs = nan(n_sentences);
% Compute Correlations
for p_idx = 1:n_pairs

    % Get correct envelopes for correlation
    idx1 = all_pairs(p_idx, 1);
    idx2 = all_pairs(p_idx, 2);

    envelope1 = envelopes_cropped{idx1}(1:min_lengths(p_idx));
    envelope2 = envelopes_cropped{idx2}(1:min_lengths(p_idx));
  
    % Perform the correlation computation
    %----------------------------------------------------------------------
    switch corr_logic
        case 'mtrf'
            if strcmp(corr_metric,'Rmse')
                [r, err] = mTRFevaluate(envelope1, envelope2, 'dim', 2, 'corr', 'Spearman', 'error', 'mse');
            else
                [r, err] = mTRFevaluate(envelope1, envelope2, 'dim', 2, 'corr', corr_metric, 'error', 'mse');
            end
            rmse = sqrt(err);
            clear err
        case 'matlab'
            r    = corr(envelope1', envelope2', 'Type', corr_metric);
            rmse = sqrt(mean((envelope1-envelope2).^2));
    end

    switch corr_metric
        case {'Pearson','Spearman'}
            sim_vals_sentcs(idx1, idx2) = apply_fisher_z_transform(r);
        case 'Rmse'
            sim_vals_sentcs(idx1, idx2) = rmse;
    end

    if mod(p_idx, update_step) == 0 || p_idx == n_pairs
        fprintf('\nProcessed %d / %d correlations.', p_idx, n_pairs);
    end
    clear idx1 idx2 envelope1 envelope2 r rmse
    
end

mask_sim_vals = tril(true(size(sim_vals_sentcs)), -1);
switch corr_metric
    case {'Pearson','Spearman'}
        mean_sim_ss_all = apply_inverse_fisher_z_transform(mean(sim_vals_sentcs(mask_sim_vals), 'all', 'omitnan'));
    case 'Rmse'
        mean_sim_ss_all = mean(sim_vals_sentcs(mask_sim_vals), 'all', 'omitnan');
end

% Compute correlations sentences and average envelope
%--------------------------------------------------------------------------

% Calculate lengths
min_lengths = min(length_avg,lengths_envelopes(:));

sim_vals_mean_sentcs = zeros(n_sentences,1);
% Compute Correlations
for p_idx = 1:n_sentences

    envelope1 = envelopes_cropped{p_idx}(1:min_lengths(p_idx));
    envelope2 = envelope_avg_cropped(1:min_lengths(p_idx));
  
    % Perform the correlation computation
    %----------------------------------------------------------------------
    switch corr_logic
        case 'mtrf'
            if strcmp(corr_metric,'Rmse')
                [r, err] = mTRFevaluate(envelope1, envelope2, 'dim', 2, 'corr', 'Spearman', 'error', 'mse');
            else
                [r, err] = mTRFevaluate(envelope1, envelope2, 'dim', 2, 'corr', corr_metric, 'error', 'mse');
            end
            rmse = sqrt(err);
            clear err
        case 'matlab'
            r    = corr(envelope1', envelope2', 'Type', corr_metric);
            rmse = sqrt(mean((envelope1-envelope2).^2));
    end

    switch corr_metric
        case {'Pearson','Spearman'}
            sim_vals_mean_sentcs(p_idx) = apply_fisher_z_transform(r);
        case 'Rmse'
            sim_vals_mean_sentcs(p_idx) = rmse;
    end

    clear envelope1 envelope2  r rmse
end

switch corr_metric
    case {'Pearson','Spearman'}
        mean_sim_ms = apply_inverse_fisher_z_transform(mean(sim_vals_mean_sentcs));
    case 'Rmse'
        mean_sim_ms = mean(sim_vals_mean_sentcs);
end

fprintf('\nFinished computations of correlations with mean envelope.\n');

%% Check sentence correlations
%--------------------------------------------------------------------------

% Marginalize distribution - mean values for each sentencee
switch corr_metric
    case {'Pearson','Spearman'}
        % Calculate Mean and STD in Z-space first
        mean_z = mean(sim_vals_sentcs, 2, 'omitnan');
        std_z  = std(sim_vals_sentcs, [], 2, 'omitnan');
        
        % Inverse transform the mean to r-space
        mean_sim_ss_marg = apply_inverse_fisher_z_transform(mean_z);
        
        % Calculate absolute upper and lower bounds in Z-space, then transform to r-space
        r_upper = apply_inverse_fisher_z_transform(mean_z + std_z);
        r_lower = apply_inverse_fisher_z_transform(mean_z - std_z);
        
        % Calculate the relative negative and positive error lengths for the errorbar() function
        std_neg = mean_sim_ss_marg - r_lower;
        std_pos = r_upper - mean_sim_ss_marg;

    case 'Rmse'
        mean_sim_ss_marg = mean(sim_vals_sentcs, 2, 'omitnan');
        std_sim_ss_marg  = std(sim_vals_sentcs, [], 2, 'omitnan');

        std_neg = std_sim_ss_marg;
        std_pos = std_sim_ss_marg;
end

% Sort values
[~, sort_idx] = sort(mean_sim_ss_marg, 'descend');

% Plot data
%----------
figure('Name', sprintf('Similarity metric (%s)', corr_metric), 'Color', 'w');
subplot(1,2,1)
hold on;
errorbar(1:n_sentences, mean_sim_ss_marg(sort_idx), std_neg(sort_idx), std_pos(sort_idx), '.', ...
        'CapSize', 2, 'Color', [0.2 0.4 0.6 0.5], 'LineWidth', 0.5);
%errorbar(mean_sim_ss_marg(sort_idx), std_sim_ss_marg(sort_idx), '.', 'CapSize', 2, 'Color', [0.2 0.4 0.6 0.5], 'LineWidth', 0.5); 
% Plot the 'x' markers with a thick LineWidth
plot(mean_sim_ss_marg(sort_idx), 'x', 'Color', [0.2 0.4 0.6], 'LineWidth', 2, 'MarkerSize', 8);
hold off;

ylabel(sprintf('Similarity metric (%s)', corr_metric));
xlabel('Sentences');
title('Sorted similarity values');
grid on;
grid minor;
axis square;
box on;

% Visualization: Gaussian Approximations 
%--------------------------------------------------------------------------
% Completely in z-space for correlation values
is_gaussian   = false(n_sentences, 1); 
[mini, maxi]  = bounds(sim_vals_sentcs, 'all', 'omitnan');
buffer        = 0.1*abs(maxi-mini);
sim_vals4plot = linspace(mini-buffer, maxi+buffer, 500);

subplot(1,2,2)
hold on;

for sent_idx = 1:n_sentences
  
    sim_vals_distr = sim_vals_sentcs(sent_idx,:);
    
    % Check for normality
    [h, ~, ~]             = swtest(sim_vals_distr(~isnan(sim_vals_distr)));
    is_gaussian(sent_idx) = (h == 0); 
    clear h              
   
    % Generate Gaussian PDF
    % Formula: (1/sig*sqrt(2*pi)) * exp(-0.5 * ((x-mu)/sig)^2)
    m = mean(sim_vals_distr, 'omitnan');
    s = std(sim_vals_distr, 'omitnan');
    y = (1 / (s * sqrt(2*pi))) * exp(-0.5 * ((sim_vals4plot - m) / s).^2);
    
    % Select color
    if is_gaussian(sent_idx)
        lineColor = [0.5, 0.5, 0.5, 0.15]; % Transparent Grey (Passed)
    else
        lineColor = [0.9, 0.2, 0.2, 0.20]; % Transparent Red (Failed)
    end
    
    % Plot
    plot(sim_vals4plot, y, 'Color', lineColor, 'LineWidth', 1.2);
end

% Improved axis label
switch corr_metric
    case {'Pearson','Spearman'}
        corr_metric2 = sprintf('%s (Z-space)', corr_metric);
    case 'Rmse'
        corr_metric2 = corr_metric;
end

% Finalize Plot and Legend
hold off;
xlabel(sprintf('Similarity metric (%s)', corr_metric2));
ylabel('Probability Density');
title(sprintf('Gaussian fits per sentence distribution (Valid distribtuions: %d/%d)', sum(is_gaussian), n_sentences));
grid on;
grid minor;
axis square;
box on;

sgtitle(sprintf('Similarity metric: %s', corr_metric))

%% Visualization of correlation values
%--------------------------------------------------------------------------

% Plotting settings
%------------------
% Choose how to select envelopes for the time course plots.
% Options: 'highest_correlation' or 'random'
% selection_method = 'random'; 
selection_method = 'highest_correlation'; 

figure('Color', 'white', 'name', 'Analysis of Olsa Sentence Envelope Correlations','WindowState', 'maximized');
t = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
sgtitle(sprintf('Analysis of Olsa Sentence Envelope Correlations (%s) using %s', corr_metric, corr_logic), 'FontSize', 16, 'FontWeight', 'bold');

% Row 1: Histograms 
%--------------------------------------------------------------------------

% Plot 1: Histogram for sentence-to-sentence correlations
%--------------------------------------------------------
ax1 = nexttile;
hold on;
switch corr_metric
    case {'Pearson','Spearman'}
        histogram(apply_inverse_fisher_z_transform(sim_vals_sentcs(mask_sim_vals)));
    case 'Rmse'
        histogram(sim_vals_sentcs(mask_sim_vals));   
end
h_line_ss = xline(mean_sim_ss_all, 'r--', 'LineWidth', 2, 'DisplayName', sprintf('Mean: %.2f', mean_sim_ss_all));
hold off;
title(sprintf('Sentence-to-Sentence Similarity (%s)', corr_metric))
xlabel(sprintf('Similarity metric (%s)', corr_metric));
ylabel('Count');
grid on;
grid minor;
box on;
legend(h_line_ss, 'Location', 'northwest');

% PLOT 2: Histogram for sentence-to-mean-sentence correlations
%-------------------------------------------------------------
ax2 = nexttile;
hold on;
switch corr_metric
    case {'Pearson','Spearman'}
        histogram(apply_inverse_fisher_z_transform(sim_vals_mean_sentcs));
    case 'Rmse'
        histogram(sim_vals_mean_sentcs); 
end
h_line_ms = xline(mean_sim_ms, 'r--', 'LineWidth', 2, 'DisplayName', sprintf('Mean: %.2f', mean_sim_ms));
hold off;
title(sprintf('Sentence-to-Mean Similarity (%s)', corr_metric))
xlabel(sprintf('Similarity metric (%s)', corr_metric));
ylabel('Count');
grid on;
grid minor;
box on;
legend(h_line_ms, 'Location', 'northwest');

% Prepare data for timecourse plot
%--------------------------------------------------------------------------

switch selection_method
    case 'highest_correlation'

        % Remove other half to get unique maxima
        sim_vals_sentcs_searchmax                = nan(size(sim_vals_sentcs));
        sim_vals_sentcs_searchmax(mask_sim_vals) = sim_vals_sentcs(mask_sim_vals);
        
        switch corr_metric
            % Search for maximum correlation
            case {'Pearson','Spearman'}
                % Find the "highest" sentence-sentence similarity and corresponding envelopes
                [max_sim_ss, max_idx_ss] = max(sim_vals_sentcs_searchmax, [], 'all');
                % Find the "highest" sentence-mean similarity and corresponding envelope
                [max_sim_sm, max_idx_sm] = max(sim_vals_mean_sentcs);
            % Search for minimum difference
            case 'Rmse'
                [max_sim_ss, max_idx_ss] = min(sim_vals_sentcs_searchmax, [], 'all');
                [max_sim_sm, max_idx_sm] = min(sim_vals_mean_sentcs);
        end

        % Convert the linear index to 2D subscripts
        [idx1, idx2] = ind2sub(size(sim_vals_sentcs), max_idx_ss);
        clear sim_vals_sentcs_searchmax

    case 'random'
        % Select a random pair of sentences
        random_idx_ss            = randi(size(all_pairs, 1));
 
        best_pair_indices        = all_pairs(random_idx_ss, :);
        idx1                     = best_pair_indices(1);
        idx2                     = best_pair_indices(2);
        max_sim_ss               = sim_vals_sentcs(idx1, idx2); % Get the correlation for this random pair
        
        % Select a random sentence to compare with the mean
        max_idx_sm               = randi(n_sentences);
        max_sim_sm               = sim_vals_mean_sentcs(max_idx_sm); % Get the correlation for this random sentence

    otherwise
        error("Invalid selection_method. Choose 'highest_correlation' or 'random'.");
end

% Transform data
if ismember(corr_metric, {'Pearson','Spearman'})
    max_sim_ss = apply_inverse_fisher_z_transform(max_sim_ss);
    max_sim_sm = apply_inverse_fisher_z_transform(max_sim_sm);
end

% Extract the chosen envelopes based on the indices from the switch block
env1_ss_full  = envelopes_cropped{idx1};
env2_ss_full  = envelopes_cropped{idx2};
env_sm_full   = envelopes_cropped{max_idx_sm};
mean_env_full = envelope_avg_cropped;

% Determine common ylimits for both plots
%----------------------------------------
% Combine all data that will be plotted in the second row
all_plot_data   = [env1_ss_full(:); env2_ss_full(:); env_sm_full(:); mean_env_full(:)];
% Find the min and max across all of it to set a common scale
y_min_common    = min(all_plot_data);
y_max_common    = max(all_plot_data);
common_y_limits = [y_min_common, y_max_common];

% Row 2: Full envelopes with shaded similarity window
%--------------------------------------------------------------------------

% Plot 3: Cropped time courses for highest sentence-sentence correlation
%-----------------------------------------------------------------------
ax3 = nexttile;
hold on;
% Define the end time of the correlation window
sim_len_ss      = min(length(env1_ss_full), length(env2_ss_full));
sim_end_time_ss = (sim_len_ss - 1) / fs_audio;
% Plot the full envelopes first to set the axes limits automatically
time_vec1_full = (0:length(env1_ss_full)-1) / fs_audio;
time_vec2_full = (0:length(env2_ss_full)-1) / fs_audio;
h1_ss = plot(time_vec1_full, env1_ss_full, 'LineWidth', 1.2);
h2_ss = plot(time_vec2_full, env2_ss_full, 'LineWidth', 1.2);
% Common ylimits
patch_ss = patch([0 sim_end_time_ss sim_end_time_ss 0], [common_y_limits(1) common_y_limits(1) common_y_limits(2) common_y_limits(2)], [0.9 0.9 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.5); 
% Send the patch to the background
uistack(patch_ss, 'bottom');
hold off;
title(sprintf('Sentence-Sentence Similarity (%s: %.2f)', corr_metric, max_sim_ss));
xlabel('Time / s'); 
ylabel('Amplitude');
legend([h1_ss, h2_ss, patch_ss], ...
       {sprintf('%s / %s', envelopes_names{idx1},envelopes_sentence{idx1}), sprintf('%s / %s', envelopes_names{idx2},envelopes_sentence{idx2}), 'Correlation Window'}, ...
       'Interpreter', 'none', 'Location', 'southwest');
grid on; 
grid minor;
box on;

% Plot 4: Full time courses for highest sentence-mean pair 
%---------------------------------------------------------
ax4 = nexttile;
hold on;
sim_len_sm         = min(length(env_sm_full), length(mean_env_full));
sim_end_time_sm    = (sim_len_sm - 1) / fs_audio;
time_vec_sm_full   = (0:length(env_sm_full)-1) / fs_audio;
time_vec_mean_full = (0:length(mean_env_full)-1) / fs_audio;
% Capture plot handles
h1_sm = plot(time_vec_sm_full, env_sm_full, 'LineWidth', 1.2);
h2_sm = plot(time_vec_mean_full, mean_env_full, 'LineWidth', 1.2);
% Common ylimits
% Create transparent patch
patch_sm = patch([0 sim_end_time_sm sim_end_time_sm 0], [common_y_limits(1) common_y_limits(1) common_y_limits(2) common_y_limits(2)], [0.9 0.9 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
uistack(patch_sm, 'bottom');
hold off;
title(sprintf('Sentence-MeanSentence Similarity (%s: %.2f)', corr_metric, max_sim_sm));
xlabel('Time / s'); 
ylabel('Amplitude');
% Update legend to include patch
legend([h1_sm, h2_sm, patch_sm], {sprintf('%s / %s', envelopes_names{max_idx_sm}, envelopes_sentence{max_idx_sm}), 'Mean Envelope', 'Correlation Window'}, 'Interpreter', 'none', 'Location', 'southwest');
grid on; 
grid minor; 
box on;

% Apply axis scaling requirements
linkaxes([ax3, ax4], 'xy');

%% Compute mean correlation with time-shift (circular)
%--------------------------------------------------------------------------

[X, Y]          = ndgrid(1:n_sentences, 1:n_sentences);
all_pairs_shift = [X(:), Y(:)];
% Exclude only the self-correlation 
all_pairs_shift = all_pairs_shift(all_pairs_shift(:,1) ~= all_pairs_shift(:,2), :);
% all_pairs_shift = all_pairs_shift(all_pairs_shift(:,1) < all_pairs_shift(:,2), :); 
n_pairs_shift   = size(all_pairs_shift, 1);  % include comparison with same sentence and reverse pairs

% Calculate lengths
min_lengths_shift = min(lengths_envelopes(all_pairs_shift(:,1)),lengths_envelopes(all_pairs_shift(:,2)));
shifts_samples    = 0:min(min_lengths_shift);
n_shifts          = length(shifts_samples);

% Define the interval for progress updates (every 10%)
n_loops     = n_pairs_shift*n_shifts;
update_step = max(1, floor(n_loops / 10));

% Initialize
sim_vals_sentcs_shift = nan(n_sentences, n_shifts); 

% Compute Correlations
%---------------------
counter = 0;

% Loop over shifts
for s_idx = 1:n_shifts
    shift = shifts_samples(s_idx);

    % Loop over valid pairs
    for p_idx = 1:n_pairs_shift

        % Get correct envelopes for correlation
        idx1 = all_pairs_shift(p_idx, 1);
        idx2 = all_pairs_shift(p_idx, 2);
    
        envelope1 = envelopes_cropped{idx1};
        envelope2 = envelopes_cropped{idx2};

        % Crop
        envelope1 = envelope1(1:min_lengths_shift(p_idx));
        envelope2 = envelope2(1:min_lengths_shift(p_idx));

        % Apply shift (in samples)
        envelope2 = circshift(envelope2, shift);
  
        % Perform the correlation computation
        %----------------------------------------------------------------------
        switch corr_logic
            case 'mtrf'
                if strcmp(corr_metric,'Rmse')
                    [r, err] = mTRFevaluate(envelope1, envelope2, 'dim', 2, 'corr', 'Spearman', 'error', 'mse');
                else
                    [r, err] = mTRFevaluate(envelope1, envelope2, 'dim', 2, 'corr', corr_metric, 'error', 'mse');
                end
                rmse = sqrt(err);
                clear err
            case 'matlab'
                r    = corr(envelope1', envelope2', 'Type', corr_metric);
                rmse = sqrt(mean((envelope1-envelope2).^2));
        end

        switch corr_metric
            case {'Pearson','Spearman'}
                sim_vals_sentcs_shift(p_idx, s_idx) = apply_fisher_z_transform(r);
            case 'Rmse'
                sim_vals_sentcs_shift(p_idx, s_idx) = rmse;
        end

        counter = counter + 1;

        if mod(counter, update_step) == 0 || counter == n_loops
            fprintf('\nProcessed %d / %d correlations.', counter, n_loops);
        end

    end % pairs
    
end % shifts
clear idx1 idx2 envelope1 envelope2 r rmse

switch corr_metric
    case {'Pearson','Spearman'}
        mean_sim_ss_shift = apply_inverse_fisher_z_transform(mean(sim_vals_sentcs_shift, 1, 'omitnan'));
    case 'Rmse'
        mean_sim_ss_shift = mean(sim_vals_sentcs_shift, 1, 'omitnan');
end

% Visualize mean correlation over shift
%--------------------------------------
shifts_time = shifts_samples/fs*1000; % ms

% Plot data
%----------
figure('Name', sprintf('Similarity metric (%s) over timeshift', corr_metric), 'Color', 'w');

hold on;
% Plot the 'x' markers with a thick LineWidth
plot(shifts_time, mean_sim_ss_shift, 'x', 'Color', [0.2 0.4 0.6], 'LineWidth', 2, 'MarkerSize', 8);
hold off;

ylabel(sprintf('Mean similarity metric (%s)', corr_metric));
xlabel('Circular shift / ms');
title(sprintf('Mean similarity over timeshift (%s)', corr_metric));
grid on;
grid minor;
% axis square;
box on;

%% Compute mean correlation with time-shift (linear cross-correlation)
%--------------------------------------------------------------------------

[X, Y]          = ndgrid(1:n_sentences, 1:n_sentences);
all_pairs_shift = [X(:), Y(:)];
all_pairs_shift = all_pairs_shift(all_pairs_shift(:,1) ~= all_pairs_shift(:,2), :); 
n_pairs_shift   = size(all_pairs_shift, 1);

max_len              = max(lengths_envelopes);
envelopes_cross_corr = cell(n_pairs_shift, 1);

% Loop over valid pairs
for p_idx = 1:n_pairs_shift

    % Get correct envelopes for correlation
    idx1 = all_pairs_shift(p_idx, 1);
    idx2 = all_pairs_shift(p_idx, 2);

    envelope1 = [envelopes_cropped{idx1}, zeros(1, max_len - lengths_envelopes(idx1))];
    envelope2 = [envelopes_cropped{idx2}, zeros(1, max_len - lengths_envelopes(idx2))];

    [r_crosscorr, lags]         = xcorr(envelope1 - mean(envelope1), envelope2 - mean(envelope2), 'normalized');   
    envelopes_cross_corr{p_idx} = r_crosscorr;
    
end
clear envelope1 envelope2 r_crosscorr

% Convert cell to matrix and compute mean across all sentence pairs
% We must ensure we align them by lags if lengths vary
envelopes_cross_corr      = cell2mat(envelopes_cross_corr); 
envelopes_cross_corr_mean = mean(envelopes_cross_corr, 1);
lags_time                 = lags / fs * 1000; % Convert to ms

% Visualization
figure('Name', 'Cross-correlation function', 'Color', 'w');
plot(lags_time, envelopes_cross_corr_mean, 'x', 'Color', [0.2 0.4 0.6], 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Lag / ms');
ylabel('Correlation Coefficient (r)');
title(sprintf('Cross-Correlation Similarity (Zero-Padded)'));
grid on;
xlim([0 max(lags_time)]); % Ensure plot starts at zero

%% Explore modulation spectrum of speech
%--------------------------------------------------------------------------

% Initialize Auditory and mTRF Toolboxes
addpath(fullfile(settings.path2amtoolbox));
amt_start;
fprintf('  - AMToolbox initialized.\n\n');

%% Select files
%--------------------------------------------------------------------------

% Select olsa files
%------------------
stim_dir     = fullfile(settings.path2bids,'stimuli','olsa','sentences'); 
contents     = dir(stim_dir);
fnames_audio = {contents.name};
fnames_audio = fnames_audio(endsWith(fnames_audio, ".wav", "IgnoreCase", true));
n_sentences  = length(fnames_audio);

% Select story files
%------------------
% session      = 'ses-01';
% stim_dir     = fullfile(settings.path2bids,'stimuli','audiobook', session); 
% contents     = dir(stim_dir);
% fnames_audio = {contents.name};
% fnames_audio = fnames_audio(endsWith(fnames_audio, ".wav", "IgnoreCase", true));
% n_sentences  = length(fnames_audio);

% Analysis Parameters 
mfmax            = 50; % Maximum modulation frequency to analyze
max_duration_sec = 30; % Threshold to start chunking (seconds)
all_psds         = [];

% Compute modulation spectrogram
%-------------------------------
% n_sentences = 1;
all_psds = []; 

% Save sentence duration
sentence_dur     = zeros(n_sentences, 1);
sentence_onsets  = zeros(n_sentences, 1);
sentence_offsets = zeros(n_sentences, 1);

for s_idx = 1:n_sentences
    fprintf('[Processing sentence %d of %d]: %s\n', s_idx, n_sentences, fnames_audio{s_idx});
    [y, fs] = audioread(fullfile(stim_dir, fnames_audio{s_idx}));

    if size(y,2) > 1
        y = mean(y,2);
    end

    % speech onset detection
    %----------------------------------------------------------------------
    % Simple Energy-Based VAD 
    % Normalize the signal 
    y_norm = y / max(abs(y));

    % figure
    % plot(y_norm)

    % Find indices where the energy is above a threshold
    % We use a moving average to avoid cutting in the middle of words
    threshold  = 0.03; 
    win_size   = round(fs * 0.1); % 100 ms window
    energy     = envelope(y_norm, win_size, 'rms');
    speech_idx = energy > threshold;

    % Compute based on raw envelope
    speech_onset_idx        = find(y_norm >= threshold);
    speech_onset_idx        = speech_onset_idx(1);
    sentence_onsets(s_idx)  = speech_onset_idx/fs;
    speech_offset_idx       = find(y_norm >= threshold);
    speech_offset_idx       = speech_offset_idx(end);
    sentence_offsets(s_idx) = (length(y_norm) - speech_offset_idx)/fs;
    sentence_dur(s_idx)     = length(y) / fs;

    figure
    hold on
    plot(y_norm)
    plot(energy)
    plot(speech_idx)
    xline(speech_onset_idx,'g')
    xline(speech_offset_idx,'g')
    hold off

    % Keep only the speech
    y = y(speech_idx);
    %----------------------------------------------------------------------

    % Determine if chunking is needed
    file_duration = length(y) / fs;

    if file_duration > max_duration_sec
        % Chunking
        chunk_samples   = max_duration_sec * fs;
        n_chunks        = floor(length(y) / chunk_samples);
        temp_chunks_psd = []; % To store 1D spectra for this file's chunks

        fprintf('-> Long file detected (%.1fs). Processing in %d chunks...\n', file_duration, n_chunks);

        for c = 1:n_chunks
            start_idx = (c-1) * chunk_samples + 1;
            end_idx   = c * chunk_samples;
            y_chunk   = y(start_idx:end_idx);
            
            % Process Chunk
            p_map_db = plot_modspecgram_TH(y_chunk', fs, 'mfmax', mfmax, 'no_colorbar');
            
            % Initialization for the first run ever
            if isempty(all_psds) && c == 1
                f_mod = linspace(0, mfmax, size(p_map_db, 2)); 
            end
            close(gcf);
            
            % Convert to linear and collapse acoustic bands
            p_map_lin       = 10.^(p_map_db / 20);
            temp_chunks_psd = [temp_chunks_psd; mean(p_map_lin, 1)];
        end
        
        % Average the chunks to get one representative spectrum for this file
        file_spectrum = mean(temp_chunks_psd, 1);

    else
        % Standard
        p_map_db = plot_modspecgram_TH(y', fs, 'mfmax', mfmax, 'no_colorbar');
        
        if isempty(all_psds)
            f_mod = linspace(0, mfmax, size(p_map_db, 2)); 
        end
        close(gcf);
        
        p_map_lin     = 10.^(p_map_db / 20);
        file_spectrum = mean(p_map_lin, 1);
    end

    % Collect for final grand average across all files
    all_psds = [all_psds; file_spectrum];
end

% Pool all recordings (Grand Average)
final_avg_lin = mean(all_psds, 1);

% Convert final result back to dB
final_avg_db = 20 * log10(final_avg_lin);

% Plotting 
%---------
figure('Color', 'w');
plot(f_mod, final_avg_db, 'LineWidth', 2, 'Color', [0.1 0.4 0.7]);
xlim([0.5 50]); % Focus on the speech-relevant range (syllabic/phonemic)
grid on;
xlabel('Modulation Frequency (Hz)', 'FontWeight', 'bold');
ylabel('Average Power (dB re: max)', 'FontWeight', 'bold');
title('Grand Average Modulation Spectrum', 'FontSize', 12);

% Add indicator for syllabic rate
line([4 4], get(gca, 'YLim'), 'Color', 'r', 'LineStyle', '--');
text(4.5, max(get(gca, 'YLim'))*0.9, 'Syllabic Rate (~4 Hz)', 'Color', 'r');

figure('Color', 'w');
subplot(1,3,1)
histogram(sentence_dur, 'BinWidth', 0.1)
title('Sentence durations')
xlabel('t / s')
axis square

subplot(1,3,2)
histogram(sentence_onsets*1000, 'BinWidth', 5)
title('Sentence Onsets')
xlabel('t / s')
axis square

subplot(1,3,3)
histogram(sentence_offsets*1000, 'BinWidth', 5)
title('Sentence Offsets (from end of audiosignal)')
xlabel('t / s')
axis square