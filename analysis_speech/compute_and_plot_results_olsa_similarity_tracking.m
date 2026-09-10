%--------------------------------------------------------------------------
% Till Habersetzer, 21.04.2026 (Updated: 03.07.2026)
% Communication Acoustics, CvO University Oldenburg
% till.habersetzer@uol.de 
%
% SCRIPT DESCRIPTION
%--------------------------------------------------------------------------
% This script performs a comprehensive analysis and visualization of 
% pre-computed decoder prediction data and acoustic similarity.
%
% The main workflow is as follows:
%   1.  **Load Data:** Loads prediction correlation values ('sorted', 
%       'sorted_circshift', 'shuffled', 'shuffled_circshift', 'single_sentences', 
%       'mean_sentence', 'shift') across multiple subjects, conditions, and sensors.
%   2.  **Preprocess:** Maps subject-specific SNRs to a global grid, computes
%       descriptive statistics in fisher z-space, and calculates baseline correlations
%       with SNR/Intelligibility (including FDR correction for p-values).
%   3.  **Assess Null Distributions:** Compares prediction accuracies against 
%       null distributions (shuffled/shifted) using paired t-tests. The 
%       assumption of normality for the paired differences is tracked via 
%       the Lilliefors test.
%   4.  **Assess Neural Tracking:** Fits Linear Mixed-Effects (LME), pooled 
%       linear, and individual linear models to quantify the relationship 
%       between neural tracking and speech intelligibility (slopes). It also 
%       computes effect sizes across conditions.
%   5.  **Acoustic Envelope Analysis:**        
%       - Extracts stimulus envelopes and calculates pairwise similarity 
%         (Pearson's rho and RMSE) between all sentence pairs.
%       - Computes similarity to the mean sentence envelope.
%       - Computes similarity decay across circular and linear time-shifts 
%         (cross-correlation) to provide a baseline for neural lag analysis.
%   6.  **Modulation Spectra Analysis:** Computes and compares the modulation 
%       spectra of acoustic envelopes and neural tracking slopes. Calculates 
%       the cross-power spectrum to identify shared temporal periodicities 
%       (e.g., syllable or word rates).
%   7.  **Compute Neuro-Acoustic Statistics:** Correlates neurally-derived 
%       slopes (and effect sizes) with acoustic similarity metrics (mean/median)
%       to assess how stimulus properties drive decoding accuracy.
%
% FIGURES GENERATED
%--------------------------------------------------------------------------
% The script generates several multi-panel figures:
%
%   1.  **Correlation vs. SNR & Intelligibility:** Visualizes neural tracking 
%       accuracy across SNRs and Intelligibility, overlaid with linear model 
%       fits and control conditions (null, shuffled, mean-sentence).
%
%   2.  **Neural Tracking Metrics:** Distribution of tracking slopes or effect 
%       sizes across single sentences or time-shifts. Incorporates continuous
%       bottom-lines or faded point-markers to denote significance against 
%       null distributions.
%
%   3.  **Selected Trials Detail:** High-resolution view of correlation-vs-
%       intelligibility for specific sentences or shifts, showing individual 
%       subject regression lines and control distributions.
%
%   4.  **Acoustic Similarity Profiles:** 
%       - Ranked mean/median similarity of sentences.
%       - Decay of acoustic similarity across circular and linear shifts.
%       - Waveform comparisons, similarity histograms, and similarity rankings 
%       - Side-by-side plots of neural tracking vs. shift and acoustic 
%         similarity vs. shift.
%
%   5.  **Modulation & Cross-Power Spectra:** Visualizes the modulation power 
%       spectra of acoustic envelopes and neural tracking, alongside their 
%       cross-spectrum, highlighting dominant rhythmic frequencies.
%
%   6.  **Acoustic Similarity vs. Neural Tracking:** Scatter plots correlating 
%       acoustic properties (x-axis) with neural tracking metrics (y-axis). 
%       Utilizes custom Alpha-mapping to visually fade slopes/trials that 
%       do not significantly differ from the null distribution.
%
%   7.  **Single-Subject Example Data:** Visualizations of individual subject 
%       scatter plots mapping neural tracking against intelligibility across 
%       different selected sentences and conditions.
%
%--------------------------------------------------------------------------

close all
clearvars
clc 

% Import main settings
%---------------------
settings_decoding

% Display ICA Configuration Feedback
%--------------------------------------------------------------------------
if isfield(settings, 'use_ica') && settings.use_ica
    fprintf('-> CONFIGURATION: ICA is ON (Suffix ''_ica-on'' applied to filenames).\n\n');
else
    fprintf('-> CONFIGURATION: ICA is OFF.\n\n');
end

%% Main settings 
%--------------------------------------------------------------------------

% Subjects 
% subjects = [2:18,20]; % remove 0 (pilot) 1,19: hearing loss
subjects = 0:20; % all
n_subj   = length(subjects);

% Conditions
conditions = {'ses-01','ses-02','ses-pooled'}; % ses-01, ses-02, ses-03 -> ses-pooled
n_cond     = length(conditions);

% Select sensors - write into settings
sensors4analysis = settings.analysis.sensors;
% sensors4analysis = {'biochannels','eeg'};
n_sens           = length(sensors4analysis);

% Analysis parameters
bpfreq           = [0.5,8];
% decoder_intgrwin = [0,75];
% bpfreq           = [4, 8];
% decoder_intgrwin = [0,250];
% bpfreq           = [0.5,4];
% bpfreq           = [0.5,30];
decoder_intgrwin = [0,400];

% Select if analysis for concatenated or single sentences should be loaded
concat_type4analysis = 'concat'; % sentences were concatenated before correlation
% concat_type4analysis = 'single'; % single sentences used for correlation analysis

corr_metric = settings.decoding.decoder.corr_metric; % accuracy metric

% Select conditions for mutiple comparison correction of p-values
mcp_method = 'fdr';
% mcp_method = 'bonferroni';
% mcp_method = 'bonferroni-holm';

mcp_scope = 'per-condition';
% mcp_scope = 'all';

% Other settings
%--------------------------------------------------------------------------

% Take absolute value of correlation
% apply_abs = true;
apply_abs  = false;
n_snrs_max = 6;

% Apply atanh transformation for statistics
apply_fisher_z = true;  

% Apply zscoring of all trials
apply_normalization = settings.decoding.apply_normalization;

% Remove onset (Load preprocessed data without OLSA sentence onsets)
% remove_onset = settings.remove_onset;
remove_onset = true;
onset_tmax   = settings.onset_tmax;

% Add paths
%--------------------------------------------------------------------------
addpath(fullfile(settings.path2project,'analysis','helper_functions'));
addpath(genpath(settings.path2mtrftoolbox));

% Remap sensor labels based on selection
%---------------------------------------
% Initialize output array
sensor_labels = cell(size(sensors4analysis));

for i = 1:length(sensors4analysis)
    switch sensors4analysis{i}
        case 'meg'
            sensor_labels{i} = 'MEG';
        case 'eeg'
            sensor_labels{i} = 'EEG';
        case 'ear_eeg'
            sensor_labels{i} = 'ear-EEG';
        otherwise
            % Fallback for unknown sensors (e.g., make it just uppercase)
            sensor_labels{i} = upper(strrep(sensors4analysis{i}, '_', '-'));
    end
end

%% Import and extract data
%--------------------------------------------------------------------------
% Import computed correlation values 
% Import snrs and intelligibilities to check consistency

fprintf('--- Loading all data ---\n');

% Determine the transformation functions once
if apply_abs
    abs_func = @abs;
    fprintf('Polarity: Applying Absolute Value transformation.\n');
else
    abs_func = @(x) x; % Identity
end

% Define the Fisher Z Function and its Inverse
if apply_fisher_z
    % Clips to 0.999 to avoid Inf at atanh(1)
    fisher_z_func     = @apply_fisher_z_transform;
    inv_fisher_z_func = @apply_inverse_fisher_z_transform;
    fprintf('Statistics: Applying Fisher Z-transform (atanh).\n');
else
    fisher_z_func     = @(x) x; % Identity
    inv_fisher_z_func = @(x) x; % Identity
end

% Use structs for clean organization
prediction_data = struct();
data_fields_all = {'sorted', 'sorted_circshift', 'shuffled', 'shuffled_circshift', 'single_sentences', 'mean_sentence', 'shift'};
   
% Pre-allocate cell arrays within the struct
for fn_idx = 1:numel(data_fields_all)
    field                   = data_fields_all{fn_idx};
    prediction_data.(field) = cell(n_subj, n_cond, n_sens);
end

% Pre-allocate standard numeric arrays
snrs_all                      = nan(n_subj, n_cond, n_sens, n_snrs_max); 
intelligibilities_all         = nan(n_subj, n_cond, n_sens, n_snrs_max); 
intelligibilities_adapted_all = nan(n_subj, n_cond, n_sens, n_snrs_max); 
n_trials_all                  = nan(n_subj, n_cond, n_sens, n_snrs_max);
subjectnames                  = cell(1, n_subj);
fnames_audio                  = {};
shift_samples                 = [];
shift_time                    = [];

% Main Loading Loop 
%------------------
for sub_idx = 1:n_subj
    subject               = sprintf('sub-%02d', subjects(sub_idx));
    subjectnames{sub_idx} = subject;

    for con_idx = 1:n_cond
        condition = conditions{con_idx};

        for sens_idx = 1:n_sens
            sensor = sensors4analysis{sens_idx};
            
            % Load Data 
            %----------
            fname = sprintf(settings.fnames.prediction_decoder_olsa_sim, subject, condition, sensor, ...
                            settings.helper.formatFreqBand(bpfreq), ...
                            settings.helper.formatIntgrWin(decoder_intgrwin), sprintf('%s_sentences', concat_type4analysis));

            if remove_onset
                fname = strrep(fname, 'decoding', 'onset-removed_decoding');
            end

            data_struct = load(fullfile(settings.path2decoding_testing(subject), fname)); 
            data        = data_struct.results;

            % Store SNRs and Intelligibilities 
            %---------------------------------
            n_snrs_sub                                                      = length(data.stats_olsa.snrs);
            snrs_all(sub_idx, con_idx, sens_idx, 1:n_snrs_sub)              = data.stats_olsa.snrs;
            intelligibilities_all(sub_idx, con_idx, sens_idx, 1:n_snrs_sub) = data.stats_olsa.intelligibilities;
            n_trials_all(sub_idx, con_idx, sens_idx, 1:n_snrs_sub)          = data.stats_olsa.n_trials_per_snr;
            
            % Adapt intelligibilities
            intellis                 = data.stats_olsa.intelligibilities;
            intellis(intellis <= 5)  = 0;
            intellis(intellis >= 95) = 100;
            intelligibilities_adapted_all(sub_idx, con_idx, sens_idx, 1:n_snrs_sub) = intellis;
            
            % Store Prediction Data 
            %----------------------
            for fn_idx = 1:numel(data_fields_all)
                field = data_fields_all{fn_idx};
                
                prediction_data.(field){sub_idx, con_idx, sens_idx} = fisher_z_func(abs_func(data.stats_olsa.(field).r));
 
                switch field
                    case {'single_sentences'}
                        fnames_audio_new = data.stats_olsa.('single_sentences').fnames;

                        if isempty(fnames_audio) 
                            fnames_audio = fnames_audio_new;
                        else 
                            assert(isequal(fnames_audio, fnames_audio_new), 'DataError: Audio filenames do not match across data structures!');
                        end
                        clear fnames_audio_new
                    case {'shift'}
                        shift_samples_new = data.stats_olsa.shift.shift_samples;
                        shift_time_new    = data.stats_olsa.shift.shift_time;
                        
                        % Check and assign or validate shift_samples
                        if isempty(shift_samples) 
                            shift_samples = shift_samples_new;
                        else 
                            assert(isequal(shift_samples, shift_samples_new), 'DataError: shift_samples do not match across data structures!');
                        end
                        
                        % Check and assign or validate shift_time
                        if isempty(shift_time) 
                            shift_time = shift_time_new;
                        else 
                            assert(isequal(shift_time, shift_time_new), 'DataError: shift_time does not match across data structures!');
                        end

                        clear shift_samples_new shift_time_new;
                end
            end % Loop over data fields
            clear data_struct data intellis n_snrs_sub

            fprintf('%s / %s / %s loaded.\n', subject, condition, sensor);
        end % Loop over sensors
    end % Loop over conditions
end % Loop over subjects
fprintf('--- Data loading complete. ---\n\n');

%% Preprocessing and Statistical Computation
%--------------------------------------------------------------------------
% - SNR Mapping: Aligns subject-specific SNR/Intelligibility levels to a global grid
% - Descriptive Stats: Computes Mean, SEM, and Percentiles (2.5%–97.5%) across distributions.
% - Correlation: Calculates Spearman's Rho for neural data vs. SNR and Intelligibility.
% - Inference: Applies FDR correction to p-values across sensors and conditions.
fprintf('--- Preprocessing and computing statistics ---\n');

% Consistency Check and SNR Mapping 
%--------------------------------------------------------------------------
% Check consistency of intelligibilities and snrs across sessions, sensors 
% and align them across subjects
unique_intelli_values     = [0, 20, 40, 50, 80, 100]; % Expected values
sub_snr_indices           = false(n_subj, n_snrs_max); % Snr mappping for each subject
intelligibilities         = nan(n_subj,n_snrs_max);
intelligibilities_adapted = nan(n_subj,n_snrs_max);
snrs                      = nan(n_subj,n_snrs_max);

for sub_idx = 1:n_subj

    % Get unique adapted intelligibilities and snrs for the subject across sessions & sensors
    sub_intellis         = intelligibilities_all(sub_idx,:,:,:);
    sub_intellis         = unique(sub_intellis(~isnan(sub_intellis)));
    sub_intellis_adapted = intelligibilities_adapted_all(sub_idx, :, :, :);
    sub_intellis_adapted = unique(sub_intellis_adapted(~isnan(sub_intellis_adapted)));
    sub_snrs             = snrs_all(sub_idx,:,:,:);
    sub_snrs             = unique(sub_snrs(~isnan(sub_snrs)));

    % Create a logical map of which SNRs this subject has
    sub_snr_idx = ismember(unique_intelli_values, sub_intellis_adapted);

    expected_intellis = 6;
    if strcmp(subjectnames{sub_idx}, 'sub-00')
        expected_intellis = 5; 
    end
    if sum(sub_snr_idx) ~= expected_intellis
        error('Unexpected number of intelligibilities for %s!', subjectnames{sub_idx});
    end

    % Store intelligibilities and snrs
    sub_snr_indices(sub_idx, :)                     = sub_snr_idx;
    intelligibilities(sub_idx, sub_snr_idx)         = sub_intellis;
    intelligibilities_adapted(sub_idx, sub_snr_idx) = sub_intellis_adapted;
    snrs(sub_idx,sub_snr_idx)                       = sub_snrs;

    clear expected_intellis sub_intellis sub_intellis_adapted sub_snrs sub_snr_idx
end
fprintf('Subject-specific SNR maps computed.\n');

% Computation of Statistics 
%--------------------------------------------------------------------------
% Mean, Standard Deviation, Percentiles for sampled distributions
prctiles = [2.5, 25, 50, 75, 97.5]; % hard-coded
n_shifts = length(shift_samples);
n_audio  = length(fnames_audio);

% Pre-allocate results in a clean struct
sim_vals = struct();
for fn_idx = 1:numel(data_fields_all) 
    field = data_fields_all{fn_idx};
    switch field
        case {'single_sentences'}
            sim_vals.(field) = nan(n_subj, n_cond, n_sens, n_snrs_max, n_audio);
        case {'shift'}
            sim_vals.(field) = nan(n_subj, n_cond, n_sens, n_snrs_max, n_shifts);
        otherwise  
            sim_vals.(field) = nan(n_subj, n_cond, n_sens, n_snrs_max); 
    end
end % fields

% Main Computation Loop 
for sub_idx = 1:n_subj
    snr_map    = sub_snr_indices(sub_idx, :); % Get the logical indices for this subject
    n_snrs_sub = sum(snr_map); % Number of SNRs for this subject
    
    for con_idx = 1:n_cond
        for sens_idx = 1:n_sens

            % Process permutation data 
            %-------------------------
            for fn_idx = 1:numel(data_fields_all)
                field = data_fields_all{fn_idx};
                
                % Check which data has benn loaded
                switch concat_type4analysis

                    case 'concat'
                        switch field
                            case {'single_sentences', 'shift'}
                                sim_vals.(field)(sub_idx, con_idx, sens_idx, snr_map, :) = prediction_data.(field){sub_idx, con_idx, sens_idx};
                            case {'sorted_circshift', 'shuffled', 'shuffled_circshift'}
                                % compute distribution mean
                                sim_vals.(field)(sub_idx, con_idx, sens_idx, snr_map, :) = mean(prediction_data.(field){sub_idx, con_idx, sens_idx}, 2);
                            otherwise  
                                sim_vals.(field)(sub_idx, con_idx, sens_idx, snr_map)    = prediction_data.(field){sub_idx, con_idx, sens_idx}; 
                            
                        end % switch

                    case 'single'
                        % Compute averages across single sentence
                        % correlation values
                        %----------------------------------------
                        switch field
                            case {'single_sentences', 'shift'}
                                sim_vals.(field)(sub_idx, con_idx, sens_idx, snr_map, :) = squeeze(mean(prediction_data.(field){sub_idx, con_idx, sens_idx}, 2, 'omitnan'));
                            case {'sorted_circshift', 'shuffled', 'shuffled_circshift'}
                                % compute distribution mean
                                sim_vals.(field)(sub_idx, con_idx, sens_idx, snr_map, :) = mean(prediction_data.(field){sub_idx, con_idx, sens_idx}, 2, 'omitnan');
                            otherwise  
                                sim_vals.(field)(sub_idx, con_idx, sens_idx, snr_map)    = squeeze(mean(prediction_data.(field){sub_idx, con_idx, sens_idx}, 2, 'omitnan')); 
                            
                        end % switch

                end % concatenation type
            end % fields
        end % sensors
    end % conditions
end % subjects

% Add grand averages
%--------------------------------------------------------------------------
sim_vals_gavg = struct();
for fn_idx = 1:numel(data_fields_all)
    field         = data_fields_all{fn_idx};
    sim_vals_data = sim_vals.(field);

    sim_vals_gavg.(field).gavg     = squeeze(mean(sim_vals_data, 1, 'omitnan'));
    n_subj_mat                     = squeeze(sum(~isnan(sim_vals_data), 1, 'omitnan'));
    % Due to being in z-space, std computation is ok, but not correct
    sim_vals_gavg.(field).gavg_sem = squeeze(std(sim_vals_data, 0, 1, 'omitnan'))./sqrt(n_subj_mat);
    clear sim_vals_data n_subj_mat field
end % fields

fprintf('Averages, metrics and grandaverages computed for all conditions.\n');
% clear prediction_data

% Add correlation and p values for neural tracking figure
%--------------------------------------------------------------------------
snrs_data    = snrs(:); 
intells_data = intelligibilities_adapted(:);

% Only compute for subset of data fields
data_fields = {'sorted', 'sorted_circshift', 'shuffled', 'shuffled_circshift', 'single_sentences', 'mean_sentence', 'shift'};

stats_conditions = struct();
for fn_idx = 1:numel(data_fields) 
    field = data_fields{fn_idx};

    % Determine if this field needs an extra dimension
    if strcmp(field, 'single_sentences')
        n_extra       = n_audio; 
        has_extra_dim = true;
    elseif strcmp(field, 'shift')
        n_extra       = n_shifts;
        has_extra_dim = true;
    else
        n_extra       = 1;
        has_extra_dim = false;
    end

    % Initialize appropriately based on dimensions
    if ~has_extra_dim
        stats_conditions.(field).snrs.p    = nan(n_cond, n_sens);
        stats_conditions.(field).snrs.r    = nan(n_cond, n_sens); 
        stats_conditions.(field).intells.p = nan(n_cond, n_sens);
        stats_conditions.(field).intells.r = nan(n_cond, n_sens);
    else
        stats_conditions.(field).snrs.p    = nan(n_cond, n_sens, n_extra);
        stats_conditions.(field).snrs.r    = nan(n_cond, n_sens, n_extra); 
        stats_conditions.(field).intells.p = nan(n_cond, n_sens, n_extra);
        stats_conditions.(field).intells.r = nan(n_cond, n_sens, n_extra);
    end

    for con_idx = 1:n_cond
        for sens_idx = 1:n_sens

            for extra_idx = 1:n_extra

                % Extract data
                if has_extra_dim
                    sim_data = squeeze(sim_vals.(field)(:, con_idx, sens_idx, :, extra_idx));
                else
                    sim_data = squeeze(sim_vals.(field)(:, con_idx, sens_idx, :)); 
                end
                sim_data  = inv_fisher_z_func(sim_data(:));
                valid_idx = ~isnan(sim_data) & ~isnan(snrs_data) & ~isnan(intells_data);

                % Correlation with SNR
                [r_pearson, p_pearson] = corr(snrs_data(valid_idx), sim_data(valid_idx), 'Type', 'Pearson'); 
    
                stats_conditions.(field).snrs.p(con_idx, sens_idx, extra_idx) = p_pearson;
                stats_conditions.(field).snrs.r(con_idx, sens_idx,extra_idx)  = r_pearson;
    
                % Correlation with Intelligibility
                [r_pearson, p_pearson] = corr(intells_data(valid_idx), sim_data(valid_idx), 'Type', 'Pearson'); 
    
                stats_conditions.(field).intells.p(con_idx, sens_idx, extra_idx) = p_pearson;
                stats_conditions.(field).intells.r(con_idx, sens_idx, extra_idx) = r_pearson;
    
                clear sim_data valid_idx r_pearson p_pearson
            end % sensors
        end % conditions
    end % extra trials
    
    % Apply multiple comparisons correction
    %--------------------------------------
    dim_idx                                     = 1; % position of conditions argument
    [stats_corrected, n_corrected]              = apply_multiple_comparison_correction(stats_conditions.(field).snrs.p, mcp_method, mcp_scope, dim_idx);
    stats_conditions.(field).snrs.p_adj         = stats_corrected;
    stats_conditions.(field).snrs.p_adj_n_tests = n_corrected;

    dim_idx                                        = 1; % position of conditions argument
    [stats_corrected, n_corrected]                 = apply_multiple_comparison_correction(stats_conditions.(field).intells.p, mcp_method, mcp_scope, dim_idx);
    stats_conditions.(field).intells.p_adj         = stats_corrected;
    stats_conditions.(field).intells.p_adj_n_tests = n_corrected;

end
clear snrs_data intells_data stats_corrected n_corrected

fprintf('Correlation and p-values computed.\n');   

%% Compute statistical contrasts against null distribution
%--------------------------------------------------------------------------

% Select null distribution
%-------------------------
null_distr_type = 'shuffled_circshift';
% null_dist = 'sorted_circshift';

% Select data fields for comparison
%----------------------------------
data_fields = {'single_sentences',  'shift'};

% Choose test-statistic
%----------------------
% stat_test_option = 'option1'; % non-parameteric (permutation test)
stat_test_option = 'option2'; % parameteric (Welch t-test / Paired t-test / Wilcoxon Signed-Rank)

% Number of permutations for testing
n_perms_test = 1000;

% Set alpha level for comparison against null distribution when pooled
% across SNRs
alpha_level = 0.05;

% Initialize data
%----------------
stats_null_distr = struct();
for fn_idx = 1:numel(data_fields)
    field = data_fields{fn_idx};

    % Determine if this field needs an extra dimension
    if strcmp(field, 'single_sentences')
        n_extra = n_audio; 
    elseif strcmp(field, 'shift')
        n_extra = n_shifts;
    else
        n_extra = 1;
    end

    stats_null_distr.(field).p               = nan(n_cond, n_sens, n_snrs_max, n_extra);
    stats_null_distr.(field).check_normality = zeros(n_cond,3); % global counter / total per condition (across sensors, snrs, trials)

    % Assess overall difference to null distribution (pooled across snrs)
    stats_null_distr.(field).difference2null = false(n_cond, n_sens, n_extra);
end

% Statistical Testing
%--------------------
for fn_idx = 1:numel(data_fields) 
    field = data_fields{fn_idx};

    % Determine if this field needs an extra dimension
    if strcmp(field, 'single_sentences')
        n_extra = n_audio; 
    elseif strcmp(field, 'shift')
        n_extra = n_shifts ;
    else
        n_extra = 1;
    end

    for con_idx = 1:n_cond
        for sens_idx = 1:n_sens

            % Statistical Test for difference
            %--------------------------------------------------------------
            stats = nan(1, n_snrs_max);
            for snr_idx = 1:n_snrs_max

                % Null distribution
                data1 = squeeze(sim_vals.(null_distr_type)(:, con_idx, sens_idx, snr_idx)); 
                 
                for extra_idx = 1:n_extra

                    data2 = squeeze(sim_vals.(field)(:, con_idx, sens_idx, snr_idx, extra_idx)); 
                 
                    switch stat_test_option
                    case 'option1' % Permutation test
                        n_perms   = n_perms_test;
                        tail      = 'two';
                        is_paired = true;
                        results   = compute_pairwise_permutation_tests([data1, data2], [1,2], n_perms, tail, is_paired);

                        stats_null_distr.(field).p(con_idx, sens_idx, snr_idx, extra_idx) = results.p_values; 
                        clear results
                    case 'option2' % Paired t-test 
              
                        % Check for Normality (Lilliefors test)
                        % h_norm = 0 (Normal), h_norm = 1 (Not Normal)
                        [h_norm, ~] = lillietest(data2 - data1);
                        % Total amount of tests
                        stats_null_distr.(field).check_normality(con_idx, 2) = stats_null_distr.(field).check_normality(con_idx, 2) + 1; % increment
                    
                        % Perform t-test anyway
                        % NOTE: We always use the paired t-test. The ~5-8% normality violation 
                        % rate aligns with the expected false positive rate (alpha = 0.05) of the 
                        % Lilliefors test. Furthermore, the t-test is highly robust to minor deviations, 
                        % and dynamically switching to Wilcoxon inflates Type I error rates.
                        [~, p_val, ~, ~] = ttest(data2, data1, 'Tail', 'both'); % x-y

                        if h_norm == 0
                            % Parametric: Paired T-Test
                            % [~, p_val, ~, ~] = ttest(data2, data1, 'Tail', 'both');
                            stats_null_distr.(field).check_normality(con_idx, 1) = stats_null_distr.(field).check_normality(con_idx, 1) + 1; % increment
                        else
                            % Non-parametric: Wilcoxon Signed-Rank
                            % [p_val, ~, ~] = signrank(data2, data1, 'tail', 'right');
                        end

                        stats_null_distr.(field).p(con_idx, sens_idx, snr_idx, extra_idx) = p_val;
                        clear p_val h_norm
                    end % test option
                end % extra trials
            end % snrs
        end % sensors
    end % conditions

    dim_idx                                = 1; % position of conditions argument
    [stats_corrected, n_corrected]         = apply_multiple_comparison_correction(stats_null_distr.(field).p, mcp_method, mcp_scope, dim_idx);
    stats_null_distr.(field).p_adj         = stats_corrected;
    stats_null_distr.(field).p_adj_n_tests = n_corrected;
 
    fprintf('Null distribution statistic for %s fields computed.\n', field);
    clear stats_corrected n_corrected dim_idx
end % data fields

fprintf('Null distribution statistic for all data fields computed (I).\n');

% Check normality counter
%------------------------
% Compute pass rate
for fn_idx = 1:numel(data_fields)
    field = data_fields{fn_idx};

    vals                                           = stats_null_distr.(field).check_normality;
    stats_null_distr.(field).check_normality(:, 3) = (vals(:,1)./vals(:,2)) * 100;
    clear vals
end

% In percentage
%--------------
% stats_null_distr.('single_sentences').check_normality(:,3)
% stats_null_distr.('shift').check_normality(:,3)

% Compute pooled contrasts across snrs
%-------------------------------------
for fn_idx = 1:numel(data_fields) 
    field = data_fields{fn_idx};

    % Determine if this field needs an extra dimension
    if strcmp(field, 'single_sentences')
        n_extra = n_audio; 
    elseif strcmp(field, 'shift')
        n_extra = n_shifts;
    else
        n_extra = 1;
    end

    for con_idx = 1:n_cond
        for sens_idx = 1:n_sens
            for extra_idx = 1:n_extra
                % Selected corrected p-values
                p_vals = squeeze(stats_null_distr.(field).p_adj(con_idx, sens_idx, :, extra_idx));
                % Count number of signifant tests
                % n_sig = sum(p_vals < alpha_level);
                n_sig = sum(p_vals(2:end) < alpha_level);
                if n_sig >= 3 % 3/5 significant
                    stats_null_distr.(field).difference2null(con_idx, sens_idx, extra_idx) = true;
                end
                clear p_vals
            end % extra trials
        end % sensors
    end % conditions
end % data fields
fprintf('Null distribution statistic for all data fields computed (II).\n');

%% Assess Neural Tracking
%--------------------------------------------------------------------------

% Select option
intercept_on = true;

% Only compute for subset of data fields
data_fields = {'sorted', 'sorted_circshift', 'shuffled', 'shuffled_circshift', 'single_sentences', 'mean_sentence', 'shift'};

% Change intelligibility values for fit: slope fits become more stable
% Adjust both in the same way

% intelligibilities4fit     = intelligibilities_adapted; % scale to some range as correlations
% unique_intelli_values4fit = unique_intelli_values;

% more staböe (slopes less small)
intelligibilities4fit     = intelligibilities_adapted/100; % scale to some range as correlations
unique_intelli_values4fit = unique_intelli_values/100;

% Initialize
%-----------
stats_neuro = struct();

for fn_idx = 1:numel(data_fields)
    field = data_fields{fn_idx};

    % Determine if this field needs an extra dimension
    if strcmp(field, 'single_sentences')
        n_extra       = n_audio; 
        has_extra_dim = true;
    elseif strcmp(field, 'shift')
        n_extra       = n_shifts;
        has_extra_dim = true;
    else
        has_extra_dim = false;
    end

    % Initialize appropriately based on dimensions
    if ~has_extra_dim
        % Linear Mixed-Effects Model
        % Save main effect and individual effects
        stats_neuro.(field).('mixed_model').intercept      = nan(n_cond, n_sens);
        stats_neuro.(field).('mixed_model').slope          = nan(n_cond, n_sens);
        stats_neuro.(field).('mixed_model').ind_intercepts = nan(n_subj, n_cond, n_sens);
        stats_neuro.(field).('mixed_model').ind_slopes     = nan(n_subj, n_cond, n_sens);
        % Save further statistic - main effect of intelligibility
        stats_neuro.(field).('mixed_model').pval           = nan(n_cond, n_sens);
        stats_neuro.(field).('mixed_model').tval           = nan(n_cond, n_sens);
        stats_neuro.(field).('mixed_model').df             = nan(n_cond, n_sens); 

        % Pooled linear model
        stats_neuro.(field).('pooled_linear_model').intercept = nan(n_cond, n_sens);
        stats_neuro.(field).('pooled_linear_model').slope     = nan(n_cond, n_sens);
        
        % Averaged individual linear model
        stats_neuro.(field).('individual_linear_model').intercept = nan(n_cond, n_sens);
        stats_neuro.(field).('individual_linear_model').slope     = nan(n_cond, n_sens);
        
        % Effect Size
        stats_neuro.(field).('effect_size') = nan(n_cond, n_sens);    
    else
        stats_neuro.(field).('mixed_model').intercept     = nan(n_cond, n_sens, n_extra);
        stats_neuro.(field).('mixed_model').slope         = nan(n_cond, n_sens, n_extra);
        stats_neuro.(field).('mixed_model').ind_intercepts= nan(n_subj, n_cond, n_sens, n_extra);
        stats_neuro.(field).('mixed_model').ind_slopes    = nan(n_subj, n_cond, n_sens, n_extra);
        stats_neuro.(field).('mixed_model').pval          = nan(n_cond, n_sens, n_extra);
        stats_neuro.(field).('mixed_model').tval          = nan(n_cond, n_sens, n_extra);    
        stats_neuro.(field).('mixed_model').df            = nan(n_cond, n_sens, n_extra); 
        
        stats_neuro.(field).('pooled_linear_model').intercept = nan(n_cond, n_sens, n_extra);
        stats_neuro.(field).('pooled_linear_model').slope     = nan(n_cond, n_sens, n_extra);
        
        stats_neuro.(field).('individual_linear_model').intercept = nan(n_cond, n_sens, n_extra);
        stats_neuro.(field).('individual_linear_model').slope     = nan(n_cond, n_sens, n_extra);
        
        stats_neuro.(field).('effect_size') = nan(n_cond, n_sens, n_extra);
    end
end

% Contrasts 
idx1        = (1:n_snrs_max-1)';
idx2        = (2:n_snrs_max)';
contrasts   = [idx1, idx2]; 
n_contrasts = size(contrasts, 1);

for fn_idx = 1:numel(data_fields) 
    field = data_fields{fn_idx};

    % Determine if this field needs an extra dimension
    if strcmp(field, 'single_sentences')
        n_extra       = n_audio; 
        has_extra_dim = true;
    elseif strcmp(field, 'shift')
        n_extra       = n_shifts;
        has_extra_dim = true;
    else
        n_extra       = 1;
        has_extra_dim = false;
    end

    for con_idx = 1:n_cond
        for sens_idx = 1:n_sens
            for extra_idx = 1:n_extra

                % Extract data
                if has_extra_dim
                    sim_data = squeeze(sim_vals.(field)(:, con_idx, sens_idx, :, extra_idx));
                else
                    sim_data = squeeze(sim_vals.(field)(:, con_idx, sens_idx, :)); 
                end

                sim_data  = inv_fisher_z_func(sim_data);
                valid_idx = ~isnan(sim_data) & ~isnan(intelligibilities4fit);

                % Linear Mixed-Effects Models
                %----------------------------------------------------------
               
                % Run Linear Mixed-Effects Model
                if intercept_on
                    % Fixed & Random Intercept + Fixed & Random Slope
                    %------------------------------------------------
                    % Reshape data into a table for Mixed-Effects
                    subject_matrix = repmat((1:n_subj)', 1, n_snrs_max);
        
                    % Flatten everything into column vectors 
                    tbl = table(categorical(subject_matrix(valid_idx)), ...
                                intelligibilities4fit(valid_idx), ... 
                                sim_data(valid_idx), ...
                                'VariableNames', {'subject_id', 'intelligibility', 'similarity_metric'});
    
                    % Fit the model
                    formula_lme = 'similarity_metric ~ 1 + intelligibility + (1 + intelligibility | subject_id)';
                    lme         = fitlme(tbl, formula_lme);
                    
                    % Extract Fixed Effects (Global)
                    intercept = lme.Coefficients.Estimate(1);
                    slope_row = strcmp(lme.CoefficientNames, 'intelligibility');
                    slope     = lme.Coefficients.Estimate(slope_row);
                    pval      = lme.Coefficients.pValue(slope_row); % idx for slope
                    tval      = lme.Coefficients.tStat(slope_row);  
                    dfval     = lme.Coefficients.DF(slope_row);
    
                    % Extract Random Effects (Individual deviations)
                    % Because there are 2 random effects (intercept and slope), beta_random
                    % is a (2 * n_subj) x 1 vector. We reshape it to N x 2.
                    [beta_random, bnames] = randomEffects(lme);
                    idx_int               = strcmp(bnames.Name, '(Intercept)');
                    idx_slope             = strcmp(bnames.Name, 'intelligibility');
    
                    % Calculate absolute individual lines
                    ind_intercepts = intercept + beta_random(idx_int);
                    ind_slopes     = slope     + beta_random(idx_slope);
                    
                else 
                    % Zero Intercept + Fixed & Random Slope
                    %------------------------------------------------
                    valid_idx2      = valid_idx;
                    % Remove 0 intelligibility condition
                    valid_idx2(:,1) = false;
                    % Reshape data into a table for Mixed-Effects
                    subject_matrix = repmat((1:n_subj)', 1, n_snrs_max);
    
                    % Flatten everything into column vectors 
                    tbl = table(categorical(subject_matrix(valid_idx2)), ...
                                intelligibilities4fit(valid_idx2), ...
                                sim_data(valid_idx2), ...
                                'VariableNames', {'subject_id', 'intelligibility', 'similarity_metric'});
    
                    % Fit the model: '-1' removes fixed intercept, '-1' inside () removes random intercept
                    formula_lme = 'similarity_metric ~ -1 + intelligibility + (-1 + intelligibility | subject_id)';
                    lme         = fitlme(tbl, formula_lme);
    
                    % Extract Fixed Effects (Global)
                    intercept = 0;
                    slope     = lme.Coefficients.Estimate(1);
                    pval      = lme.Coefficients.pValue(1); % idx for slope
                    tval      = lme.Coefficients.tStat(1);  
                    dfval     = lme.Coefficients.DF(1);
    
                    % Extract Random Effects (Individual deviations)
                    [beta_random, bnames] = randomEffects(lme);
                    idx_slope             = strcmp(bnames.Name, 'intelligibility');
    
                    % Calculate absolute individual lines
                    ind_intercepts = zeros(n_subj, 1); % Everyone starts exactly at 0
                    ind_slopes     = slope + beta_random(idx_slope);
    
                    clear valid_idx2
                end
    
                % Store LME Stats (Using extra_idx implicitly handles 2D vs 3D)
                stats_neuro.(field).('mixed_model').intercept(con_idx, sens_idx, extra_idx) = intercept;
                stats_neuro.(field).('mixed_model').slope(con_idx, sens_idx, extra_idx)     = slope;
    
                % Store the individual Random Effects
                stats_neuro.(field).('mixed_model').ind_intercepts(:, con_idx, sens_idx, extra_idx) = ind_intercepts;
                stats_neuro.(field).('mixed_model').ind_slopes(:, con_idx, sens_idx, extra_idx)     = ind_slopes;
    
                % Store statistic
                stats_neuro.(field).('mixed_model').pval(con_idx, sens_idx, extra_idx) = pval;
                stats_neuro.(field).('mixed_model').tval(con_idx, sens_idx, extra_idx) = tval;
                stats_neuro.(field).('mixed_model').df(con_idx, sens_idx, extra_idx)   = dfval;
    
                clear subject_matrix tbl formula_lme lme intercept slope beta_random bnames ind_intercepts ind_slopes idx_slope pval tval dfval
                
                % Pooled Linear Regression (The "Naive" Approach) 
                %--------------------------------------------------------------
                % - All points are assumed to be independent
                % - Ignores the fact that points are grouped by participant
                if intercept_on
                    mdl       = fitlm(intelligibilities4fit(valid_idx), sim_data(valid_idx));
                    intercept = mdl.Coefficients.Estimate(1);
                    slope     = mdl.Coefficients.Estimate(2);
                else % Set intercept to 0
                    mdl       = fitlm(intelligibilities4fit(valid_idx), sim_data(valid_idx), 'y ~ -1 + x1');
                    % 'y ~ -1 + x' removes the intercept term
                    intercept = 0;
                    slope     = mdl.Coefficients.Estimate(1);
                end     
    
                stats_neuro.(field).('pooled_linear_model').intercept(con_idx, sens_idx, extra_idx) = intercept;
                stats_neuro.(field).('pooled_linear_model').slope(con_idx, sens_idx, extra_idx)     = slope;
                clear mdl intercept slope slope_se
    
                % Averaging Individual Linear Regressions
                %--------------------------------------------------------------
                % - separate linear regressions
                % - average resulting slopes and intercepts
    
                % Preallocate arrays to store each participant's slope and intercept
                intercepts = nan(n_subj, 1);
                slopes     = nan(n_subj, 1);
    
                for sub_idx = 1:n_subj
    
                    ind_intelli_data = intelligibilities4fit(sub_idx, valid_idx(sub_idx,:));
                    ind_sim_data     = sim_data(sub_idx, valid_idx(sub_idx,:));
    
                    if intercept_on
                        mdl       = fitlm(ind_intelli_data, ind_sim_data);
                        intercept = mdl.Coefficients.Estimate(1);
                        slope     = mdl.Coefficients.Estimate(2);
                    else % Set intercept to 0
                        mdl       = fitlm(ind_intelli_data, ind_sim_data, 'y ~ -1 + x1');
                        % 'y ~ -1 + x' removes the intercept term
                        intercept = 0;
                        slope     = mdl.Coefficients.Estimate(1);
                    end     
        
                    intercepts(sub_idx) = intercept;
                    slopes(sub_idx)     = slope;
                    clear ind_intelli_data ind_sim_data intercept slope mdl 
                end
    
                stats_neuro.(field).('individual_linear_model').intercept(con_idx, sens_idx, extra_idx) = mean(intercepts, 'omitnan');
                stats_neuro.(field).('individual_linear_model').slope(con_idx, sens_idx, extra_idx)     = mean(slopes, 'omitnan');
                clear intercepts slopes
    
                % Mean effect size
                %--------------------------------------------------------------
                effect_sizes = nan(1, n_contrasts);
                for contr_idx = 1:n_contrasts
    
                    data1 = sim_data(:, contrasts(contr_idx,1));
                    data2 = sim_data(:, contrasts(contr_idx,2));
                    % Remove nans
                    idx1  = ~isnan(data1);
                    idx2  = ~isnan(data2);
                    idx   = and(idx1,idx2);
                    data1 = data1(idx);
                    data2 = data2(idx);
    
                    % Hedges gav for paired samples
                    %------------------------------
                    difference = data2 - data1;   
                    mean_diff  = mean(difference, 'omitnan');
                    std_avg    = (std(data1, 0, 'omitnan') + std(data2, 0, 'omitnan')) / 2;
        
                    % Apply Hedges' Correction and compute effect size
                    n_pairs     = sum(idx);
                    df          = n_pairs - 1;
                    j_factor    = 1 - (3 / (4 * df - 1)); 
                    effect_size = (mean_diff / std_avg) * j_factor;
    
                    % Store effect size
                    if ~ismember(effect_size,[-Inf, Inf])
                        effect_sizes(contr_idx) = effect_size;
                    end 
                    
                    clear data1 data2 difference mean_diff std_avg df j_factor effect_size idx1 idx2
                end % contrasts
                 
                % Store mean effect size
                stats_neuro.(field).('effect_size')(con_idx, sens_idx, extra_idx) = mean(effect_sizes, 'omitnan');
                clear effect_sizes valid_idx sim_data

            end % extra dimension

            fprintf('%s finished (Session: %d / Sensor: %d).\n', field, con_idx, sens_idx)

        end % sensors
    end % conditions

    % Apply multiple comparisons correction
    %--------------------------------------
    dim_idx                                              = 1; % position of conditions argument
    [stats_corrected, n_corrected]                       = apply_multiple_comparison_correction(stats_neuro.(field).('mixed_model').pval, mcp_method, mcp_scope, dim_idx);
    stats_neuro.(field).('mixed_model').pval_adj         = stats_corrected;
    stats_neuro.(field).('mixed_model').pval_adj_n_tests = n_corrected;
end % data fields

fprintf('Neurtal tracking metrics computed.\n');   

%% Correlation vs. SNR and Intelligibility
%--------------------------------------------------------------------------
% All control conditions are colored in red
% Percentiles in lower plot is based on control conditions

% User Selections - only a single condition possible
conditions2plot   = 'ses-pooled';
% datatypes_to_plot = {'sorted_circshift', 'shuffled', 'mean_sentence', 'sorted'};
datatypes_to_plot = {'shuffled_circshift', 'shuffled', 'mean_sentence', 'sorted'};
% datatypes_to_plot = {'shuffled_circshift', 'sorted_circshift', 'shuffled', 'mean_sentence', 'sorted'};

% Choose linear model for plotting
% lin_model2plot = 'pooled_linear_model';
% lin_model2plot = 'individual_linear_model';
lin_model2plot = 'mixed_model';

% Find Data and Calculate Axis Limits 
%------------------------------------
con_idx = ismember(conditions, conditions2plot);

% Aggregate all relevant data to calculate global axis limits
n_datatypes = numel(datatypes_to_plot);
all_data    = cell(1,n_datatypes);

for dt_idx = 1:numel(datatypes_to_plot)
    field            = datatypes_to_plot{dt_idx};
    sim_data         = sim_vals.(field)(:, con_idx, :, :); 
    all_data{dt_idx} = inv_fisher_z_func(sim_data(:));
end
y_min    = min([all_data{:}],[],"all"); 
y_max    = max([all_data{:}],[],"all");
ylim_min = y_min - 0.05 * (y_max - y_min);
ylim_max = y_max + 0.1 * (y_max - y_min);

x_min    = min(snrs,[],"all"); 
x_max    = max(snrs,[],"all"); 
xlim_min = x_min - 0.1 * (x_max - x_min);
xlim_max = x_max + 0.1 * (x_max - x_min);

% Plot data
%--------------------------------------------------------------------------
snrs_data = snrs(:);

% Define font sizes for plot elements
titleFontSize     = 24; % Title for each individual subplot
axisLabelFontSize = 20; % X and Y axis labels (e.g., 'SNR / dB')
tickLabelFontSize = 20; % The numbers on the axes (e.g., -20, -10, 0)
legendFontSize    = 14; % Legend text
textBoxFontSize   = 14; % The correlation text box
markerSize        = 30;

% Define plotting aesthetics
if n_datatypes>1
    % colors = {[1,0,0], [0,0,1]}; % (publication) 'null distribution', 'sorted'
    colors = {[1,0,0], [1, 0.5, 0], [0, 0.5, 0], [0,0,1], [0.5, 0, 0.5]}; % 'null distribution', 'shuffled', 'mean_sentence', 'sorted', 'new_category'

    % markers = {'s','o'}; % (publication) 'null distribution', 'sorted'
    markers = {'s','d','>','o','^'}; % 'null distribution', 'shuffled', 'mean_sentence', 'sorted', 'new_category'
else
    colors = {'k'};
end

figure('color','white','Name', 'Correlation vs. SNR','WindowState', 'maximized');
tiledlayout(2, n_sens, 'TileSpacing', 'compact', 'Padding', 'compact');
plot_handles = gobjects(1, numel(datatypes_to_plot)); % Store handles for the legend

% Correlation vs. SNR
%--------------------
for sens_idx = 1:n_sens
    ax          = nexttile;
    ax.FontSize = tickLabelFontSize;
    ax.TickLabelInterpreter = 'latex';
    hold(ax, 'on');
    
    s_combined = {}; % Use a cell array for multi-line text
    ptext      = 'Pearson:';
   
    for dt_idx = 1:n_datatypes
        field = datatypes_to_plot{dt_idx};

        % Plot the aggregated data for this specific slice
        %-------------------------------------------------
        sim_data  = squeeze(sim_vals.(field)(:, con_idx, sens_idx, :)); 
        sim_data  = inv_fisher_z_func(sim_data (:));
        valid_idx = ~isnan(sim_data) & ~isnan(snrs_data);

        plot_handles(dt_idx) = scatter(snrs_data(valid_idx), sim_data(valid_idx), ...
                                       markerSize, ... 
                                       'MarkerFaceColor', colors{dt_idx}, ...
                                       'MarkerEdgeColor', 'k', ...
                                       'MarkerFaceAlpha', 0.5, ... 
                                       'MarkerEdgeAlpha', 1, ...
                                       'Marker', markers{dt_idx});

        % Compute and display Correlation value
        %--------------------------------------
        if ~(strcmp(field, 'sorted_circshift') || strcmp(field, 'shuffled_circshift'))
            r_pearson = stats_conditions.(field).snrs.r(con_idx, sens_idx);
            p_pearson = stats_conditions.(field).snrs.p_adj(con_idx, sens_idx);
    
            % Add correlation and p-value into plot 
            if strcmp(field,'sorted')
                ptext2add = sprintf('$\\circ \\ \\rho \\approx %.2f \\ [p_{adj} < 10^{%d}]$', r_pearson, ceil(log10(p_pearson)));
            elseif strcmp(field,'shuffled')
                ptext2add = sprintf('$\\diamond \\ \\rho \\approx %.2f \\ [p_{adj} < 10^{%d}]$', r_pearson, ceil(log10(p_pearson)));
            elseif strcmp(field,'mean_sentence')
                ptext2add = sprintf('$\\triangleright \\ \\rho \\approx %.2f \\ [p_{adj} < 10^{%d}]$', r_pearson, ceil(log10(p_pearson)));
            end
            ptext = [ptext, {ptext2add}];   
            
        end

    end % datatypes
    text(0.05, 0.95, ptext, ...
             'Units', 'normalized', ... % Position relative to the axes
             'VerticalAlignment', 'top', ...
             'HorizontalAlignment', 'left', ...
             'BackgroundColor', 'white', ...
             'EdgeColor', 'black', ...
             'Interpreter', 'latex', ...
             'FontSize', textBoxFontSize, ...
             'FontName', 'Helvetica', ...
             'Color', 'k');
       
    % Set axes properties
    title(sensor_labels{sens_idx}, 'FontSize', titleFontSize);
    box on;
    grid on;
    grid minor;
    % axis square;
    xlim([xlim_min, xlim_max]);
    ylim([ylim_min, ylim_max]);
    xlabel('SNR / dB', 'FontSize', axisLabelFontSize, 'FontWeight', 'bold');
    
    % Add Y-label only to the first plot
    if sens_idx == 1
        if apply_abs
            ylabel(sprintf("|r_{%s}|", corr_metric),'FontSize', axisLabelFontSize, 'FontWeight', 'bold');
        else
            ylabel(sprintf("r_{%s}", corr_metric), 'FontSize', axisLabelFontSize, 'FontWeight', 'bold');
        end
        % Add the legend only to the first tile
        ldg = legend(plot_handles, strrep(datatypes_to_plot,'_','-'), 'Location', 'west', 'FontSize', legendFontSize);
    else
        % For all other plots in the row, hide the y-tick labels
        yticklabels([]);
    end
    
    hold(ax, 'off');

end % sensors

% Correlation vs. Intelligibility
%--------------------------------------------------------------------------
figure('color','white','Name', sprintf('Correlation vs. Intelligibility (%s)',lin_model2plot),'WindowState', 'maximized');
tiledlayout(2, n_sens, 'TileSpacing', 'compact', 'Padding', 'compact');
plot_handles = gobjects(1, numel(datatypes_to_plot)); % Store handles for the legend

% plotting offsets
if n_datatypes == 1
    offsets = 0;
else
    offset_lim = 3*max(intelligibilities4fit, [], 'all')/100;
    offsets    = linspace(-offset_lim, offset_lim, n_datatypes);
end
local_lims = [0, max(intelligibilities4fit, [], 'all')];
spread     = max(intelligibilities4fit, [], 'all')/100;

if max(local_lims)==1
    label_x     = 'Intelligibility [0-1]';
    label_xtick = string(unique_intelli_values4fit);
else
    label_x     = 'Intelligibility / %';
    label_xtick = string(unique_intelli_values);
end

for sens_idx = 1:n_sens
    ax          = nexttile;
    ax.FontSize = tickLabelFontSize;
    ax.TickLabelInterpreter = 'latex';
    hold(ax, 'on');

    ptext = 'Pearson correlation / Slope';

    % Plot control distributions
    %---------------------------
    for dt_idx = 1:n_datatypes
        field = datatypes_to_plot{dt_idx};

        % Plot the aggregated data for this specific slice
        %-------------------------------------------------
        sim_data  = squeeze(sim_vals.(field)(:, con_idx, sens_idx, :)); 
        sim_data  = inv_fisher_z_func(sim_data);
        gavg_data = inv_fisher_z_func(squeeze(sim_vals_gavg.(field).gavg(con_idx, sens_idx, :)));

        % Add jitter for distribution
        jitter    = rand(n_subj, 1); % jitter for subjects
        x_jitter  = repmat(unique_intelli_values4fit, n_subj, 1) + offsets(dt_idx) + (jitter - 0.5) * spread; % Adjust 0.2 for more/less spread
        valid_idx = ~isnan(sim_data) & ~isnan(x_jitter);

        plot_handles(dt_idx) = scatter(x_jitter(valid_idx), sim_data(valid_idx), ...
                                       markerSize, ... 
                                       'MarkerFaceColor', colors{dt_idx}, ...
                                       'MarkerEdgeColor', 'k', ...
                                       'MarkerFaceAlpha', 0.5, ... 
                                       'MarkerEdgeAlpha', 1, ...
                                       'Marker', markers{dt_idx});
   
        % Add grandaverage mean
        plot(unique_intelli_values4fit + offsets(dt_idx), gavg_data, ....
             'LineStyle', 'none', ...
             'Color', 'k', ...
             'LineWidth', 2, ...
             'Marker', markers{dt_idx}, ...
             'MarkerFaceColor', colors{dt_idx}, ...
             'MarkerSize', 10);

        % Add linear fit
        %--------------
        if strcmp(lin_model2plot, 'mixed_model')
            % Extract the individual slopes and intercepts
            ind_m = stats_neuro.(field).(lin_model2plot).ind_slopes(:, con_idx, sens_idx);
            ind_b = stats_neuro.(field).(lin_model2plot).ind_intercepts(:, con_idx, sens_idx);
            
            for sub_idx = 1:n_subj
                % Safety check to skip any NaNs
                if ~isnan(ind_m(sub_idx))
                    y_fit_ind = ind_m(sub_idx) * local_lims + ind_b(sub_idx);
                    
                    plot(local_lims, y_fit_ind, 'Color', [colors{dt_idx}, 0.3], ...
                         'LineStyle', '-', 'LineWidth', 0.5);
                end
            end
            clear ind_m ind_b y_fit_ind
        end

        m       = stats_neuro.(field).(lin_model2plot).slope(con_idx, sens_idx);
        b       = stats_neuro.(field).(lin_model2plot).intercept(con_idx, sens_idx);
        pval    = stats_neuro.(field).(lin_model2plot).pval_adj(con_idx, sens_idx);
        stars_m = get_significance_stars(pval);

        y_fit = m * local_lims + b;

        plot(local_lims, y_fit, 'Color', colors{dt_idx}, 'LineStyle', '-', 'LineWidth', 2);

        % Compute and display Correlation value
        %--------------------------------------
        if ~(strcmp(field, 'sorted_circshift') || strcmp(field, 'shuffled_circshift'))
            r_pearson     = stats_conditions.(field).intells.r(con_idx, sens_idx);
            p_pearson     = stats_conditions.(field).intells.p_adj(con_idx, sens_idx);
            stars_pearson = get_significance_stars(p_pearson);
            
            switch field
                case 'sorted'        
                    sym = '\circ';
                case 'shuffled'      
                    sym = '\diamond';
                case 'mean_sentence' 
                    sym = '\triangleright';
                otherwise            
                    sym = '';
            end

            if ~isempty(sym)
                ptext2add = sprintf('$%s \\ \\rho \\approx %.2f^{%-3s} \\ / \\ \\beta_{I} \\approx %.2f^{%-3s}$', sym, r_pearson, stars_pearson, m, stars_m);
            end
            ptext = [ptext, {ptext2add}];

        end
        clear m b yfit pval stars_m stars_pearson

    end % datatypes

    text(0.02, 0.98, ptext, ...
             'Units', 'normalized', ... % Position relative to the axes
             'VerticalAlignment', 'top', ...
             'HorizontalAlignment', 'left', ...
             'BackgroundColor', 'white', ...
             'EdgeColor', 'black', ...
             'Interpreter', 'latex', ...
             'FontSize', textBoxFontSize, ...
             'Color', 'k');

    % Set axes properties
    title(sensor_labels{sens_idx}, 'FontSize', titleFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
    box on;
    grid on;
    grid minor;
    % axis square;
    xlim([local_lims(1)-diff(local_lims)*5/100, local_lims(2)+diff(local_lims)*5/100]);
    ylim([ylim_min, ylim_max]);
    xlabel(label_x, 'FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');

    % Set the tick marks to exactly match your scaled fitting points
    xticks(unique_intelli_values4fit);
    % Overwrite the text of those ticks with your original labels
    xticklabels(label_xtick);
    
    % Add Y-label only to the first plot
    if sens_idx == 1
        if apply_abs
            ylabel(sprintf('$\\left|r_{%s}\\right|$', corr_metric), 'FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
        else
            ylabel(sprintf('$r_{%s}$', corr_metric), 'FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
        end
        % Add the legend only to the first tile
        ldg = legend(plot_handles, strrep(datatypes_to_plot,'_','-'), 'Location', 'west', 'FontSize', legendFontSize, 'Interpreter', 'latex');
    else
        % For all other plots in the row, hide the y-tick labels
        yticklabels([]);
    end
    
    hold(ax, 'off');

end % sensors

%% Visualize neural tracking metrics across "shifts/sentences"
%--------------------------------------------------------------------------

% Choose datatype
% data2plot = 'shift';
data2plot = 'single_sentences';

% Choose metric
metric2plot = 'mixed_model';
% metric2plot = 'pooled_linear_model';
% metric2plot = 'individual_linear_model';
% metric2plot = 'effect_size';

% User Selections - only a single condition possible
conditions2plot = 'ses-pooled';

% Find Data and Calculate Axis Limits 
%------------------------------------
con_idx = ismember(conditions, conditions2plot);

% Define font sizes for plot elements
titleFontSize     = 24; % Title for each individual subplot
axisLabelFontSize = 20; % X and Y axis labels (e.g., 'SNR / dB')
tickLabelFontSize = 20; % The numbers on the axes (e.g., -20, -10, 0)
markerSize        = 8;

% Aggregate all relevant data to calculate global axis limits
%------------------------------------------------------------
switch data2plot
    case 'shift'
        label_x = 'Shift / ms';
        x_vec   = shift_time*1000;
    case 'single_sentences'
        label_x = 'sentences';
        x_vec   = 1:n_audio;
end

switch metric2plot
    case {'mixed_model', 'pooled_linear_model', 'individual_linear_model'}
        sim_stats = stats_neuro.(data2plot).(metric2plot).slope(con_idx,:,:);
        label_y   = sprintf('%s (slope)', strrep(metric2plot, '_', '-'));
    case 'effect_size'
        sim_stats = stats_neuro.(data2plot).(metric2plot)(con_idx,:,:);
        label_y   = strrep(metric2plot, '_', '-');
end

y_min      = min(sim_stats(:)); 
y_max      = max(sim_stats(:));
ylim_min   = y_min - 0.05 * (y_max - y_min);
ylim_max   = y_max + 0.05 * (y_max - y_min);

x_min      = min(x_vec,[],"all"); 
x_max      = max(x_vec,[],"all"); 
xlim_min   = x_min - 0.05 * (x_max - x_min);
xlim_max   = x_max + 0.05 * (x_max - x_min);

figure('color','white','Name', sprintf('Data: %s / Metric: %s', data2plot, metric2plot), 'WindowState', 'maximized');
tiledlayout(2, n_sens, 'TileSpacing', 'compact', 'Padding', 'compact');

for sens_idx = 1:n_sens
    ax          = nexttile;
    ax.FontSize = tickLabelFontSize;
    hold(ax, 'on');

    % Plot metric
    %------------
    switch metric2plot
        case {'mixed_model', 'pooled_linear_model', 'individual_linear_model'}
            sim_stats = squeeze(stats_neuro.(data2plot).(metric2plot).slope(con_idx, sens_idx, :));
        case 'effect_size'
            sim_stats = squeeze(stats_neuro.(data2plot).(metric2plot)(con_idx, sens_idx, :));
    end

    if strcmp(data2plot, 'single_sentences')
        % Sort values for each sensor
        [sim_stats, ~] = sort(sim_stats, 'descend');
    end

    plot(x_vec, sim_stats, 'x', 'Color', [0.2 0.4 0.6], 'LineWidth', 2, 'MarkerSize', markerSize);
  
    % Set axes properties
    title(sensor_labels{sens_idx}, 'FontSize', titleFontSize);
    box on;
    grid on;
    grid minor;
    % axis square;
    xlim([xlim_min, xlim_max]);
    ylim([ylim_min, ylim_max]);
    xlabel(label_x, 'FontSize', axisLabelFontSize, 'FontWeight', 'bold');
    
    % Add Y-label only to the first plot
    if sens_idx == 1
        ylabel(label_y, 'FontSize', axisLabelFontSize, 'FontWeight', 'bold');
    else
        % For all other plots in the row, hide the y-tick labels
        yticklabels([]);
    end
    
    hold(ax, 'off');

end % sensors

sgtitle(sprintf('Data: %s / Metric: %s', strrep(data2plot,'_','-'), strrep(metric2plot,'_','-')), 'FontSize', titleFontSize, 'FontWeight', 'bold')

%% Visualize neural tracking metrics and correlaton values for selected trials
%-----------------------------------------------------------------------------

% Choose datatype
data2plot = 'shift';
% data2plot = 'single_sentences';

% Choose linear model for plotting
lin_model2plot = 'mixed_model';
% lin_model2plot = 'pooled_linear_model';
% lin_model2plot = 'individual_linear_model';

% Choose number of selection
n_sel = 3;

% User Selections - only a single condition possible
conditions2plot = 'ses-pooled';

% Aggregate all relevant data to calculate global axis limits
%------------------------------------------------------------
con_idx = ismember(conditions, conditions2plot);

sim_data   = sim_vals.(data2plot)(:, con_idx, :, :, :); 
sim_data   = inv_fisher_z_func(sim_data);
y_min      = min(sim_data(:)); 
y_max      = max(sim_data(:));
ylim_min   = y_min - 0.05 * (y_max - y_min);
ylim_max   = y_max + 0.1 * (y_max - y_min);

offset_lim = 3*max(intelligibilities4fit, [], 'all')/100;
offsets    = linspace(-offset_lim, offset_lim, n_sel);
local_lims = [0, max(intelligibilities4fit, [], 'all')];
spread     = max(intelligibilities4fit, [], 'all')/100;

if max(local_lims)==1
    label_x     = 'Intelligibility [0-1]';
    label_xtick = string(unique_intelli_values4fit);
else
    label_x     = 'Intelligibility / %';
    label_xtick = string(unique_intelli_values);
end

% Define plotting aesthetics
colors  = {[1,0,0], [0,0,1], [0, 0.5, 0], [1, 0.5, 0], [0.5, 0, 0.5]};
markers = {'d', 'o', '>', '<'; ...
           '\diamond', '\circ', '\triangleright', '\triangleleft'};
    
% Define font sizes for plot elements
sgtitleFontSize   = 24; % Super-title for the whole figure
titleFontSize     = 24; % Title for each individual subplot
axisLabelFontSize = 20; % X and Y axis labels (e.g., 'SNR / dB')
tickLabelFontSize = 20; % The numbers on the axes (e.g., -20, -10, 0)
legendFontSize    = 14; % Legend text
textBoxFontSize   = 14; % The correlation text box
markerSize        = 30;

% Correlation vs. Intelligibility
%--------------------------------------------------------------------------
figure('color','white','Name', sprintf('Correlation vs. Intelligibility (%s/%s)',data2plot, lin_model2plot),'WindowState', 'maximized');
tiledlayout(2, n_sens, 'TileSpacing', 'compact', 'Padding', 'compact');
plot_handles = gobjects(1, n_sel); % Store handles for the legend

for sens_idx = 1:n_sens
    ax                      = nexttile;
    ax.FontSize             = tickLabelFontSize;
    ax.TickLabelInterpreter = 'latex';
    hold(ax, 'on');

    ptext           = 'Pearson correlation / Slope';
    legend_combined = {}; % Use a cell array for multi-line text
  
    % Plot control distributions
    %---------------------------
    sim_stats = squeeze(stats_neuro.(data2plot).(lin_model2plot).slope(con_idx, sens_idx, :));
    sim_data  = squeeze(sim_vals.(data2plot)(:, con_idx, sens_idx, :, :));
    sim_data  = inv_fisher_z_func(sim_data);
    gavg_data = inv_fisher_z_func(squeeze(sim_vals_gavg.(data2plot).gavg(con_idx, sens_idx, :, :)));

    switch data2plot
        case 'shift'
            % Shifts are sorted
            % trial_sel = round(linspace(1, n_shifts, n_sel));
            % only up to 200 ms
            if shift_samples(floor(n_shifts/2))+1==0
                shift_offset = floor(n_shifts/2); % shifts might be centered around 0
            else
                shift_offset = 0;
            end
            trial_sel         = fliplr(shift_offset + round(linspace(1, 15, n_sel))); % start with large shift in decrease
            trial_vals4legend = round(shift_time(trial_sel)*1000); % shifts in ms
            clear shift_offset
        case 'single_sentences'
            % Sort values for each sensor
            [~, sort_idx]     = sort(sim_stats, 'ascend');
            trial_sel         = round(linspace(1, n_audio, n_sel));
            trial_vals4legend = trial_sel; % ranking number
            trial_sel         = sort_idx(trial_sel); 
            % trial_vals4legend = trial_sel; % actual sentence number

            % sentence number
    end

    % Loop over selected trials
    for sel_idx = 1:n_sel

        % Select data
        trl_idx       = trial_sel(sel_idx);
        sim_data_sel  = squeeze(sim_data(:, :, trl_idx));
        gavg_data_sel = squeeze(gavg_data(:, trl_idx));
        
        % Add jitter for distribution
        jitter    = rand(n_subj, 1); 
        x_jitter  = repmat(unique_intelli_values4fit, n_subj, 1) + offsets(sel_idx) + (jitter - 0.5) * spread; 
        valid_idx = ~isnan(sim_data_sel) & ~isnan(x_jitter);

        plot_handles(sel_idx) = scatter(x_jitter(valid_idx), sim_data_sel(valid_idx), ...
                                        markerSize, ... 
                                        'MarkerFaceColor', colors{sel_idx}, ...
                                        'MarkerEdgeColor', 'k', ...
                                        'MarkerFaceAlpha', 0.5, ... 
                                        'MarkerEdgeAlpha', 1, ...
                                        'Marker', markers{1, sel_idx});
   
        % Add grandaverage mean
        plot(unique_intelli_values4fit + offsets(sel_idx), gavg_data_sel, ....
             'LineStyle', 'none', ...
             'Color', 'k', ...
             'LineWidth', 2, ...
             'Marker', markers{1, sel_idx}, ...
             'MarkerFaceColor', colors{sel_idx}, ...
             'MarkerSize', 10);

        % Add linear fit
        %--------------
        if strcmp(lin_model2plot, 'mixed_model')
            % Extract the individual slopes and intercepts
            ind_m = stats_neuro.(data2plot).(lin_model2plot).ind_slopes(:, con_idx, sens_idx, trl_idx);
            ind_b = stats_neuro.(data2plot).(lin_model2plot).ind_intercepts(:, con_idx, sens_idx, trl_idx);
            
            for sub_idx = 1:n_subj
                % Safety check to skip any NaNs
                if ~isnan(ind_m(sub_idx))
                    y_fit_ind = ind_m(sub_idx) * local_lims + ind_b(sub_idx);
                    
                    plot(local_lims, y_fit_ind, 'Color', [colors{sel_idx}, 0.3], ...
                         'LineStyle', '-', 'LineWidth', 0.5);
                end
            end
            clear ind_m ind_b y_fit_ind
        end

        m       = stats_neuro.(data2plot).(lin_model2plot).slope(con_idx, sens_idx, trl_idx);
        b       = stats_neuro.(data2plot).(lin_model2plot).intercept(con_idx, sens_idx, trl_idx);
        pval    = stats_neuro.(data2plot).(lin_model2plot).pval_adj(con_idx, sens_idx, trl_idx);
        stars_m = get_significance_stars(pval);
       
        y_fit = m * local_lims + b;

        plot(local_lims, y_fit, 'Color', colors{sel_idx}, 'LineStyle', '-', 'LineWidth', 2);

        % Update legend entry
        %--------------------
        r_pearson     = stats_conditions.(data2plot).intells.r(con_idx, sens_idx, trl_idx);
        p_pearson     = stats_conditions.(data2plot).intells.p_adj(con_idx, sens_idx, trl_idx);
        stars_pearson = get_significance_stars(p_pearson);

        ptext2add = sprintf('$%s \\ \\rho \\approx %.2f^{%-3s} \\ / \\ \\beta_{I} \\approx %.2f^{%-3s}$', markers{2, sel_idx}, r_pearson, stars_pearson, m, stars_m);
        ptext     = [ptext, {ptext2add}];

        switch data2plot
            case 'shift'
                trl_type = sprintf('Shift %d ms',trial_vals4legend(sel_idx));
            case 'single_sentences'
                trl_type = sprintf('Sentence %d',trial_vals4legend(sel_idx));
        end

        legend_combined{sel_idx} = trl_type;
       
        clear trl_type trl_idx sim_data_sel m b yfit pval r_pearson p_pearson stars_pearson stars_m

    end % trials

    text(0.02, 0.98, ptext, ...
             'Units', 'normalized', ... % Position relative to the axes
             'VerticalAlignment', 'top', ...
             'HorizontalAlignment', 'left', ...
             'BackgroundColor', 'white', ...
             'EdgeColor', 'black', ...
             'Interpreter', 'latex', ...
             'FontSize', textBoxFontSize, ...
             'Color', 'k');

    % Set axes properties
    title(sensor_labels{sens_idx}, 'FontSize', titleFontSize, 'Interpreter', 'latex');
    box on;
    grid on;
    grid minor;
    % axis square;
    xlim([local_lims(1)-diff(local_lims)*5/100, local_lims(2)+diff(local_lims)*5/100]);
    ylim([ylim_min, ylim_max]);
    xlabel(label_x, 'FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');

    % Set the tick marks to exactly match your scaled fitting points
    xticks(unique_intelli_values4fit);
    % Overwrite the text of those ticks with your original labels
    xticklabels(label_xtick);
    
    % Add Y-label only to the first plot
    if sens_idx == 1
        if apply_abs
            ylabel(sprintf('$\\left|r_{%s}\\right|$', corr_metric), 'FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
        else
            ylabel(sprintf('$r_{%s}$', corr_metric), 'FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
        end
        % Add the legend only to the first tile
        ldg = legend(plot_handles, legend_combined, 'Location', 'west', 'FontSize', legendFontSize, 'Interpreter', 'latex');
    else
        % For all other plots in the row, hide the y-tick labels
        yticklabels([]);
    end
    
    hold(ax, 'off');

end % sensors
% sgtitle(sprintf('Correlation vs. Intelligibility (%s/%s)',strrep(data2plot,'_','-'), strrep(lin_model2plot,'_','-')), 'FontSize', sgtitleFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex')

%% Import Acoustic data and compute similarity
%--------------------------------------------------------------------------
% Description: 
%   This section imports and processes acoustic data to quantify the structural 
%   similarity between different OLSA sentences. 
%
%   Key Processing Steps:
%   1. Data Import & Preprocessing: Loads pre-computed acoustic envelopes and 
%      raw audio. Applies detection masks for cropping, handles optional onset 
%      removal, and performs amplitude normalization (e.g., max-abs-scaling).
%   2. Pairwise Similarity Matrix: Computes cross-correlation (Fisher z-transformed) 
%      and RMSE between every possible pair of sentences in the dataset using the 
%      mTRF toolbox.
%   3. Statistical Summaries: Calculates marginalized means/medians for the 
%      similarity matrices and checks the resulting distributions for normality 
%      using the Shapiro-Wilk test.
%   4. Grand Average Comparison: Evaluates the acoustic similarity (Corr/RMSE) 
%      of each individual sentence against the grand average acoustic envelope.
%
%   Outputs: Results and summary statistics are stored in the `audio_stats` structure.
%--------------------------------------------------------------------------

% Import envelopes
%--------------------------------------------------------------------------

% Extact envelopes
%-----------------
fname               = sprintf(settings.fnames.audio_olsa, settings.helper.formatFreqBand(bpfreq));     
audio_envelopes     = importdata(fullfile(settings.path2derivatives,'stimuli',fname));
fs                  = audio_envelopes.fs;
envelopes           = cell(1, n_audio);
envelopes_orig      = cell(1, n_audio);
envelopes_cropped   = cell(1, n_audio);
envelopes_detection = cell(1, n_audio);

if remove_onset
    % Remove onset 
    start_idx = floor(onset_tmax * fs) + 1;

    % Correct each audiofile
    %-----------------------
    field_names = fieldnames(audio_envelopes);
    for f_idx = 1:length(field_names)
        field_name = field_names{f_idx};
        % Apply for all fields except sampling frequency
        if ~strcmp(field_name, 'fs')
            data_field                   = audio_envelopes.(field_name);
            audio_envelopes.(field_name) = data_field(start_idx:end);
        end
        clear data_field
    end
end

% Import mean envelope
envelope_avg           = audio_envelopes.envelope_avg;
envelope_avg_detection = audio_envelopes.audio_detection_avg;
envelope_avg_cropped   = envelope_avg(envelope_avg_detection);
clear envelope_avg_detection

% Import raw audio as well
stim_dir   = fullfile(settings.path2bids,'stimuli','olsa','sentences'); 
olsa_audio = cell(1, n_audio);

% Match envelopes with detection window and crop
for trl_idx = 1:n_audio
    fname_audio                  = fnames_audio{trl_idx};
    envelopes{trl_idx}           = audio_envelopes.(sprintf('envelope_%s', fname_audio));
    audio_detection              = audio_envelopes.(sprintf('audio_detection_%s', fname_audio));
    envelopes_cropped{trl_idx}   = envelopes{trl_idx}(logical(audio_detection));
    envelopes_detection{trl_idx} = audio_detection;
    envelopes_orig{trl_idx}      = envelopes{trl_idx}(logical(audio_detection)); % crop them

    % Import raw audio
    [audio, fs_orig]    = audioread(fullfile(stim_dir,[fname_audio,'.wav']));
    olsa_audio{trl_idx} = audio(:,1)';

    if remove_onset
        % Remove onset 
        start_idx           = floor(onset_tmax * fs_orig) + 1;
        olsa_audio{trl_idx} = olsa_audio{trl_idx}(start_idx:end);
    end
    clear fname_audio audio_detection audio

end
lengths_envelopes = cellfun(@length, envelopes_cropped);
clear audio_envelopes 

% Compute envelope averages with removed sentences
%-------------------------------------------------
min_lengths              = min(length(envelope_avg_cropped), lengths_envelopes(:));
remaining_envelopes_avgs = cell(1, n_audio);

for s_idx = 1:n_audio 
    all_indices             = 1:n_audio;
    all_indices(s_idx)      = []; % remove sentence
    remaining_envelopes     = envelopes(all_indices); % Take entire envelopes without cutted tail
    max_length              = max(cellfun(@length, remaining_envelopes));
    remaining_envelopes     = cellfun(@(x) [x, zeros(1, max_length - length(x))], remaining_envelopes, 'UniformOutput', false);
    remaining_envelopes     = vertcat(remaining_envelopes{:});
    remaining_envelopes_avg = mean(remaining_envelopes(:, 1:min_lengths(s_idx)), 1);

    remaining_envelopes_avgs{s_idx} = remaining_envelopes_avg;  
end
clear all_indices remaining_envelopes remaining_envelopes_avg max_length

% Import sentences
%-----------------
envelopes_sentence = cell(n_audio,1);
for trl_idx = 1:n_audio
    envelopes_sentence{trl_idx} = fileread(fullfile(settings.path2bids,'stimuli','olsa','sentences',[fnames_audio{trl_idx},'.txt']));
end

% Add standard deviation for average envelope 
envelopes_min    = cellfun(@(x) x(:,1:min(cellfun(@numel, envelopes))), envelopes, 'UniformOutput', false);
envelopes_min    = vertcat(envelopes_min{:}); %  % Vertically concatenate all padded row vectors into a matrix
envelope_avg_std = std(envelopes_min, 1);
clear envelopes_min

% Optional: Apply normalization
%------------------------------
if apply_normalization
    % Z-score
    %--------
    % envelopes_cropped    = apply_zscore_epochs(envelopes_cropped,'audio');
    % envelope_avg_cropped = (envelope_avg_cropped-mean(envelope_avg_cropped,'all'))/std(envelope_avg_cropped,0,'all');

    % Max-abs-scaling- > does not lead to change in correlation
    %----------------
    % envelopes_cropped    = apply_scaling_epochs(envelopes_cropped,'audio');
    for s_idx = 1:n_audio
        envelopes_cropped{s_idx} = envelopes_cropped{s_idx} ./ max(abs(envelopes_cropped{s_idx}), [], 2);
        envelopes{s_idx}         = envelopes{s_idx} ./ max(abs(envelopes{s_idx}), [], 2);
        olsa_audio{s_idx}        = olsa_audio{s_idx} ./ max(abs(olsa_audio{s_idx}), [], 2);

        remaining_envelopes_avgs{s_idx} = remaining_envelopes_avgs{s_idx} ./ max(abs(remaining_envelopes_avgs{s_idx}), [], 2);
    end
    envelope_avg_cropped = envelope_avg_cropped ./ max(abs(envelope_avg_cropped), [], 2);
 
    fprintf('Normalization to neuro and audio data applied.\n')
end

% Compute correlations between all pairs of sentences
%--------------------------------------------------------------------------
[X, Y]    = ndgrid(1:n_audio, 1:n_audio);
all_pairs = [X(:), Y(:)];
all_pairs = all_pairs(all_pairs(:,1) ~= all_pairs(:,2), :); % inlude comparison with same sentence
n_pairs   = size(all_pairs, 1);

% Calculate lengths
min_lengths = min(lengths_envelopes(all_pairs(:,1)),lengths_envelopes(all_pairs(:,2)));

% Define the interval for progress updates (every 10%)
update_step = max(1, floor(n_pairs / 10));

% Init data
%----------
audio_stats                                = struct();
audio_stats.('corr').sim_vals_sentcs       = nan(n_audio);
audio_stats.('rmse').sim_vals_sentcs       = nan(n_audio);
audio_stats.('corr').sim_vals_2mean_sentc  = nan(n_audio, 1);
audio_stats.('rmse').sim_vals_2mean_sentc  = nan(n_audio, 1);

% Compute Correlations
for p_idx = 1:n_pairs

    % Get correct envelopes for correlation
    idx1 = all_pairs(p_idx, 1);
    idx2 = all_pairs(p_idx, 2);

    envelope1 = envelopes_cropped{idx1}(1:min_lengths(p_idx));
    envelope2 = envelopes_cropped{idx2}(1:min_lengths(p_idx));
  
    % Perform the correlation computation
    %------------------------------------
    % mTRF-toolbox
    [r, err] = mTRFevaluate(envelope1, envelope2, 'dim', 2, 'corr', corr_metric, 'error', 'mse');
    rmse     = sqrt(err);
            
    % 'matlab'
    % r    = corr(envelope1', envelope2', 'Type', corr_metric);
    % rmse = sqrt(mean((envelope1-envelope2).^2));

    audio_stats.('corr').sim_vals_sentcs(idx1, idx2) = apply_fisher_z_transform(r);
    audio_stats.('rmse').sim_vals_sentcs(idx1, idx2) = rmse;
   
    if mod(p_idx, update_step) == 0 || p_idx == n_pairs
        fprintf('Processed %d / %d correlations.\n', p_idx, n_pairs);
    end
    clear idx1 idx2 envelope1 envelope2 r err rmse
    
end
fprintf('Pairwise similarity metrics for audio computed.\n');

% Add marginalized distributions and overall average
%---------------------------------------------------
sim_vals_sentcs                                  = audio_stats.('corr').sim_vals_sentcs;
audio_stats.('corr').sim_vals_sentcs_marg_mean   = mean(sim_vals_sentcs, 2, 'omitnan'); % remains in z-space
audio_stats.('corr').sim_vals_sentcs_marg_median = median(sim_vals_sentcs, 2, 'omitnan'); % remains in z-space
mask                                             = tril(true(size(sim_vals_sentcs)), -1);
audio_stats.('corr').sim_vals_sentcs_mean        = mean(sim_vals_sentcs(mask), 'all', 'omitnan'); % remains in z-space
audio_stats.('corr').sim_vals_sentcs_median      = median(sim_vals_sentcs(mask), 'all', 'omitnan'); % remains in z-space

sim_vals_sentcs                                  = audio_stats.('rmse').sim_vals_sentcs;
audio_stats.('rmse').sim_vals_sentcs_marg_mean   = mean(sim_vals_sentcs, 2, 'omitnan');
audio_stats.('rmse').sim_vals_sentcs_marg_median = median(sim_vals_sentcs, 2, 'omitnan'); 
mask                                             = tril(true(size(sim_vals_sentcs)), -1);
audio_stats.('rmse').sim_vals_sentcs_mean        = mean(sim_vals_sentcs(mask), 'all', 'omitnan'); % mean similarity between all sentences
audio_stats.('rmse').sim_vals_sentcs_median      = median(sim_vals_sentcs(mask), 'all', 'omitnan'); % men similarity between all sentences

clear sim_vals_sentcs mask

% Check for normality
%--------------------
check_sim_vals_sentcs_normality_corr = nan(n_audio,1);
check_sim_vals_sentcs_normality_rmse = nan(n_audio,1);

for s_idx = 1:n_audio
    sim_vals_sentcs_distr                       = audio_stats.('corr').sim_vals_sentcs(s_idx,:);
    [h, ~, ~]                                   = swtest(sim_vals_sentcs_distr(~isnan(sim_vals_sentcs_distr)));
    check_sim_vals_sentcs_normality_corr(s_idx) = (h == 0); 
    clear h   

    sim_vals_sentcs_distr                       = audio_stats.('rmse').sim_vals_sentcs(s_idx,:);
    [h, ~, ~]                                   = swtest(sim_vals_sentcs_distr(~isnan(sim_vals_sentcs_distr)));
    check_sim_vals_sentcs_normality_rmse(s_idx) = (h == 0); 
    clear h   
end

audio_stats.('corr').check_sim_vals_sentcs_normality = check_sim_vals_sentcs_normality_corr;
audio_stats.('rmse').check_sim_vals_sentcs_normality = check_sim_vals_sentcs_normality_rmse;
clear sim_vals_sentcs_distr h check_sim_vals_sentcs_normality_corr check_sim_vals_sentcs_normality_rmse

% Compute correlations sentences and average envelope
%--------------------------------------------------------------------------

% Calculate lengths
min_lengths = min(length(envelope_avg_cropped), lengths_envelopes(:));

% Compute Correlations
for s_idx = 1:n_audio

    envelope1 = envelopes_cropped{s_idx}(1:min_lengths(s_idx));
    % envelope2 = envelope_avg_cropped(1:min_lengths(s_idx));

    % Alternatively: Compute mean execept selected envelope
    envelope2 = remaining_envelopes_avgs{s_idx};

    % Perform the correlation computation
    %------------------------------------
    % mTRF-toolbox
    [r, err] = mTRFevaluate(envelope1, envelope2, 'dim', 2, 'corr', corr_metric, 'error', 'mse');
    rmse     = sqrt(err);
            
    % 'matlab'
    % r    = corr(envelope1', envelope2', 'Type', corr_metric);
    % rmse = sqrt(mean((envelope1-envelope2).^2));

    audio_stats.('corr').sim_vals_2mean_sentc(s_idx) = apply_fisher_z_transform(r);
    audio_stats.('rmse').sim_vals_2mean_sentc(s_idx) = rmse;

    clear envelope1 envelope2  r rmse
end
fprintf('Similarity to mean sentence computed.\n');

% Add mean off distribution
%--------------------------
sim_vals_2mean_sentc                             = audio_stats.('corr').sim_vals_2mean_sentc;
audio_stats.('corr').sim_vals_2mean_sentc_mean   = mean(sim_vals_2mean_sentc(:), 'omitnan'); % remains in z-space
audio_stats.('corr').sim_vals_2mean_sentc_median = median(sim_vals_2mean_sentc(:), 'omitnan'); 

sim_vals_2mean_sentc                             = audio_stats.('rmse').sim_vals_2mean_sentc;
audio_stats.('rmse').sim_vals_2mean_sentc_mean   = mean(sim_vals_2mean_sentc(:), 'omitnan'); % remains in z-space
audio_stats.('rmse').sim_vals_2mean_sentc_median = median(sim_vals_2mean_sentc(:), 'omitnan'); 
clear sim_vals_2mean_sentc

% Correlation between
% - Correlation of each sentence envelope to mean meanvelope
% - Mean Correlation of each envelope to all others except itself
figure
scatter(audio_stats.('corr').sim_vals_2mean_sentc, audio_stats.('corr').sim_vals_sentcs_marg_mean)
xlabel('Similarity to Mean Envelope')
ylabel('Mean Similarity between Envelope Pairs')

%% Additional plots - Acoustic similarity between sentences for paper
%--------------------------------------------------------------------------
% Description:
%   This block generates a multi-panel figure to visualize the acoustic 
%   similarity between a target OLSA sentence and the rest of the dataset.
%
%   Key Visualization Steps:
%   1. Envelope & Audio Plot: Overlays the normalized temporal envelope and 
%      raw audio waveform of a selected sentence against another sentence, 
%      highlighting the shared time window used for correlation.
%   2. Distribution Histogram: Plots the distribution of correlation values 
%      between the target sentence and all other sentences, marking the 
%      target's marginalized mean/median.
%   3. Rank-Ordered Similarity: Displays all sentences ranked by their 
%      marginalized similarity scores, with the target sentence highlighted.
%--------------------------------------------------------------------------

% Select mean or median
%----------------------
acoustic_sim2use = 'mean';
% acoustic_sim2use = 'median';

% Define font sizes for plot elements
axisLabelFontSize = 20; 
tickLabelFontSize = 20; 
legendFontSize    = 12; 
textBoxFontSize   = 12; 
markerSize        = 10;

% Select data
%------------
switch acoustic_sim2use
    case 'mean'
        sim_vals_sentcs_marg = apply_inverse_fisher_z_transform(audio_stats.('corr').sim_vals_sentcs_marg_mean);
        metric_label         = 'mean';
    case 'median'
        sim_vals_sentcs_marg = apply_inverse_fisher_z_transform(audio_stats.('corr').sim_vals_sentcs_marg_median);
        metric_label         = 'median';
end

% Sort values
[sim_vals_sentcs_marg_sorted, sort_idx1] = sort(sim_vals_sentcs_marg, 'descend');
% Select first sentence for plot
s_idx1    = sort_idx1(7);
mean4plot = sim_vals_sentcs_marg(s_idx1);

% 1.) Plot envelopes + audio signal 
%--------------------------------------------------------------------------
% Select whether audio waveforms should be plotted as well
plot_waveform = true;

distr4plot     = apply_inverse_fisher_z_transform(audio_stats.('corr').sim_vals_sentcs(s_idx1, :));
% Select second sentence for plot
[~, sort_idx2] = sort(distr4plot, 'descend', 'MissingPlacement', 'last'); % missing values are placed last
s_idx2         = sort_idx2(5);

color1 = [0.8 0.1 0.1];
color2 = [0.2 0.4 0.6];
% color1 = [0.4940 0.1840 0.5560];
% color2 = [0.9290 0.6940 0.1250];

% Select sentences
%-----------------
if ~plot_waveform
    envelope1 = envelopes_cropped{s_idx1};
    envelope2 = envelopes_cropped{s_idx2};
    time_vec1 = (0:length(envelope1)-1) / fs;
    time_vec2 = (0:length(envelope2)-1) / fs;
    % Define the end time of the correlation window
    corr_len_samp = min(length(envelope1), length(envelope2));
    corr_len_time = (corr_len_samp - 1) / fs;

else % Resample
    [p, q]    = rat(fs_orig / fs); % Finds the rational factor for resampling

    % Use cropped envelopes -> generate edge artifacts, not recommended!
    % envelope1 = envelopes_cropped{s_idx1};
    % envelope2 = envelopes_cropped{s_idx2};
    % envelope1 = resample(envelope1, p, q);
    % envelope2 = resample(envelope2, p, q);

    % Resample entire envelope (not cropped) to avoid edge artifacts
    envelope1                               = resample(envelopes{s_idx1}, p, q);
    envelope2                               = resample(envelopes{s_idx2}, p, q);
    audio_detection1                        = resample(envelopes_detection {s_idx1}, p, q);
    audio_detection2                        = resample(envelopes_detection {s_idx2}, p, q);
    audio_detection1(audio_detection1>0.5)  = 1;
    audio_detection1(audio_detection1<=0.5) = 0;
    audio_detection2(audio_detection2>0.5)  = 1;
    audio_detection2(audio_detection2<=0.5) = 0;
    envelope1                               = envelope1(logical(audio_detection1));
    envelope2                               = envelope2(logical(audio_detection2));

    time_vec1     = (0:length(envelope1)-1) / fs_orig;
    time_vec2     = (0:length(envelope2)-1) / fs_orig;
    corr_len_samp = min(length(envelope1), length(envelope2));
    corr_len_time = (corr_len_samp - 1) / fs_orig;
    time_vec3     = (0:length(olsa_audio{s_idx1})-1) / fs_orig;
    time_vec4     = (0:length(olsa_audio{s_idx2})-1) / fs_orig; 
    clear audio_detection1 audio_detection2
end

figure('Name', 'Plot envelopes', 'Color', 'w', 'WindowState', 'maximized');
subplot(2,4,1:2)
ax = gca;
hold(ax, 'on');
ax.FontSize = tickLabelFontSize;
ax.TickLabelInterpreter = 'latex';

% Plot correlation patch
patch_ss = patch([0 corr_len_time corr_len_time 0], [-1 -1 1 1], [0.9 0.9 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.5); 

if plot_waveform
    plot(time_vec3, olsa_audio{s_idx1}, 'Color', [color1, 0.05], 'LineWidth', 1);
    plot(time_vec4, olsa_audio{s_idx2}, 'Color', [color2, 0.05], 'LineWidth', 1);
end

% Plot envelopes
p1 = plot(time_vec1, envelope1, 'Color', color1, 'LineWidth', 2);
p2 = plot(time_vec2, envelope2, 'Color', color2, 'LineWidth', 2);

% Include textbox with correlation value
r_symbol = sprintf('r_{%s}', corr_metric);
ptext    = {sprintf('$%s \\approx %.2f$', r_symbol, distr4plot(s_idx2))};
text(0.02, 0.02, ptext, ...
         'Units', 'normalized', ... 
         'VerticalAlignment', 'bottom', ...
         'HorizontalAlignment', 'left', ...
         'BackgroundColor', 'white', ...
         'EdgeColor', 'black', ...
         'Interpreter', 'latex', ...
         'FontSize', textBoxFontSize, ...
         'Color', 'k');

xlabel('Time / s', 'FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex'); 
ylabel('Normalized amplitude', 'FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
legend([p1, p2, patch_ss], ...
       {sprintf('%s.wav (%s)', fnames_audio{s_idx1}, regexprep(envelopes_sentence{s_idx1}(1:end-1), {'ä', 'ö', 'ü', 'ß'}, {'\\"a', '\\"o', '\\"u', '{\\ss}'})), ...
        sprintf('%s.wav (%s)', fnames_audio{s_idx2}, regexprep(envelopes_sentence{s_idx2}(1:end-1), {'ä', 'ö', 'ü', 'ß'}, {'\\"a', '\\"o', '\\"u', '{\\ss}'})), ...
        'Correlation window'}, ...
       'Interpreter', 'latex', 'Location', 'southeast', 'FontSize', legendFontSize);
hold(ax, 'off');
grid on; 
grid minor;
box on;
ylim([-1,1]);
% axis square;

% 2.) Plot histrogram and mean 
%--------------------------------------------------------------------------
% Select number of bins
% n_bins = ceil((max(distr4plot)-min(distr4plot))/0.05);
n_bins = ceil(sqrt(numel(distr4plot)));

figure('Name', 'Plot Sentence Distribution', 'Color', 'w', 'WindowState', 'maximized');
subplot(2,4,3)
ax = gca;
hold(ax, 'on');
ax.FontSize = tickLabelFontSize;
ax.TickLabelInterpreter = 'latex';
% histogram(distr4plot, 'BinWidth', 0.05);
histogram(distr4plot, n_bins);
p1 = xline(mean4plot, 'Color', color1, 'LineStyle', '-', 'LineWidth', 4, 'DisplayName', sprintf('$\\rho_{%s}: %.2f$', metric_label, mean4plot));
xlabel(sprintf('$r_{%s}$', corr_metric), 'FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
ylabel('Sentence count','FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
grid on;
grid minor;
box on;
legend(p1, 'Location', 'northwest', 'FontSize', legendFontSize, 'Interpreter', 'latex');
hold(ax, 'off');
axis square;

% 3.) Plot sorted marginalized distributions
%--------------------------------------------------------------------------
valid_idx                      = true(size(sim_vals_sentcs_marg_sorted));
valid_idx(sort_idx1 == s_idx1) = false; % exclude selected sentence

figure('Name', 'Plot Sentence Similarity ranking', 'Color', 'w', 'WindowState', 'maximized');
subplot(2,4,4)
ax = gca;
hold(ax, 'on');
ax.FontSize = tickLabelFontSize;
ax.TickLabelInterpreter = 'latex';
plot(find(valid_idx), sim_vals_sentcs_marg_sorted(valid_idx), 'x', 'Color', [0.2 0.4 0.6], 'LineWidth', 2, 'MarkerSize', markerSize);
p1 = plot(find(~valid_idx), sim_vals_sentcs_marg_sorted(~valid_idx), ...
    'x', 'Color', color1, 'LineWidth', 3, 'MarkerSize', markerSize*1.5, 'DisplayName', ...
    regexprep(envelopes_sentence{s_idx1}, {'ä', 'ö', 'ü', 'ß'}, {'\\"a', '\\"o', '\\"u', '{\\ss}'}));
xlabel('Ranked sentences', 'FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
ylabel(sprintf('$\\rho_{%s}$', metric_label), 'FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
grid on;
grid minor;
box on;
legend(p1, 'Location', 'southwest', 'FontSize', legendFontSize, 'Interpreter', 'latex');
hold(ax, 'off');
axis square;

clear color1 color2

%% Visualize mean-sentence envelope with std
%--------------------------------------------------------------------------
% Description:
%   This block visualizes the grand average acoustic envelope of the dataset 
%   and computes its temporal autocorrelation to identify dominant rhythmic structures.
%
%   Key Processing & Visualization Steps:
%   1. Mean Envelope Plot: Plots the normalized grand average envelope over time, 
%      overlaid with a shaded region representing ±1 Standard Deviation to 
%      illustrate variance across the sentence dataset.
%   2. Autocorrelation Function: Computes and plots the normalized cross-correlation 
%      of the mean envelope with itself (xcorr). This time-domain analysis 
%      highlights fundamental periodicities in the speech signal (e.g., inherent 
%      syllable or word pacing).
%--------------------------------------------------------------------------
% Define font sizes for plot elements
axisLabelFontSize = 35; 
tickLabelFontSize = 35; 
legendFontSize    = 35; 

% Normalize for plot
norm_fac = max(abs(envelope_avg));
time_vec = (0:length(envelope_avg)-1) / fs;

% Limits
data2plot = ([envelope_avg(:)' - envelope_avg_std(:)', fliplr(envelope_avg(:)' + envelope_avg_std(:)')])/norm_fac;
y_limits  = [min(data2plot), max(data2plot)];

figure('Name', 'Mean-envelope', 'Color', 'w', 'WindowState', 'maximized');
ax = gca;
hold(ax, 'on');
ax.FontSize = tickLabelFontSize;
ax.TickLabelInterpreter = 'latex';

h_std = fill([time_vec(:)', fliplr(time_vec(:)')], ...
             data2plot, ...
             [0.5 0.5 0.5], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
h_mean = plot(time_vec, envelope_avg/norm_fac, 'LineWidth', 5, 'Color', 'k'); % Capture handle

xlabel('Time / s', 'FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex'); 
ylabel('Normalized amplitude', 'FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
ylim(y_limits);

% Updated Legend configuration
%------------------------------
% Append your new handles (h_mean, h_bg) to the existing handles array
legend([h_mean, h_std], ...
       {'Mean-Sentence Envelope', ...
        '$\pm 1$ Std. Dev.'}, ...
        'Interpreter', 'latex', 'Location', 'northeast', 'FontSize', legendFontSize);
hold(ax, 'off');
grid on; 
grid minor;
box on;
% ylim([-1,1]);
% axis square;

clear time_vec norm_fac h_mean h_std data2plot y_limits

% Compute auto-correlation (zero-padding)
%----------------------------------------
[r_auto_corr, lags_auto_corr] = xcorr(envelope_avg, envelope_avg, 'normalized');  
lags_auto_corr_time           = lags_auto_corr / fs; 

figure('Name','Auto-correlation function', 'Color', 'w');
plot(lags_auto_corr_time*1000, r_auto_corr, 'x', 'Color', [0.2 0.4 0.6], 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Lag / ms');
ylabel('Correlation Coefficient (r)');
title(sprintf('Auto-Correlation (Mean-Sentence)'));
grid on;
xlim([0 max(lags_auto_corr_time*1000)]); % Ensure plot starts at zero

clear r_auto_corr lags_auto_corr lags_auto_corr_time 

%% Compute mean/median correlation with time-shift
%--------------------------------------------------------------------------
% Description:
%   This section calculates the pairwise acoustic similarity of sentence 
%   envelopes across various temporal lags to identify underlying rhythmic 
%   periodicities (e.g., word and syllable rates).
%
%   Key Processing Steps:
%   1. Pair Generation: Creates a matrix of all unique sentence pairs, 
%      excluding auto-correlations to prevent artificial zero-lag spikes.
%   2. Circular Cross-Correlation: Computes time-shifted correlations using 
%      optimized frequency-domain (FFT) or time-domain methods. Uses 
%      `fftshift` to perfectly align Lag 0 across pairs of varying lengths.
%   3. Temporal Cropping: Restricts the computed correlation lags to match 
%      the exact temporal window used in the neural tracking pipeline 
%      (intersecting with `shift_samples`).
%   4. Marginalized Statistics: Computes the grand average and median cross-
%      correlation across all aligned and cropped sentence pairs.
%   5. Spectral Validation: Plots the time-domain circular cross-correlation 
%      and computes its frequency-domain power spectrum to reveal dominant Hz peaks.
%   6. Linear Cross-Correlation: Computes a traditional, zero-padded `xcorr` 
%      across all pairs as a baseline comparison against the circular method.
%
%--------------------------------------------------------------------------

% Select computation method
%--------------------------
comp_method      = 'fft'; % mtrf, matlab, mult, fft
shift_cross_corr = true; % shifts cross corr to -shift,0,+shift (center-aligned)

% Pair Generation & Filtering: 
% Generates all possible combinations of audio pairs and efficiently filters 
% the matrix to retain only unique, non-reversed cross-pairs (e.g., keeping [1, 2] 
% while discarding auto-correlations like [1, 1] and redundant reversed pairs like [2, 1]). 
% This mathematically isolates the true cross-comparisons and cuts the 
% required computation time in half.

[X, Y]    = ndgrid(1:n_audio, 1:n_audio);
all_pairs = [X(:), Y(:)];
% all_pairs = all_pairs(all_pairs(:,1) <= all_pairs(:,2), :); % inlude comparison with same sentence
% all_pairs = all_pairs(all_pairs(:,1) < all_pairs(:,2), :); 
all_pairs = all_pairs(all_pairs(:,1) ~= all_pairs(:,2), :); % exclude auto-corr
n_pairs   = size(all_pairs, 1); % include comparison with same sentence and reverse pairs

% all_pairs = fliplr(all_pairs); % Investigate order of pairs

% Calculate lengths
min_lengths   = min(lengths_envelopes(all_pairs(:,1)),lengths_envelopes(all_pairs(:,2)));
max_length    = max(min_lengths);
global_center = floor(max_length / 2) + 1;

% Define the interval for progress updates (every 10%)
n_loops     = sum(min_lengths);
update_step = max(1, floor(n_loops / 10));

% Init data
%----------
audio_stats.('rmse').sim_vals_sentcs_shift = nan(n_pairs, max_length);
audio_stats.('corr').sim_vals_sentcs_shift = nan(n_pairs, max_length);

% Compute Correlations
counter = 0;

% Loop over shifts
for p_idx = 1:n_pairs

    % Get correct envelopes for correlation
    idx2  = all_pairs(p_idx, 1);
    idx1  = all_pairs(p_idx, 2);
    n_min = min_lengths(p_idx);

    % Select and crop
    envelope1 = envelopes_cropped{idx1}(1:n_min);
    envelope2 = envelopes_cropped{idx2}(1:n_min);

    % Pre-compute normalized vectors for FFT and Mult 
    if ismember(comp_method, {'fft', 'mult'})
        switch corr_metric
            case 'Spearman'
                s1 = tiedrank(envelope1) - mean(tiedrank(envelope1));
                s2 = tiedrank(envelope2) - mean(tiedrank(envelope2));
            case 'Pearson'
                s1 = envelope1 - mean(envelope1);
                s2 = envelope2 - mean(envelope2);
        end
    end

    if strcmp(comp_method, 'fft')
        cross_corr = ifft(fft(s1) .* conj(fft(s2))) ./ n_min;
        audio_stats.('corr').sim_vals_sentcs_shift(p_idx, 1:n_min) = cross_corr;
    end

    % figure
    % plot(cross_corr)
    % plot(fftshift(cross_corr))

    shift2compute = 0:min_lengths(p_idx)-1;

    for s_idx = 1:n_min
        shift = shift2compute(s_idx);
        
        switch comp_method
            case 'mtrf'     
                % mTRF-toolbox
                [r, err] = mTRFevaluate(envelope1, circshift(envelope2, shift), 'dim', 2, 'corr', corr_metric, 'error', 'mse');
                rmse     = sqrt(err);

                audio_stats.('corr').sim_vals_sentcs_shift(p_idx, s_idx) = apply_fisher_z_transform(r);
                audio_stats.('rmse').sim_vals_sentcs_shift(p_idx, s_idx) = rmse;

            case 'matlab' 
                % 'matlab'
                r    = corr(envelope1', circshift(envelope2, shift)', 'Type', corr_metric);
                rmse = sqrt(mean((envelope1-circshift(envelope2, shift)).^2));

                audio_stats.('corr').sim_vals_sentcs_shift(p_idx, s_idx) = apply_fisher_z_transform(r);
                audio_stats.('rmse').sim_vals_sentcs_shift(p_idx, s_idx) = rmse;

            case 'mult'
                % Simple multiplicatio; afterwards normalized
                r    = sum(s1 .* circshift(s2, shift))./min_lengths(p_idx);
                rmse = sqrt(mean((envelope1-circshift(envelope2, shift)).^2));

                audio_stats.('corr').sim_vals_sentcs_shift(p_idx, s_idx) = r;
                audio_stats.('rmse').sim_vals_sentcs_shift(p_idx, s_idx) = rmse; 

            case 'fft'
                rmse = sqrt(mean((envelope1-circshift(envelope2, shift)).^2));
                audio_stats.('rmse').sim_vals_sentcs_shift(p_idx, s_idx) = rmse;
        end

        counter = counter + 1;

        if mod(counter, update_step) == 0 || counter == n_loops
            fprintf('Processed %d / %d correlations.\n', counter, n_loops);
        end
        
    end % shifts

    % Apply normalization
    %--------------------
    if ismember(comp_method, {'fft', 'mult'})
        % Normalize
        cross_corr = audio_stats.('corr').sim_vals_sentcs_shift(p_idx, :);
        
        switch corr_metric
            case 'Spearman'
                norm_fac = std(tiedrank(envelope1), 1) * std(tiedrank(envelope2), 1);
            case 'Pearson'
                norm_fac = std(envelope1, 1)*std(envelope2, 1);
        end
       
        audio_stats.('corr').sim_vals_sentcs_shift(p_idx, :) = apply_fisher_z_transform(cross_corr./norm_fac);
         
    end

    % Apply shift correction
    %-----------------------
    if shift_cross_corr
        local_center = floor(n_min / 2) + 1;
        offset       = global_center - local_center;

        % Align cross correlation
        cross_corr = audio_stats.('corr').sim_vals_sentcs_shift(p_idx, 1:n_min);
        cross_corr = fftshift(cross_corr);

        audio_stats.('corr').sim_vals_sentcs_shift(p_idx, :) = nan; % reset data
        audio_stats.('corr').sim_vals_sentcs_shift(p_idx, (1:n_min) + offset) = cross_corr;

        % Align rmse
        rmse_vals = audio_stats.('rmse').sim_vals_sentcs_shift(p_idx, 1:n_min);
        rmse_vals = fftshift(rmse_vals);
        
        audio_stats.('rmse').sim_vals_sentcs_shift(p_idx, :) = nan; % reset data
        audio_stats.('rmse').sim_vals_sentcs_shift(p_idx, (1:n_min) + offset) = rmse_vals;
    end

end % pairs

clear idx1 idx2 envelope1 envelope2 n_min r err rmse cross_corr rmse_vals
fprintf('Time shift similarity for audio computed\n');

% Add marginalized distribution
%------------------------------
if shift_cross_corr
    shift_vector = (1:max_length) - global_center;
else
    shift_vector = 0 : (max_length - 1);
    % shift_vector = 0 : (n_shifts - 1); % optional case
end

% Optional: Cut off outer skirts (keep maximal shift shared across all envelope pairs)
%-------------------------------------------------------------------------------------
[shift_vector, idx_keep, ~] = intersect(shift_vector, shift_samples);

audio_stats.('corr').sim_vals_sentcs_shift = audio_stats.('corr').sim_vals_sentcs_shift(:, idx_keep);
audio_stats.('rmse').sim_vals_sentcs_shift = audio_stats.('rmse').sim_vals_sentcs_shift(:, idx_keep);
clear idx_keep
%-------------------------------------------------------------------------------------
% Append shift vector
audio_stats.('corr').sim_vals_sentcs_shift_samples = shift_vector;
audio_stats.('rmse').sim_vals_sentcs_shift_samples = shift_vector;

sim_vals_sentcs_shift                             = audio_stats.('corr').sim_vals_sentcs_shift;
audio_stats.('corr').mean_sim_vals_sentcs_shift   = mean(sim_vals_sentcs_shift, 1, 'omitnan');
audio_stats.('corr').median_sim_vals_sentcs_shift = median(sim_vals_sentcs_shift, 1, 'omitnan');
sim_vals_sentcs_shift                             = audio_stats.('rmse').sim_vals_sentcs_shift;
audio_stats.('rmse').mean_sim_vals_sentcs_shift   = mean(sim_vals_sentcs_shift, 1, 'omitnan');
audio_stats.('rmse').median_sim_vals_sentcs_shift = median(sim_vals_sentcs_shift, 1, 'omitnan');
clear sim_vals_sentcs_shift

% Example plots - Cross Correlation and Spectrum
%-----------------------------------------------
cross_corr = apply_inverse_fisher_z_transform(audio_stats.('corr').mean_sim_vals_sentcs_shift);

n_fft             = 2^nextpow2(length(cross_corr));
[P1, f_axis]      = compute_spectrum(cross_corr, n_fft, fs, 'Detrend', true, 'Window', 'tukey');
shift_vector_time = audio_stats.('corr').sim_vals_sentcs_shift_samples*1000/fs;

figure
subplot(1,2,1)
hold on; 
plot(shift_vector_time, cross_corr, 'b')
% plot(shift_vector_time, cross_corr.*hanning(length(cross_corr))', 'r')
plot(shift_vector_time, cross_corr.*tukeywin(length(cross_corr), 0.1)', 'r')
xlabel('Circular Shift / ms')
ylabel('Cross Correlation Function')
grid on;
grid minor; 
box on;
axis square;
if shift_cross_corr
    xline(0, '--k', 'LineWidth', 1);
end

subplot(1,2,2)
plot(f_axis, 20*log10(P1))
xlabel('f / Hz')
ylabel('Power Spectrum')
xlim([0,10])
grid on;
grid minor; 
box on;
axis square;

sgtitle(sprintf('comp-method: %s / shift-cross-corr: %d', comp_method, shift_cross_corr))

clear shift_vector shift_vector_time f_axis P1 n_fft cross_corr

% Compute linear cross-correlations between Olsa sentences
%--------------------------------------------------------------------------
max_len              = max(lengths_envelopes);
envelopes_cross_corr = cell(n_pairs, 1);

% Loop over valid pairs
for p_idx = 1:n_pairs

    % Get correct envelopes for correlation
    idx1 = all_pairs(p_idx, 1);
    idx2 = all_pairs(p_idx, 2);

    envelope1 = [envelopes_cropped{idx1}, zeros(1, max_len - lengths_envelopes(idx1))];
    envelope2 = [envelopes_cropped{idx2}, zeros(1, max_len - lengths_envelopes(idx2))];

    [r_crosscorr, lags_cross_corr] = xcorr(envelope1 - mean(envelope1), envelope2 - mean(envelope2), 'normalized');   
    envelopes_cross_corr{p_idx}    = r_crosscorr;
    
end
clear envelope1 envelope2 r_crosscorr

% Convert cell to matrix and compute mean across all sentence pairs
% We must ensure we align them by lags if lengths vary
envelopes_cross_corr        = cell2mat(envelopes_cross_corr); 
envelopes_cross_corr        = apply_fisher_z_transform(envelopes_cross_corr);
envelopes_cross_corr_mean   = mean(envelopes_cross_corr, 1);
envelopes_cross_corr_median = median(envelopes_cross_corr, 1);
lags_cross_corr_time        = lags_cross_corr / fs; 

audio_stats.('corr').sim_vals_sentcs_shift_crosscorr_mean   = envelopes_cross_corr_mean;
audio_stats.('corr').sim_vals_sentcs_shift_crosscorr_median = envelopes_cross_corr_median;
fprintf('Linear cross-correlation computed.\n');

% Visualization
%--------------
envelopes_cross_corr2plot = apply_inverse_fisher_z_transform(audio_stats.('corr').sim_vals_sentcs_shift_crosscorr_mean); % mean
% envelopes_cross_corr2plot = apply_inverse_fisher_z_transform(audio_stats.('corr').sim_vals_sentcs_shift_crosscorr_median); % median

figure('Name','Cross-correlation function', 'Color', 'w');
plot(lags_cross_corr_time*1000, envelopes_cross_corr2plot, 'x', 'Color', [0.2 0.4 0.6], 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Lag / ms');
ylabel('Correlation Coefficient (r)');
title(sprintf('Cross-Correlation Similarity (Zero-Padded)'));
grid on;
xlim([0 max(lags_cross_corr_time*1000)]); % Ensure plot starts at zero

clear envelopes_cross_corr envelopes_cross_corr2plot envelopes_cross_corr_mean envelopes_cross_corr_median lags_cross_corr_time

%% Check for normality
%--------------------------------------------------------------------------
% Evaluates if the correlation and RMSE distributions at each time shift 
% are normally distributed using the Shapiro-Wilk test (swtest). 
% Flags distributions as 1 (normal) if the null hypothesis is not rejected.
% Note: At N=~9900, this test has excessive statistical power and becomes 
% hypersensitive, failing the data for microscopic, meaningless deviations 
% from a perfect bell curve. 

n_shifts2compute = length(audio_stats.('corr').sim_vals_sentcs_shift_samples);

check_sim_vals_shift_normality_corr = nan(n_shifts2compute, 1);
check_sim_vals_shift_normality_rmse = nan(n_shifts2compute, 1);

for s_idx = 1:n_shifts2compute
    sim_vals_sentcs_shift_distr = audio_stats.('corr').sim_vals_sentcs_shift(:, s_idx);
    if sum(~isnan(sim_vals_sentcs_shift_distr))>2 % shifts lag must have availabe samples
        [h, ~, ~]                                  = swtest(sim_vals_sentcs_shift_distr(~isnan(sim_vals_sentcs_shift_distr)));
        check_sim_vals_shift_normality_corr(s_idx) = (h == 0); 
        clear h
    end

    sim_vals_sentcs_shift_distr = audio_stats.('rmse').sim_vals_sentcs_shift(:, s_idx);
    if sum(~isnan(sim_vals_sentcs_shift_distr))>2
        [h, ~, ~]                                  = swtest(sim_vals_sentcs_shift_distr(~isnan(sim_vals_sentcs_shift_distr)));
        check_sim_vals_shift_normality_rmse(s_idx) = (h == 0); 
        clear h   
    end
end

audio_stats.('corr').check_sim_vals_shift_normality = check_sim_vals_shift_normality_corr;
audio_stats.('rmse').check_sim_vals_shift_normality = check_sim_vals_shift_normality_rmse;
clear sim_vals_sentcs_shift_distr h check_sim_vals_shift_normality_corr check_sim_vals_shift_normality_rmse n_shifts2compute

%% Alternative: Pooled Global Correlation (Neural-Aligned Baseline)
%--------------------------------------------------------------------------
% This section computes a single, global cross-correlation and error metric 
% per shift by concatenating all sentence pairs into one massive continuous 
% array. Crucially, the circular shift is applied safely within each individual 
% sentence before concatenation, preventing artificial boundary-leakage between 
% adjacent sentences. This pooling method evaluates the global variance of 
% the entire dataset at once, mirroring the statistical evaluation of neural 
% decoding models for an exact "apples-to-apples" baseline comparison.
%
% OPTIMIZATION NOTE: 
% The current loop explicitly computes the correlation for every single shift. 
% However, computation time could be cut in half IF the 'all_pairs' matrix 
% was generated to include both permutations of every sentence pair 
% (e.g., including both [1,2] and [2,1]). Under that specific condition, 
% the resulting cross-correlation function is perfectly symmetric around 
% Lag 0. One could simply compute Lag 0 and the positive shifts, and mirror 
% the array to reconstruct the negative lags. The full iterative loop is 
% retained here to ensure the script remains robust and mathematically 
% accurate regardless of how 'all_pairs' is filtered upstream (e.g., if 
% reverse pairs were excluded to save memory).
%--------------------------------------------------------------------------

% Apply linear or circular shift
apply_shift_type = 'circshift';
% apply_shift_type = 'linshift';

shifts2compute   = shift_samples;
n_shifts2compute = length(shifts2compute);

audio_stats.('corr').nt_sim_vals_sentcs_shift = nan(1, n_shifts2compute);
audio_stats.('rmse').nt_sim_vals_sentcs_shift = nan(1, n_shifts2compute);

% concat envelopes of all pairs
envelopes1 = envelopes_cropped(all_pairs(:,1));
envelopes2 = envelopes_cropped(all_pairs(:,2));

% Concatenate sentences and apply minimum duration
envelopes1 = cellfun(@(x,y) x(:,1:y), envelopes1, num2cell(min_lengths), 'UniformOutput', false);
envelopes2 = cellfun(@(x,y) x(:,1:y), envelopes2, num2cell(min_lengths), 'UniformOutput', false);

envelopes1_concat = [envelopes1{:}];
for s_idx = 1:n_shifts2compute
    shift = shifts2compute(s_idx);

    % Apply linear shift (in samples) with zero-padding
    if strcmp(apply_shift_type, 'linshift')
        envelopes_shifted = cell(size(envelopes2));
        for c_idx = 1:length(envelopes2)
            env         = envelopes2{c_idx};
            env_shifted = zeros(size(env)); % Pre-fill with zeros
            
            n_samples = size(env, 2);
            
            if shift > 0 && shift < n_samples
                % Shift right (delay), pad beginning with zeros
                env_shifted(:, shift+1:end) = env(:, 1:end-shift);
            elseif shift < 0 && abs(shift) < n_samples
                % Shift left (advance), pad end with zeros
                env_shifted(:, 1:end+shift) = env(:, 1-shift:end);
            elseif shift == 0
                % No shift
                env_shifted = env;
            end
            % Note: If abs(shift) >= n_samples, it safely remains all zeros
            
            envelopes_shifted{c_idx} = env_shifted;
        end
    end

    % Apply circshift (in samples)
    if strcmp(apply_shift_type, 'circshift')
        envelopes_shifted = cellfun(@(x) circshift(x, shift), envelopes2, 'UniformOutput', false);
    end

    envelopes2_concat = [envelopes_shifted{:}];
    [r, err] = mTRFevaluate(envelopes1_concat, ...
                            envelopes2_concat,...
                            'error', 'mse', ...
                            'dim', 2,...
                            'corr', corr_metric);

    rmse = sqrt(err);

    audio_stats.('corr').nt_sim_vals_sentcs_shift(s_idx) = apply_fisher_z_transform(r);
    audio_stats.('rmse').nt_sim_vals_sentcs_shift(s_idx) = rmse;

    if mod(s_idx, ceil(n_shifts2compute/10)) == 0 || s_idx == n_shifts2compute 
            fprintf('Processed %d / %d shifts.\n', s_idx, n_shifts2compute);
    end

end

% Append shift vector
audio_stats.('corr').nt_sim_vals_sentcs_shift_samples = shifts2compute;
audio_stats.('rmse').nt_sim_vals_sentcs_shift_samples = shifts2compute;

shift_vector_time = audio_stats.('corr').nt_sim_vals_sentcs_shift_samples*1000/fs;

% Visualize Results
%------------------
cross_corr   = apply_inverse_fisher_z_transform(audio_stats.('corr').nt_sim_vals_sentcs_shift);
n_fft        = 2^nextpow2(length(cross_corr));
[P1, f_axis] = compute_spectrum(cross_corr, n_fft, fs, 'Detrend', true, 'Window', 'tukey');

figure
subplot(1,2,1)
hold on; 
plot(shift_vector_time, cross_corr, 'b')
% plot(shift_vector_time, cross_corr.*hanning(length(cross_corr))', 'r')
plot(shift_vector_time, cross_corr.*tukeywin(length(cross_corr), 0.1)', 'r')
xlabel('Circular Shift / ms')
ylabel('Cross Correlation Function')
grid on;
grid minor; 
box on;
axis square;
if shift_cross_corr
    xline(0, '--k', 'LineWidth', 1);
end

subplot(1,2,2)
plot(f_axis, 20*log10(P1))
xlabel('f / Hz')
ylabel('Power Spectrum')
xlim([0,10])
grid on;
grid minor; 
box on;
axis square;

sgtitle(sprintf('Concatenated Envelopes (%s)', apply_shift_type))

clear envelopes1 envelopes2 envelopes_shifted envelopes1_concat envelopes2_concat r err rmse shifts2compute n_shifts2compute cross_corr
clear n_fft P1 f_axis

%% Compute Modulation spectra
%--------------------------------------------------------------------------

% Select options for fft
option_detrend = true;
option_window  = 'tukey'; % 'none', 'hanning', 'tukey'

% Select circular envelope type for computation
% circ_shift_type = 'average'; % mean envelope cross correlation
circ_shift_type = 'concat'; % concatenated cross correlation similar to neural tracking analysis

% Select way for cross spectrum computation
% cross_spec_type = 'wiener-khinchin'; % no padding, no window -> simple resolution
cross_spec_type = 'custom'; % using custom code (padding, window) -> higher resolution

% Select condition for neural tracking
conditions2plot = 'ses-pooled';

% Apply 1/f noise correction for original envelopes
apply_noise_correction = false;

% Compute modulation spectra for sentences
%--------------------------------------------------------------------------
% Cropped envelopes
n_fft       = 2^nextpow2(max(cellfun(@length, envelopes_cropped)));
n_freq_bins = floor(n_fft / 2) + 1;

mod_spec                          = struct();
mod_spec.('envelope').n_fft       = n_fft;
mod_spec.('envelope').all_spectra = zeros(n_freq_bins, n_audio);

% Original envelopes
[b, a]      = butter(4, 50 / (fs_orig / 2), 'low');
n_fft       = 2^nextpow2(min(cellfun(@length, olsa_audio)));
n_freq_bins = floor(n_fft / 2) + 1;

mod_spec.('envelope_orig').n_fft       = n_fft;
mod_spec.('envelope_orig').all_spectra = zeros(n_freq_bins, n_audio);

% Computations
%-------------
for s_idx = 1:n_audio

    % Cropped envelopes
    %------------------
    envelope     = envelopes_cropped{s_idx};
    [P1, f_axis] = compute_spectrum(envelope, mod_spec.('envelope').n_fft, fs, ...
                                    'Detrend', option_detrend, ...
                                    'Window', option_window);
   
    mod_spec.('envelope').all_spectra(:, s_idx) = P1.^2;
    if s_idx == 1
        mod_spec.('envelope').freq_axis = f_axis;
    end

    % Modulation spectrum of raw audio from scratch
    %----------------------------------------------
    % Compute broadband envelope of raw audio
    envelope = abs(hilbert(olsa_audio{s_idx}));
    envelope = filtfilt(b, a, envelope); % lowpass
    
    [P1, f_axis] = compute_spectrum(envelope, mod_spec.('envelope_orig').n_fft, fs_orig, ...
                                    'Detrend', option_detrend, ...
                                    'Window', option_window);

    mod_spec.('envelope_orig').all_spectra(:, s_idx) = P1.^2';
    if s_idx == 1
        mod_spec.('envelope_orig').freq_axis = f_axis';
    end

end
clear envelope P1 f_axis n_fft n_freq_bins b a

% Add average spectra
mod_spec.('envelope').avg_spectra      = mean(mod_spec.('envelope').all_spectra, 2);
mod_spec.('envelope_orig').avg_spectra = mean(mod_spec.('envelope_orig').all_spectra, 2);

if  apply_noise_correction 
    % Apply optional 1/f correction for power spectra (unfiltered version)
    %--------------------------------------------------------------------------
    % Peelle, Jonathan E., Joachim Gross, and Matthew H. Davis. 
    % "Phase-locked responses to speech in human auditory cortex are enhanced during comprehension." 
    % Cerebral cortex 23.6 (2013): 1378-1387.
    
    f_axis   = mod_spec.('envelope_orig').freq_axis;
    avg_spec = mod_spec.('envelope_orig').avg_spectra;
    
    % Define fitting range
    fit_idx = (f_axis > 5) & (f_axis < 40); 
    
    % Fit the model to the average spectrum
    p = polyfit(log10(f_axis(fit_idx)), log10(avg_spec(fit_idx)), 1);
    
    % % Fixed slope instead
    % log_f       = log10(f_axis(fit_idx));
    % log_spec    = log10(avg_spec(fit_idx));
    % fixed_slope = -1;
    % 
    % % Fit a 0-degree polynomial (just the intercept) to the adjusted data
    % p_intercept = polyfit(log_f, log_spec - (fixed_slope * log_f), 0);
    % % Reconstruct the standard polynomial vector [slope, intercept]
    % p = [fixed_slope, p_intercept];
    
    % Generate the 1/f noise model for the entire frequency axis
    noise_model = 10^p(2) .* f_axis.^p(1);
    
    % Handle the DC component (0 Hz) to prevent division by zero/Inf issues
    if f_axis(1) == 0
        noise_model(1) = inf; 
    end
    
    % Show original spectrum with fit
    %--------------------------------
    figure 
    % log-space
    subplot(1,2,1)
    hold on;
    plot(log10(f_axis), log10(avg_spec), 'b') % Original spectrum 
    plot(log10(f_axis(fit_idx)), polyval(p, log10(f_axis(fit_idx))), 'r', 'LineWidth', 2) % Add fit
    hold off;
    axis square
    legend({'Original spectrum', 'Fitted noise'})
    title('log-space')
    % Normal frequency axis
    subplot(1,2,2)
    hold on;
    plot(f_axis(fit_idx), avg_spec(fit_idx), 'b') % Mark fit range
    plot(f_axis, noise_model, 'r') % Add noise model
    hold off;
    axis square
    legend({'Original spectrum', 'Fitted noise'})
    title('Regular frequency axis')
    xlim([5,40])
    
    % Apply the correction
    %---------------------
    mod_spec.('envelope_orig').avg_spectra = avg_spec ./ noise_model; % Subtraction in log-space
    mod_spec.('envelope_orig').all_spectra = mod_spec.('envelope_orig').all_spectra ./ noise_model;
    clear f_axis avg_spec fit_idx noise_model p
end

% Advanced method?!
% -----------------
% num_sentences_to_concat = 2;
% apply_lowpass           = true;
% 
% [spectra_matrix, f_axis] = compute_olsa_modulation_spectra(olsa_audio, fs_orig, num_sentences_to_concat, apply_lowpass, ...
%                            'fs_new', 441, ...
%                            'Detrend', true, ...
%                            'Window', 'tukey');
% 
% figure
% plot(f_axis, 10*log10(mean(spectra_matrix, 2).^2))
% title(num2str(num_sentences_to_concat))
% xlim([0,10])
% grid on;
% grid minor;
% box on;
% axis square;
% clear num_sentences_to_concat apply_lowpass spectra_matrix f_axis

% Compute Spectra for Neural Tracking and Sentence Similarity
%--------------------------------------------------------------------------
con_idx = find(ismember(conditions, conditions2plot));

% Neural Tracking
%----------------
sim_stats                                = squeeze(stats_neuro.('shift').('mixed_model').slope(con_idx, :, :));
n_fft                                    = 2^nextpow2(size(sim_stats, 2));
mod_spec.('neural_tracking').n_fft       = n_fft;
n_freq_bins                              = floor(n_fft / 2) + 1;
mod_spec.('neural_tracking').all_spectra = zeros(n_freq_bins, n_sens);

% Computations
for sens_idx = 1:n_sens
    [P1, f_axis] = compute_spectrum(sim_stats(sens_idx, :), mod_spec.('neural_tracking').n_fft, fs, ...
                                    'Detrend', option_detrend, ...
                                    'Window', option_window);

    mod_spec.('neural_tracking').all_spectra(:, sens_idx) = P1(:).^2';
    if sens_idx == 1
        mod_spec.('neural_tracking').freq_axis = f_axis';
    end

end

% Add average spectra
mod_spec.('neural_tracking').avg_spectra = mean(mod_spec.('neural_tracking').all_spectra, 2);

if false
    % Show original shifts
    figure
    hold on;
    for sens_idx = 1:n_sens
        plot(shift_samples*1000/fs, sim_stats(sens_idx, :), 'LineWidth', 1, 'DisplayName', sensor_labels{sens_idx})
        xlabel('Circular Shift / ms')
        ylabel('Fixed-Effects Slope')
        grid on;
        grid minor; 
        box on;
    end
    plot(shift_samples*1000/fs, mean(sim_stats, 1), 'Color', 'k', 'LineWidth', 2.5, 'DisplayName', 'Average')
    legend;
    hold off;

    % Show example spectra
    figure
    plot(f_axis, 20*log10(mod_spec.('neural_tracking').avg_spectra))
    xlabel('f / Hz')
    ylabel('Power Spectrum')
    title(sprintf('Neural Tracking Data'))
    xlim([0,10])
    grid on;
    grid minor; 
    box on;
end

clear P1 f_axis n_fft n_freq_bins sim_stats 

% Circular Shift
%---------------
switch circ_shift_type
    case 'concat'
        sim_vals_sentcs_shift = apply_inverse_fisher_z_transform(audio_stats.('corr').nt_sim_vals_sentcs_shift);
    case 'average'
        sim_vals_sentcs_shift = apply_inverse_fisher_z_transform(audio_stats.('corr').mean_sim_vals_sentcs_shift);
end

mod_spec.('acoustic_circ').n_fft = 2^nextpow2(length(sim_vals_sentcs_shift));

[P1, f_axis] = compute_spectrum(sim_vals_sentcs_shift, mod_spec.('acoustic_circ').n_fft, fs, ...
                                'Detrend', option_detrend, ...
                                'Window', option_window);

mod_spec.('acoustic_circ').spectra   = P1.^2';
mod_spec.('acoustic_circ').freq_axis = f_axis';

if false
    figure
    plot(f_axis, 20*log10(P1))
    xlabel('f / Hz')
    ylabel('Power Spectrum')
    title(sprintf('Circular Shift'))
    xlim([0,10])
    grid on;
    grid minor; 
    box on;
end

clear P1 f_axis sim_vals_sentcs_shift

% Cross Correlation and Cross Spectrum
%--------------------------------------------------------------------------
% Between Neural Tracking Slopes and Circular/Linear Shift
cross_spectrum_struct = struct();

% Load data
signal_nt_raw       = squeeze(stats_neuro.('shift').('mixed_model').slope(con_idx, :, :));
signal_nt_zero_samp = find(shift_samples ==0);

switch circ_shift_type
    case 'concat'
        signal_circ_raw    = apply_inverse_fisher_z_transform(audio_stats.('corr').nt_sim_vals_sentcs_shift);
        shift_samples_circ = audio_stats.('corr').nt_sim_vals_sentcs_shift_samples;
    case 'average'
        signal_circ_raw    = apply_inverse_fisher_z_transform(audio_stats.('corr').mean_sim_vals_sentcs_shift);
        shift_samples_circ = audio_stats.('corr').sim_vals_sentcs_shift_samples;
end

% Align data and crop to same length
%-----------------------------------
% It outputs the new shared X-axis, plus the indices where those values live 
% in the original arrays.
[shift_axis_aligned, idx_nt, idx_circ] = intersect(shift_samples, shift_samples_circ);

% Crop both signals instantly using the matched indices
signal_nt_raw   = signal_nt_raw(:, idx_nt);
signal_circ_raw = signal_circ_raw(1, idx_circ);
clear idx_nt idx_circ

if false
    figure;
    hold on;
    plot(shift_axis_aligned*1000/fs, signal_nt_raw(1, :), 'LineWidth', 1.5, 'DisplayName', 'Neural Tracking');
    plot(shift_axis_aligned*1000/fs, signal_circ_raw, 'LineWidth', 1.5, 'DisplayName', sprintf('Acoustic Rhythm (%s envelope correlation)',circ_shift_type));
    legend;
    title(sensor_labels{1})
end

% Circular Shift
%---------------
n_min       = size(signal_nt_raw, 2); 
n_fft       = 2^nextpow2(n_min);

switch cross_spec_type
    case 'wiener-khinchin'
        num_freq_bins = floor(n_min / 2) + 1;
        f_axis        = (0:num_freq_bins-1) * (fs/n_min);
   
    case 'custom'
        num_freq_bins = floor(n_fft / 2) + 1;
        f_axis        = (0:num_freq_bins-1) * (fs /n_fft);      
end

cross_spectrum_struct.('nt_circ_shift').n_fft           = n_fft;
cross_spectrum_struct.('nt_circ_shift').n_min           = n_min;
cross_spectrum_struct.('nt_circ_shift').cross_spec      = zeros(num_freq_bins, n_sens);
cross_spectrum_struct.('nt_circ_shift').cross_corr      = zeros(n_min, n_sens);
cross_spectrum_struct.('nt_circ_shift').freq_axis       = f_axis;
cross_spectrum_struct.('nt_circ_shift').shift_axis      = shift_axis_aligned;
cross_spectrum_struct.('nt_circ_shift').circ_shift_type = circ_shift_type; % type used during computation;

% Computation
%------------
for sens_idx = 1:n_sens
    
    % Circular Shift
    %---------------
    n_min       = cross_spectrum_struct.('nt_circ_shift').n_min;
    n_fft       = cross_spectrum_struct.('nt_circ_shift').n_fft;
    signal_nt   = signal_nt_raw(sens_idx, :);
    signal_circ = signal_circ_raw;
 
    if option_detrend
        signal_nt   = detrend(signal_nt, 'linear');
        signal_circ = detrend(signal_circ, 'linear');
    end

    % Compute Cross Correlation 
    %--------------------------
    % (Circular) Cross Correlation is computed based on Wiener-Khinchin Theorem
    cross_spectrum = (fft(signal_nt) .* conj(fft(signal_circ))) ./ n_min;
    cross_corr     = fftshift(ifft(cross_spectrum));

    % Use Pearson (just a scaling; can be ignored)
    % norm_fac   = std(signal_nt, 1)*std(signal_circ, 1);
    % cross_corr = cross_corr./norm_fac;
    % clear norm_fac

    % Compute Cross Spectrum 
    %-----------------------
    switch cross_spec_type
        case 'wiener-khinchin'
            P2             = abs(cross_spectrum);
            P1             = P2(1:num_freq_bins);
            P1(2:end-1)    = 2 * P1(2:end-1); % skipping DC at index 1 and Nyquist at the end
            
        case 'custom'
            % Apply window function
            switch option_window
            case 'hanning'
                window = hanning(n_min);
            case 'tukey'
                window = tukeywin(n_min, 0.1);
            case 'none'
                window = ones(n_min, 1);
            end

            cross_spectrum = (fft(signal_nt(:)'.*window(:)', n_fft) .* conj(fft(signal_circ(:)'.*window(:)', n_fft))) ./ n_min;
            P2             = abs(cross_spectrum);
            P1             = P2(1:num_freq_bins);
            P1(2:end-1)    = 2 * P1(2:end-1);
    end

    % Store
    cross_spectrum_struct.('nt_circ_shift').cross_corr(:, sens_idx) = cross_corr(:);
    cross_spectrum_struct.('nt_circ_shift').cross_spec(:, sens_idx) = P1(:);

    % Visualize results (always false; check manually)
    if false
        figure

        subplot(1,2,1)
        hold on; 
        plot(shift_axis_aligned*1000/fs, cross_corr, 'b')
        xlabel('Circular Shift / ms')
        ylabel('Cross Correlation Function')
        title(sprintf('circ-shift-type: %s envelope correlation', circ_shift_type))
        grid on;
        grid minor; 
        box on;
        if shift_cross_corr
            xline(0, '--k', 'LineWidth', 1);
        end
        
        subplot(1,2,2)
        plot(f_axis, 20*log10(P1))
        xlabel('f / Hz')
        ylabel('Power Spectrum')
        title(sprintf('cross-spec-type: %s computation', cross_spec_type))
        xlim([0,10])
        grid on;
        grid minor; 
        box on;

        sgtitle(sensor_labels{sens_idx})
    end

end % Loop over sensors

% Add average spectra
cross_spectrum_struct.('nt_circ_shift').avg_cross_corr = mean(cross_spectrum_struct.('nt_circ_shift').cross_corr, 2);
cross_spectrum_struct.('nt_circ_shift').avg_cross_spec = mean(cross_spectrum_struct.('nt_circ_shift').cross_spec, 2);

clear n_min n_fft num_freq_bins P1 P2 cross_spectrum cross_corr f_axis shift_axis 

% Visualize Spectra 
%--------------------------------------------------------------------------

% Define font sizesI for plot elements
titleFontSize     = 20; % Title for each individual subplot
axisLabelFontSize = 20; % X and Y axis labels (e.g., 'SNR / dB')
tickLabelFontSize = 20; % The numbers on the axes (e.g., -20, -10, 0)

% (1) Modulation Spectra 
%--------------------------------------------------------------------------
figure('Name', 'OLSA Modulation Spectra', 'Color', 'w', 'WindowState', 'maximized');

% Left: Processed Envelope
subplot(1, 2, 1);
ax1                      = gca;
ax1.FontSize             = tickLabelFontSize;
ax1.TickLabelInterpreter = 'latex';
hold on;

h_all = plot(mod_spec.('envelope').freq_axis, 10 * log10(mod_spec.('envelope').all_spectra), 'Color', [0.5 0.5 0.5 0.5], 'LineWidth', 1); 
h_avg = plot(mod_spec.('envelope').freq_axis, 10 * log10(mod_spec.('envelope').avg_spectra), 'k', 'LineWidth', 2.5);

legend([h_all(1), h_avg], {'Single Sentence', 'Average Spectrum'}, 'Interpreter', 'latex', 'Location', 'northeast');
title('Cropped Envelopes during experiment', 'FontSize', titleFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
xlabel('Modulation Frequency / Hz', 'FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
ylabel('Power / dB', 'FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
xlim([0, 10]); 
% ylim([-50,0])
grid on;
grid minor;
box on;
axis square;

% Left: Original Sentence Envelope
subplot(1, 2, 2);
ax2                      = gca;
ax2.FontSize             = tickLabelFontSize;
ax2.TickLabelInterpreter = 'latex';
hold on;

h_all = plot(mod_spec.('envelope_orig').freq_axis, 10 * log10(mod_spec.('envelope_orig').all_spectra), 'Color', [0.5 0.5 0.5 0.5], 'LineWidth', 1); 
h_avg = plot(mod_spec.('envelope_orig').freq_axis, 10 * log10(mod_spec.('envelope_orig').avg_spectra), 'k', 'LineWidth', 2.5);

legend([h_all(1), h_avg], {'Single Sentence', 'Average Spectrum'}, 'Interpreter', 'latex', 'Location', 'northeast');
title(sprintf('Original Broadband Envelopes (1/f noise correction: %d)', apply_noise_correction), 'FontSize', titleFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
xlabel('Modulation Frequency / Hz', 'FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
ylabel('Power / dB', 'FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
xlim([0, 30]); % OLSA speech modulation usually peaks around 4 Hz
% ylim([-50,0])
grid on;
grid minor;
box on;
axis square;

sgtitle('Modulation Spectra OLSA sentences', 'FontSize', titleFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex')

% (2) Visualize Power Spectra: Acoustic Similarity + Neural Tracking (+Modulation Spectrum, optional)
%----------------------------------------------------------------------------------------------------
% Optional Normalization to 0 dB
apply_normalization4plot = false;

% Add modultion spectrum
add_mod_spec = true;

spectrum_envelope_avg = 10*log10(mod_spec.('envelope').avg_spectra);
nt_spectrum_avg       = 10*log10(mod_spec.('neural_tracking').avg_spectra);
nt_spectrum_all       = 10*log10(mod_spec.('neural_tracking').all_spectra);
as_spectrum_circ      = 10*log10(mod_spec.('acoustic_circ').spectra);
y_label               = 'Power / dB';

if apply_normalization4plot
    offset                = max(spectrum_envelope_avg);
    spectrum_envelope_avg = spectrum_envelope_avg - offset;

    offset                = max(nt_spectrum_avg);
    nt_spectrum_avg       = nt_spectrum_avg - offset;
    nt_spectrum_all       = nt_spectrum_all - offset;

    offset                = max(as_spectrum_circ);
    as_spectrum_circ      = as_spectrum_circ - offset;

    y_label = 'Normalized Power / dB';
end

colors        = distinguishable_colors(n_sens + 3);
h_legend      = [];
labels_legend = {};

figure('Name', 'All Power Spectra', 'Color', 'w', 'WindowState', 'maximized');
ax                      = gca;
ax.FontSize             = tickLabelFontSize;
ax.TickLabelInterpreter = 'latex';
hold on;

% Plot Neural Tracking Spectra
for sens_idx = 1:n_sens
    h_sens = plot(mod_spec.('neural_tracking').freq_axis, nt_spectrum_all(:, sens_idx), 'LineWidth', 1, 'Color', [colors(sens_idx, :), 0.2]); 

    h_legend(end+1)      = h_sens; 
    labels_legend{end+1} = sensor_labels{sens_idx};
end  
% Average Neural Tracking
h_avg                = plot(mod_spec.('neural_tracking').freq_axis, nt_spectrum_avg, 'LineWidth', 2.5, 'LineStyle', '-', 'Color', colors(n_sens+1, :));
h_legend(end+1)      = h_avg;
labels_legend{end+1} = 'Avg. Neu. Tr. Slopes (Circ. Shift)';
% Acoustic Similarity
h_circ               = plot(mod_spec.('acoustic_circ').freq_axis, as_spectrum_circ, 'LineWidth', 2.5, 'LineStyle', '--', 'Color', colors(n_sens+2, :));
h_legend(end+1)      = h_circ;
labels_legend{end+1} = sprintf('Ac. Sim. (Circ. Shift / type: %s)', circ_shift_type);
% Modulation Spectrum
if add_mod_spec
    h_mod                = plot(mod_spec.('envelope').freq_axis, spectrum_envelope_avg, 'LineWidth', 2.5, 'LineStyle', '-.', 'Color', colors(n_sens+3, :));
    h_legend(end+1)      = h_mod;
    labels_legend{end+1} = 'Modulation Spectrum OLSA';
end

legend(h_legend, labels_legend, 'Interpreter', 'latex', 'Location', 'southwest');
title('Power Spectra', 'FontSize', titleFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
xlabel('Modulation Frequency / Hz', 'FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
ylabel(y_label, 'FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
xlim([0, 10]); 
% ylim([-50,0])
grid on;
grid minor;
box on;
axis square; 

clear spectrum_envelope_avg nt_spectrum_avg nt_spectrum_all as_spectrum_circ y_label offset h_sens h_avg h_circ h_mod h_legend labels_legend
clear add_mod_spec apply_normalization4plot

% (3) Visualize Cross Spectrum (+ Modulation Spectrum, optional)
%--------------------------------------------------------------------------
% Optional Normalization to 0 dB
apply_normalization4plot = false;

% Add modultion spectrum
add_mod_spec = true;

spectrum_envelope_avg = 10*log10(mod_spec.('envelope').avg_spectra);
cross_spec_all        = 10*log10(cross_spectrum_struct.('nt_circ_shift').cross_spec);
cross_spec_avg        = 10*log10(cross_spectrum_struct.('nt_circ_shift').avg_cross_spec);
y_label               = 'Power / dB';

if apply_normalization4plot
    offset                = max(spectrum_envelope_avg);
    spectrum_envelope_avg = spectrum_envelope_avg - offset;

    offset                = max(cross_spec_avg);
    cross_spec_avg        = cross_spec_avg - offset;
    cross_spec_all        = cross_spec_all - offset;

    y_label = 'Normalized Power / dB';
end

colors        = distinguishable_colors(n_sens + 2);
h_legend      = [];
labels_legend = {};

figure('Name', 'Cross Spectrum: Neural Tracking Slopes and Acoustic Similarity (Circular Shift)', 'Color', 'w', 'WindowState', 'maximized');
ax                      = gca;
ax.FontSize             = tickLabelFontSize;
ax.TickLabelInterpreter = 'latex';
hold on;

% Plot Cross Spectra
for sens_idx = 1:n_sens
    h_sens = plot(cross_spectrum_struct.('nt_circ_shift').freq_axis, cross_spec_all(:, sens_idx), 'LineWidth', 1, 'Color', [colors(sens_idx, :), 0.2]); 

    h_legend(end+1)      = h_sens; 
    labels_legend{end+1} = sensor_labels{sens_idx};
end  
% Average Cross Spectrum
h_avg                = plot(cross_spectrum_struct.('nt_circ_shift').freq_axis, cross_spec_avg, 'LineWidth', 2.5, 'LineStyle', '-', 'Color', colors(n_sens+1, :));
h_legend(end+1)      = h_avg;
labels_legend{end+1} = sprintf('Average Cross Spectrum (type: %s)', circ_shift_type);
% Modulation Spectrum
if add_mod_spec
    h_mod                = plot(mod_spec.('envelope').freq_axis, spectrum_envelope_avg, 'LineWidth', 2.5, 'LineStyle', '--', 'Color', colors(n_sens+2, :));
    h_legend(end+1)      = h_mod;
    labels_legend{end+1} = 'Modulation Spectrum OLSA';
end

legend(h_legend, labels_legend, 'Interpreter', 'latex', 'Location', 'southwest');
title('Cross Spectrum: Neural Tracking Slopes and Acoustic Similarity (Circular Shift)', 'FontSize', titleFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
xlabel('Frequency / Hz', 'FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
ylabel(y_label, 'FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
xlim([0, 10]); 
% ylim([-80, 0])
grid on;
grid minor;
box on;
axis square;

clear spectrum_envelope_avg cross_spec_avg cross_spec_all y_label offset h_sens h_avg h_mod h_legend labels_legend
clear add_mod_spec apply_normalization4plot

%% Visualize similaritiy metrics for audio
%--------------------------------------------------------------------------

% Choose metric
%--------------
metric2plot = 'corr';
% metric2plot = 'rmse';

% Select circular envelope type 
% circ_shift_type = 'average'; % mean/median envelope cross correlation
circ_shift_type = 'concat'; % does not have selection (single computaion, no averging possible)

% Choose mean or median values (only availabe for circ_shift_type ='average')
plot_mean = true; % otherwise median is plotted

%--------------------------------------------------------------------------

switch circ_shift_type
    case 'concat'
        sim_vals_sentcs_marg       = audio_stats.(metric2plot).sim_vals_sentcs_marg_mean;
        mean_sim_vals_sentcs_shift = audio_stats.(metric2plot).nt_sim_vals_sentcs_shift;
        shift_time2plot            = audio_stats.(metric2plot).nt_sim_vals_sentcs_shift_samples*1000/fs; % ms
    case 'average'
        if plot_mean
            sim_vals_sentcs_marg       = audio_stats.(metric2plot).sim_vals_sentcs_marg_mean;
            mean_sim_vals_sentcs_shift = audio_stats.(metric2plot).mean_sim_vals_sentcs_shift;
        else 
            sim_vals_sentcs_marg       = audio_stats.(metric2plot).sim_vals_sentcs_marg_median;
            mean_sim_vals_sentcs_shift = audio_stats.(metric2plot).median_sim_vals_sentcs_shift;
        end
        shift_time2plot = audio_stats.(metric2plot).sim_vals_sentcs_shift_samples*1000/fs; % ms
end

% Sort values
[sim_vals_sentcs_marg_sorted, ~] = sort(sim_vals_sentcs_marg, 'descend');

if strcmp(metric2plot, 'corr')
    sim_vals_sentcs_marg_sorted = apply_inverse_fisher_z_transform(sim_vals_sentcs_marg_sorted);
    mean_sim_vals_sentcs_shift  = apply_inverse_fisher_z_transform(mean_sim_vals_sentcs_shift);
end

if plot_mean
    metric_label = 'mean';
else
    metric_label = 'median';
end

% Define font sizes for plot elements
titleFontSize     = 20; % Title for each individual subplot
axisLabelFontSize = 20; % X and Y axis labels (e.g., 'SNR / dB')
tickLabelFontSize = 20; % The numbers on the axes (e.g., -20, -10, 0)
markerSize        = 8;
textBoxFontSize   = 20; 

figure('Name', sprintf('Similarity metric (%s) between sentence pairs (%s)', metric2plot, metric_label), 'Color', 'w', 'WindowState', 'maximized');

% Sentence similarity
%--------------------------------------------------------------------------

% Plot data
%----------
subplot(1,2,1)
% figure('Name', 'Sorted mean similarity for sentences', metric2plot), 'Color', 'w');
plot(sim_vals_sentcs_marg_sorted, 'x', 'Color', [0.2 0.4 0.6], 'LineWidth', 2, 'MarkerSize', markerSize);
hold off;
ylabel(sprintf('Similarity metric (%s)', metric2plot), 'FontSize', axisLabelFontSize, 'FontWeight', 'bold');
xlabel('Sentences', 'FontSize', axisLabelFontSize, 'FontWeight', 'bold');
ax = gca;
ax.FontSize = tickLabelFontSize;
title('Sorted mean similarity for sentences', 'FontSize', titleFontSize, 'FontWeight', 'bold');
grid on;
grid minor;
axis square;
box on;

% Include textbox for normality check
check_normality = audio_stats.(metric2plot).check_sim_vals_sentcs_normality;
text(0.95, 0.95, sprintf('N_{normal}: %d / %d', sum(check_normality), numel(check_normality)), ...
    'Units', 'normalized', ...
    'VerticalAlignment', 'top', ...
    'HorizontalAlignment', 'right', ...
    'FontSize', textBoxFontSize, ...
    'FontWeight', 'normal', ...
    'BackgroundColor', 'none', ...
    'EdgeColor', 'k');

clear sim_vals_sentcs_marg sim_vals_sentcs_marg_sorted check_normality

% Similarity using time-shift
%--------------------------------------------------------------------------

% Plot data
%----------
subplot(1,2,2)
%figure('Name', 'Mean similarity for shifted sentences', 'Color', 'w');

hold on;
% Plot the 'x' markers with a thick LineWidth
plot(shift_time2plot, mean_sim_vals_sentcs_shift, 'x', 'Color', [0.2 0.4 0.6], 'LineWidth', 2, 'MarkerSize', markerSize);
hold off;

ylabel(sprintf('Similarity metric (%s)', metric2plot), 'FontSize', axisLabelFontSize, 'FontWeight', 'bold');
xlabel('Circular shift / ms', 'FontSize', axisLabelFontSize, 'FontWeight', 'bold');
ax = gca;
ax.FontSize = tickLabelFontSize;
title('Mean similarity for shifted sentences', 'FontSize', titleFontSize, 'FontWeight', 'bold');
grid on;
grid minor;
axis square;
box on;

% Include textbox for normality check
check_normality = audio_stats.('corr').check_sim_vals_shift_normality;
text(0.95, 0.95, sprintf('N_{normal}: %d / %d', sum(check_normality), numel(check_normality)), ...
    'Units', 'normalized', ...
    'VerticalAlignment', 'top', ...
    'HorizontalAlignment', 'right', ...
    'FontSize', textBoxFontSize, ...
    'FontWeight', 'normal', ...
    'BackgroundColor', 'none', ...
    'EdgeColor', 'k');

clear shift_time2plot mean_sim_vals_sentcs_shift

sgtitle(sprintf('Similarity metric (%s) between sentence pairs (%s used / %s envelope similarity)', metric2plot, metric_label, circ_shift_type), 'FontSize', titleFontSize, 'FontWeight', 'bold')

%% Additional plots - Neural Tracking and acoustic similarity with shifts
%--------------------------------------------------------------------------

% Choose metric
%--------------
metric2plot = 'mixed_model';
% metric2plot = 'pooled_linear_model';
% metric2plot = 'individual_linear_model';
% metric2plot = 'effect_size';

% User Selections - only a single condition possible
conditions2plot = 'ses-pooled';

% Select mean or median (only relevant if circ_shift_type = 'average')
%---------------------------------------------------------------------
acoustic_sim2use = 'mean';
% acoustic_sim2use = 'median';

% Add labels for shift peaks
%---------------------------
add_peak_labels = true;

% Find Data and Calculate Axis Limits 
%------------------------------------
con_idx = ismember(conditions, conditions2plot);

% Highlight sentence/shift showing selected significant statistic
%-----------------------------------------------------------------
% Select which statistic to plot 
show_stats4plot = true;
% stats4plot_type = 'contrast2null'; % significant difference to null distribution
% stats4plot_type = 'significant_slope'; % Slope significantly different from 0
stats4plot_type = 'both'; % Both combined

% Set xlimits for data
%---------------------
xlim_data = [-1000, 1000]; % ms

% Define font sizes for plot elements
axisLabelFontSize = 20; 
tickLabelFontSize = 20; 
legendFontSize    = 15; 
markerSize        = 6;
textBoxFontSize   = 10;
titleFontSize     = 20;

% Adapt p-values based on selected statistic and shift-window
%------------------------------------------------------------
% Aggregate all relevant data to calculate global axis limits
[~, idx4plot] = min(abs(shift_time(:)*1000 - xlim_data));
idx4plot      = idx4plot(1):idx4plot(2);

if any(strcmp(stats4plot_type, {'contrast2null', 'both'}))

    dim_idx               = 1;
    p_vals                = stats_null_distr.('shift').p(:, :, :, idx4plot);
    [p_vals, n_corrected] = apply_multiple_comparison_correction(p_vals, mcp_method, mcp_scope, dim_idx);
    
    p_vals        = squeeze(p_vals(con_idx, :, :, :)); % Select condition
    contrast2null = false(n_sens, numel(idx4plot)); % Init
    for sens_idx = 1:n_sens
        for extra_idx = 1:numel(idx4plot)
            % Count number of signifant tests
            % n_sig = sum(p_vals < alpha_level);
            n_sig = sum(p_vals(sens_idx, 2:end, extra_idx) < 0.05); % ignore first snr
            if n_sig >= 3 % 3/5 significant
                contrast2null(sens_idx, extra_idx) = true;
            end
        end % trials
    end % sensors
    clear p_vals n_corrected n_sig dim_idx
end

if any(strcmp(stats4plot_type, {'significant_slope', 'both'}))

    % Significant Slope
    dim_idx               = 1;
    p_vals                = stats_neuro.('shift').(metric2plot).pval(:, :, idx4plot);
    [p_vals, n_corrected] = apply_multiple_comparison_correction(p_vals, mcp_method, mcp_scope, dim_idx);
    
    p_vals = squeeze(p_vals(con_idx, :, :));
    
    significance_slope              = false(size(p_vals));
    significance_slope(p_vals<0.05) = true;
    clear p_vals n_corrected dim_idx

end
    
if ~any(strcmp(stats4plot_type, {'contrast2null', 'significant_slope', 'both'}))
    error('Unexpected option for statistic (%s)!', show_sign_stats)
end

% Select final statistic
if strcmp(stats4plot_type, 'contrast2null')
    stats4plot = contrast2null;
    clear contrast2null
elseif strcmp(stats4plot_type, 'significant_slope')
    stats4plot = significance_slope;
    clear significance_slope
else
    stats4plot = and(contrast2null, significance_slope);
    clear contrast2null significance_slope
end

% Compute axis limits
%--------------------
switch metric2plot
    case {'mixed_model', 'pooled_linear_model', 'individual_linear_model'}
        sim_stats = stats_neuro.('shift').(metric2plot).slope(con_idx,:,idx4plot);
        % label_y   = sprintf('%s (slope)', strrep(metric2plot, '_', '-'));
        label_y = 'Fixed-effect slope $\left(\beta_{I}\right)$';
    case 'effect_size'
        sim_stats = stats_neuro.('shift').(metric2plot)(con_idx,:,idx4plot);
        % label_y   = strrep(metric2plot, '_', '-');
        label_y   = 'Neural Tracking Slope';
end

y_min      = min(sim_stats(:)); 
y_max      = max(sim_stats(:));
ylim_min   = y_min - 0.05 * (y_max - y_min) - 0.015; % offset correction
ylim_max   = y_max + 0.05 * (y_max - y_min);

x_min         = min(shift_time(idx4plot)*1000,[],"all"); 
x_max         = max(shift_time(idx4plot)*1000,[],"all"); 
xlim_min      = x_min - 0.05 * (x_max - x_min);
xlim_max      = x_max + 0.05 * (x_max - x_min);

% 1.) Neural Tracking Slopes with shift
%--------------------------------------------------------------------------
markers = {'^','x','*'};
colors  = {[0, 0, 1], [1, 0, 0], [0, 0.7, 0]};
offsets = [0.01, 0.015, 0.02];

figure('color','white', ...
       'Name', sprintf('Neural Tracking with shifts (%s / stats4plot: %s / acoustic-sim2use: %s / circ-shift-type: %s)', metric2plot, stats4plot_type, acoustic_sim2use, circ_shift_type), ...
       'WindowState', 'maximized');
subplot(1,3,1)

ax          = gca;
ax.FontSize = tickLabelFontSize;
ax.TickLabelInterpreter = 'latex';
hold(ax, 'on');

% Plot metric
%------------
switch metric2plot
    case {'mixed_model', 'pooled_linear_model', 'individual_linear_model'}
        sim_stats = squeeze(stats_neuro.('shift').(metric2plot).slope(con_idx, :, :));
    case 'effect_size'
        sim_stats = squeeze(stats_neuro.('shift').(metric2plot)(con_idx, :, :));
end

for sens_idx = 1:n_sens

    plot(shift_time(idx4plot)*1000, sim_stats(sens_idx, idx4plot), markers{sens_idx}, 'Color', colors{sens_idx}, 'LineWidth', 1.5, 'MarkerSize', markerSize, 'DisplayName', sensor_labels{sens_idx});
   
    % Add line for difference to null distribution
    if show_stats4plot
        stats_idx          = squeeze(stats4plot(sens_idx,:));
        alphas             = ones(size(stats_idx));
        alphas(~stats_idx) = nan; 
        % Plot
        plot(shift_time(idx4plot)*1000, (y_min-offsets(sens_idx)) *alphas, '-', 'Color', colors{sens_idx}, 'LineWidth', 2, 'HandleVisibility', 'off');
        
        clear stats_idx alphas
    end
end % sensors

% Add average
%------------
sim_stats_avg = mean(sim_stats(:, idx4plot), 1);
plot(shift_time(idx4plot)*1000, sim_stats_avg, '-', 'Color', 'k', 'LineWidth', 2, 'DisplayName', 'Average');
        
% Add 0-line
xline(0, '--k', 'HandleVisibility', 'off'); 

% Add peaks
%----------
if add_peak_labels 
    % Use Average data to mount points
    [~, locs_max]      = findpeaks(sim_stats_avg, shift_time(idx4plot)*1000);
    [~, locs_min]      = findpeaks(-sim_stats_avg, shift_time(idx4plot)*1000);
    all_extrema_x_vals = sort([locs_max, locs_min]);
    
    % Filter for only positive x-axis values (original)
    targeted_extrema = all_extrema_x_vals(...
        (all_extrema_x_vals > 0) | ...
        (all_extrema_x_vals >= -200 & all_extrema_x_vals < 0) | ...
        (all_extrema_x_vals >= -550 & all_extrema_x_vals <= -500)...
    );

    % % Filter for only positive x-axis values (onset removed)
    % targeted_extrema = all_extrema_x_vals(...
    %     (all_extrema_x_vals > 0) | ...
    %     (all_extrema_x_vals >= -200 & all_extrema_x_vals < 0) | ...
    %     (all_extrema_x_vals >= -400 & all_extrema_x_vals <= -300)...
    % );

    % Separate and sort negative values by absolute magnitude (easier for
    % plotting)
    neg_vals         = targeted_extrema(targeted_extrema < 0);
    [~, s_idx]       = sort(abs(neg_vals), 'ascend');
    targeted_extrema = [neg_vals(s_idx), targeted_extrema(targeted_extrema >= 0)];
    clear neg_vals s_idx

    n_extrema = length(targeted_extrema);
    
    % Plot vertical lines and staggered text boxes
    hold on; 
    ylims   = [ylim_min, ylim_max];
    y_range = ylims(2) - ylims(1); % Calculate the total height of the y-axis
    
    % Define how many staggered levels you want, and how far apart they are
    step_down_amount = 0.13 * y_range; 
    y_start          = ylims(2) * 0.95; % Top baseline height

    % Separate counters for left and right stacking
    y_start_neg = ylims(2) * 0.55; % Top baseline height for the LEFT side
    y_start_pos = ylims(2) * 0.95; % Top baseline height for the RIGHT side
    count_neg   = 0;
    count_pos   = 0;
          
    h_texts = gobjects(n_extrema, 1);
    for i = 1:n_extrema
        x_val = targeted_extrema(i);

        % Direct the box, line, and calculate the independent height based on side
        if x_val < 0
            xlim_target = xlim_min + 75; % Shifts the X anchor point to the right (in ms)
            horz_align  = 'left';
            y_val       = y_start_neg - (count_neg * step_down_amount);
            count_neg   = count_neg + 1; 
        else
            xlim_target = xlim_max;
            horz_align  = 'right';
            y_val       = y_start_pos - (count_pos * step_down_amount);
            count_pos   = count_pos + 1; 
        end
    
        % Draw a dashed vertical line 
        plot([x_val, x_val], [ylims(1), y_val], '--k', 'HandleVisibility', 'off');
        % Draw a dashed hrizontal line
        plot([x_val, xlim_target], [y_val, y_val], '--k', 'HandleVisibility', 'off');
    
        % Calculate frequency in Hz (since x_val is in ms)
        freq_val = 1000 / abs(x_val);
        
        % Create a multi-line cell array for the text
        label_text = sprintf('$$\\begin{array}{l} \\tau \\approx %.0f \\, \\mathrm{ms} \\\\ f \\approx %.1f \\, \\mathrm{Hz} \\end{array}$$', ...
                             x_val, freq_val);

        % Add the text box at the newly calculated staggered height
        h_texts(i) = text(xlim_target, y_val, label_text, ...
            'VerticalAlignment', 'top', ...     
            'HorizontalAlignment', horz_align, ...  
            'BackgroundColor', 'white', ...     
            'EdgeColor', 'black', ...  
            'FontSize', textBoxFontSize, ...
            'FontWeight', 'bold', ...
            'Interpreter', 'latex', ...
            'Margin', 3);  
    end
    
    % Force ALL text boxes to the absolute top of the visual stack
    uistack(h_texts, 'top');
    hold off;
    clear all_extrema_x_vals locs_max locs_min targeted_extrema n_extrema num_levels step_down_amount y_levels x_val y_val freq_val label_text xlim_target horz_align
end

% Adjust xticks
% step_size  = 250;
step_size  = abs(diff(xlim_data)) / 4;
start_tick = ceil(xlim_data(1) / step_size) * step_size;
end_tick   = floor(xlim_data(2) / step_size) * step_size;
xticks(start_tick : step_size : end_tick);

box on;
grid on;
grid minor;
axis square;
xlim([xlim_min, xlim_max]);
ylim([ylim_min, ylim_max]);
xlabel('Circular shift $(\tau)$ / ms', 'FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
ylabel(label_y, 'FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
title({'Neural Tracking Slopes'}, 'FontSize', titleFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
hold(ax, 'off');
legend('Location', 'northwest', 'FontSize', legendFontSize, 'Interpreter', 'latex');
clear stats4plot sim_stats sim_stats_avg

% 2.) Acoustic similarity between Olsa sentences with circular shift
%--------------------------------------------------------------------------

% Select data
%------------
% Option that was selected during computation
switch cross_spectrum_struct.('nt_circ_shift').circ_shift_type
    case 'concat'
        sim_vals_sentcs_shift = apply_inverse_fisher_z_transform(audio_stats.('corr').nt_sim_vals_sentcs_shift);
        metric_label          = 'Concatenated Envelopes';
        shift_time2plot       = audio_stats.('corr').nt_sim_vals_sentcs_shift_samples/fs; % s
    case 'average'
        switch acoustic_sim2use
            case 'mean'
                sim_vals_sentcs_shift = apply_inverse_fisher_z_transform(audio_stats.('corr').mean_sim_vals_sentcs_shift);
                metric_label          = 'Mean Envelopes';
            case 'median'
                sim_vals_sentcs_shift = apply_inverse_fisher_z_transform(audio_stats.('corr').median_sim_vals_sentcs_shift);
                metric_label          = 'Median Envelopes';
        end
        shift_time2plot = audio_stats.('corr').sim_vals_sentcs_shift_samples/fs; % s
end

[~, idx4plot] = min(abs(shift_time2plot(:)*1000 - xlim_data));
idx4plot      = idx4plot(1):idx4plot(2);

x_min         = min(shift_time2plot(idx4plot)*1000,[],"all"); 
x_max         = max(shift_time2plot(idx4plot)*1000,[],"all"); 
xlim_min      = x_min - 0.05 * (x_max - x_min);
xlim_max      = x_max + 0.05 * (x_max - x_min);

figure('color','white','Name', sprintf('Acoustic similarity with shifts (%s)', metric_label), 'WindowState', 'maximized');
subplot(1,3,2)

ax          = gca;
ax.FontSize = tickLabelFontSize;
ax.TickLabelInterpreter = 'latex';
hold(ax, 'on');

% Plot the 'x' markers with a thick LineWidth
% [0.2 0.4 0.6]
plot(shift_time2plot(idx4plot)*1000, sim_vals_sentcs_shift(idx4plot), 'x', 'Color', 'k', 'LineWidth', 2, 'MarkerSize', markerSize, 'DisplayName', 'OLSA Envelope Similarity');

% Add 0-line
xline(0, '--k', 'HandleVisibility', 'off'); 

% Add peaks
%----------
if add_peak_labels 
    [~, locs_max]      = findpeaks(sim_vals_sentcs_shift(idx4plot), shift_time2plot(idx4plot)*1000);
    [~, locs_min]      = findpeaks(-sim_vals_sentcs_shift(idx4plot), shift_time2plot(idx4plot)*1000);
    all_extrema_x_vals = sort([locs_max, locs_min]);
    
    % Filter for only positive x-axis values
    pos_extrema_x = all_extrema_x_vals(all_extrema_x_vals > 0);
    n_extrema     = length(pos_extrema_x);
    
    % Plot vertical lines and staggered text boxes
    hold on; 
    ylims = ylim; 
    y_range = ylims(2) - ylims(1); % Calculate the total height of the y-axis
    
    % Define how many staggered levels you want, and how far apart they are
    num_levels       = n_extrema;
    step_down_amount = 0.13 * y_range; 
          
    % Pre-calculate the exact y-coordinates for our staggered levels
    y_levels = ylims(2)*0.95 - (0:(num_levels-1)) * step_down_amount;
    
    h_texts = gobjects(n_extrema, 1);
    for i = 1:n_extrema
        x_val = pos_extrema_x(i);
        y_val = y_levels(i);
    
        % Draw a dashed vertical line 
        plot([x_val, x_val], [ylims(1), y_val], '--k', 'HandleVisibility', 'off');
        % Draw a dashed hrizontal line
        plot([x_val, xlim_max], [y_val, y_val], '--k', 'HandleVisibility', 'off');
    
        % Calculate frequency in Hz (since x_val is in ms)
        freq_val = 1000 / x_val;
        
        % Create a multi-line cell array for the text
        label_text = sprintf('$$\\begin{array}{l} \\tau \\approx %.0f \\, \\mathrm{ms} \\\\ f \\approx %.1f \\, \\mathrm{Hz} \\end{array}$$', ...
                             x_val, freq_val);
        
        % Add the text box at the newly calculated staggered height
        h_texts(i) = text(xlim_max, y_val, label_text, ...
                          'VerticalAlignment', 'top', ...     
                          'HorizontalAlignment', 'right', ...  
                          'BackgroundColor', 'white', ...     
                          'EdgeColor', 'black', ...  
                          'FontSize', textBoxFontSize, ...
                          'FontWeight', 'bold', ...
                          'Interpreter', 'latex', ...
                          'Margin', 3);  
    end
    
    % Force ALL text boxes to the absolute top of the visual stack
    uistack(h_texts, 'top');
    hold off;
    clear all_extrema_x_vals locs_max locs_min pos_extrema_x n_extrema num_levels step_down_amount y_levels x_val y_val freq_val label_text 
end

xticks(start_tick : step_size : end_tick);
box on;
grid on;
grid minor;
axis square;
xlim([xlim_min, xlim_max]);
xlabel('Circular shift $(\tau)$ / ms', 'FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
ylabel('Acoustic similarity $\left(\rho_{shift}\right)$', 'FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
% legend('Location', 'northwest', 'FontSize', legendFontSize, 'Interpreter', 'latex');
title({'OLSA Envelope Similarity'}, 'FontSize', titleFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
hold(ax, 'off');

clear sim_vals_sentcs_shift

% 3.) Modulation Spectrum and Cross-Power Spectrum
%--------------------------------------------------------------------------

% Plot STD or individual time-series
line2plot = 'std'; % std or all

% Load data and normalize
spectrum_envelope_avg = 10 * log10(mod_spec.('envelope').avg_spectra);
offset                = max(spectrum_envelope_avg);
spectrum_envelope_avg = spectrum_envelope_avg - offset;

spectrum_envelope_all = 10 * log10(mod_spec.('envelope').all_spectra);
spectrum_envelope_std = std(spectrum_envelope_all, 0, 2);
spectrum_envelope_all = spectrum_envelope_all - offset;

cross_spec_avg = 10*log10(cross_spectrum_struct.('nt_circ_shift').avg_cross_spec);
offset         = max(cross_spec_avg);
cross_spec_avg = cross_spec_avg - offset;

cross_spec_all = 10 * log10(cross_spectrum_struct.('nt_circ_shift').cross_spec);
cross_spec_std  = std(cross_spec_all, 0, 2);
cross_spec_all = cross_spec_all  - offset;

figure('Name', 'Cross-Power Spectrum and Modulation Spectrum', 'Color', 'w', 'WindowState', 'maximized');
subplot(1,3,3)

ax          = gca;
ax.FontSize = tickLabelFontSize;
ax.TickLabelInterpreter = 'latex';
hold(ax, 'on');

switch line2plot
    case 'all'
        h1 = plot(mod_spec.('envelope').freq_axis, spectrum_envelope_all, ...
            'Color', [0.5 0.5 0.5 0.5], 'LineWidth', 1, ...
            'DisplayName', 'Single Sent. Mod. Spec.'); 
        % h2 = plot(cross_spectrum_struct.('nt_circ_shift').freq_axis(:)', cross_spec_all, ...
        %     'Color', [0.0 0.0 1.0 0.5], 'LineWidth', 1, ...
        %     'DisplayName', 'Single Sensor Cr.-Pow. Spec.');
    case 'std'
        h1 = fill([mod_spec.('envelope').freq_axis(:)', fliplr(mod_spec.('envelope').freq_axis(:)')], ...
                   [spectrum_envelope_avg(:)' - spectrum_envelope_std(:)', fliplr(spectrum_envelope_avg(:)' + spectrum_envelope_std(:)')] ...
                   , [0.5 0.5 0.5], 'FaceAlpha', 0.3, 'EdgeColor', 'none', ...
                   'DisplayName', '$\pm 1$ Std. Dev.');
        % h2 = fill([cross_spectrum_struct.('nt_circ_shift').freq_axis(:)', fliplr(cross_spectrum_struct.('nt_circ_shift').freq_axis(:)')], ...
        %            [cross_spec_avg(:)' - cross_spec_std(:)', fliplr(cross_spec_avg(:)' + cross_spec_std(:)')] ...
        %            , [0 0 1], 'FaceAlpha', 0.3, 'EdgeColor', 'none', ...
        %            'DisplayName', '$\pm 1$ Std. Dev.');
end

h3 = plot(mod_spec.('envelope').freq_axis, spectrum_envelope_avg, 'Color', 'k', 'LineWidth', 2.5, 'DisplayName', 'Avg. Mod. Spec.');
% h4 = plot(cross_spectrum_struct.('nt_circ_shift').freq_axis, cross_spec_avg, 'Color','b', 'LineWidth', 2.5, 'DisplayName', 'Avg. Cr.-Pow. Spec.');

% Set x-axis resolution to 1 Hz
xticks(0:1:9);
xtickangle(0);
xlim([0, 9]); 
% ylim([-50, 10]);
ylim([-15, 6]);

% % Add peaks
% %----------
% if add_peak_labels 
%     % [~, locs_max1]     = findpeaks(spectrum_envelope_avg, mod_spec.('envelope').freq_axis(:)');
%     locs_max1          = [];
%     [~, locs_max2]     = findpeaks(cross_spec_avg, cross_spectrum_struct.('nt_circ_shift').freq_axis(:)');
%     all_extrema_x_vals = sort([locs_max1, locs_max2]);
% 
%     % Filter for only positive x-axis values
%     pos_extrema_x = all_extrema_x_vals(all_extrema_x_vals > 0.5 & all_extrema_x_vals < 4);
%     n_extrema     = length(pos_extrema_x);
% 
%     % Plot vertical lines and staggered text boxes
%     hold on; 
%     ylims   = ylim; 
%     y_range = ylims(2) - ylims(1);
%     xlims   = xlim;
% 
%     % Define how many staggered levels you want, and how far apart they are
%     num_levels       = n_extrema;
%     step_down_amount = 0.1 * y_range; 
% 
%     % Pre-calculate the exact y-coordinates for our staggered levels
%     y_levels = ylims(2)*0.8 - (0:(num_levels-1)) * step_down_amount;
% 
%     % Manual color adjustment
%     % color4box = {'blue', 'black'};
%     color4box = repmat({'blue'}, 1, n_extrema);
% 
%     h_texts = gobjects(n_extrema, 1);
%     for i = 1:n_extrema
%         x_val = pos_extrema_x(i);
%         y_val = y_levels(i);
% 
%         % Draw a dashed vertical line 
%         plot([x_val, x_val], [ylims(1), y_val], '--', 'Color', color4box{i}, 'HandleVisibility', 'off');
%         % Draw a dashed hrizontal line
%         plot([x_val, xlims(2)], [y_val, y_val], '--', 'Color', color4box{i}, 'HandleVisibility', 'off');
% 
%         % Create a single-line LaTeX string showing only the x-value 
%         label_text = sprintf('$$f \\approx %.1f \\, \\mathrm{Hz}$$', x_val);
% 
%         % Add the text box at the newly calculated staggered height 
%         h_texts(i) = text(xlims(2), y_val, label_text, ...
%                           'VerticalAlignment', 'top', ...     
%                           'HorizontalAlignment', 'right', ...  
%                           'BackgroundColor', 'white', ...     
%                           'EdgeColor', color4box{i}, ...  
%                           'Color', color4box{i}, ...
%                           'FontSize', textBoxFontSize, ...
%                           'FontWeight', 'bold', ...
%                           'Interpreter', 'latex', ...
%                           'Margin', 3);  
%     end
% 
%     % Force ALL text boxes to the absolute top of the visual stack
%     uistack(h_texts, 'top');
%     hold off;
%     clear all_extrema_x_vals locs_max locs_min pos_extrema_x n_extrema num_levels step_down_amount y_levels x_val y_val freq_val label_text 
% end

% Add phoneme and word rates
%--------------------------------------------------------------------------
% (3.6 +/- 0.3) Hz (range: 3.1--4.4) and a word rate of (2.1 +/- 0.2) Hz (range: 1.8--2.5)
% Values were computed in analyze_olsa_speech_rates.m
phoneme_mean = 3.6;
phoneme_std  = 0.3;
phoneme_min  = 3.1;
phoneme_max  = 4.4;

word_mean = 2.1;
word_std  = 0.2;
word_min  = 1.8; 
word_max  = 2.5;

ylims              = ylim;
y_range            = ylims(2) - ylims(1);
text_y_pos_phoneme = ylims(1) + 0.03 * y_range; 
text_y_pos_word    = ylims(1) + 0.18 * y_range; 

% Plot Word Rate shaded area and line (Red) - removed HandleVisibility off
h_word_patch = fill([word_min, word_max, word_max, word_min], ...
                    [ylims(1), ylims(1), ylims(2), ylims(2)], ...
                    'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none');

h_word_line = plot([word_mean, word_mean], ylims, 'r', 'LineWidth', 2);

% Plot Phoneme Rate shaded area and line (Blue) - removed HandleVisibility off
h_phoneme_patch = fill([phoneme_min, phoneme_max, phoneme_max, phoneme_min], ...
                       [ylims(1), ylims(1), ylims(2), ylims(2)], ...
                       'b', 'FaceAlpha', 0.1, 'EdgeColor', 'none');

h_phoneme_line = plot([phoneme_mean, phoneme_mean], ylims, 'b', 'LineWidth', 2);

% Textboxes 
str_word    = sprintf('Word rate \n$$%.1f \\pm %.1f$$ Hz', word_mean, word_std);
h_word_text = text(word_mean, text_y_pos_word, str_word, ...
                  'VerticalAlignment', 'bottom', ... % Changed to bottom to grow upward
                  'HorizontalAlignment', 'center', ...
                  'BackgroundColor', 'white', ...
                  'EdgeColor', 'r', ...
                  'Color', 'r', ...
                  'FontSize', textBoxFontSize, ... 
                  'FontWeight', 'bold', ...
                  'Interpreter', 'latex', ...
                  'Margin', 3);

str_phoneme    = sprintf('Phoneme rate\n$$%.1f \\pm %.1f$$ Hz', phoneme_mean, phoneme_std);
h_phoneme_text = text(phoneme_mean, text_y_pos_phoneme, str_phoneme, ...
                     'VerticalAlignment', 'bottom', ... % Changed to bottom to grow upward
                     'HorizontalAlignment', 'center', ...
                     'BackgroundColor', 'white', ...
                     'EdgeColor', 'b', ...
                     'Color', 'b', ...
                     'FontSize', textBoxFontSize, ...
                     'FontWeight', 'bold', ...
                     'Interpreter', 'latex', ...
                     'Margin', 3);

% Send patches and lines to the very bottom layer
uistack(h_word_patch, 'bottom');
uistack(h_phoneme_patch, 'bottom');
uistack(h_word_line, 'bottom');
uistack(h_phoneme_line, 'bottom');
%--------------------------------------------------------------------------

box on;
grid on;
grid minor;
axis square;

legend([h3, h1(1)], 'Interpreter', 'latex', 'Location', 'northeast', 'FontSize', legendFontSize);
% legend([h3, h1(1), h4, h2(1)], 'Interpreter', 'latex', 'Location', 'southwest', 'FontSize', legendFontSize);
% title({'OLSA Modulation Spectrum', 'and Cross-Power'}, 'FontSize', titleFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
title('OLSA Modulation Spectrum', 'FontSize', titleFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
xlabel('Modulation frequency / Hz', 'FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
ylabel('Normalized power / dB', 'FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
hold(ax, 'off');

clear line2plot shift2plot spectrum_envelope_avg spectrum_envelope_all cross_spec_avg cross_spec_all spectrum_envelope_std cross_spec_std offset h1 h2 h3 h4 ax

%% Acoustic Similarity vs. Neural Tracking
%--------------------------------------------------------------------------
% Compute correlation values between combinations

data_fields       = {'single_sentences','shift'};
acoustic_metrics  = {'corr', 'rmse'};
neural_metrics    = {'mixed_model', 'pooled_linear_model', 'individual_linear_model', 'effect_size'};

% Select circular envelope type for computation
% circ_shift_type = 'average'; % mean envelope cross correlation
circ_shift_type = 'concat'; % concatenated cross correlation similar to neural tracking analysis

stats_neuro_audio = struct();
stats_neuro_audio.('shift').circ_shift_type = circ_shift_type;

for fn_idx = 1:numel(data_fields)    
    field = data_fields{fn_idx};
    for ac_idx = 1:numel(acoustic_metrics) 
        acoustic_metric = acoustic_metrics{ac_idx};
        for neu_idx = 1:numel(neural_metrics)
            neural_metric = neural_metrics{neu_idx};
            
            stats_neuro_audio.(field).(acoustic_metric).(neural_metric).p_mean   = nan(n_cond, n_sens);
            stats_neuro_audio.(field).(acoustic_metric).(neural_metric).r_mean   = nan(n_cond, n_sens); 
            stats_neuro_audio.(field).(acoustic_metric).(neural_metric).p_median = nan(n_cond, n_sens);
            stats_neuro_audio.(field).(acoustic_metric).(neural_metric).r_median = nan(n_cond, n_sens); 
        end
    end
end

for fn_idx = 1:numel(data_fields)    
    field = data_fields{fn_idx};

    for ac_idx = 1:numel(acoustic_metrics) 
        acoustic_metric = acoustic_metrics{ac_idx};

        switch field
            case 'single_sentences'
                sim_vals_acoustic_mean   = audio_stats.(acoustic_metric).sim_vals_sentcs_marg_mean;
                sim_vals_acoustic_median = audio_stats.(acoustic_metric).sim_vals_sentcs_marg_median;          
            case 'shift'
                switch circ_shift_type
                    case 'concat'
                        sim_vals_acoustic_mean   = audio_stats.(acoustic_metric).nt_sim_vals_sentcs_shift; % no mean / median selection possible
                        sim_vals_acoustic_median = audio_stats.(acoustic_metric).nt_sim_vals_sentcs_shift; % -> both are the same
                    case 'average'
                        sim_vals_acoustic_mean   = audio_stats.(acoustic_metric).mean_sim_vals_sentcs_shift;
                        sim_vals_acoustic_median = audio_stats.(acoustic_metric).median_sim_vals_sentcs_shift;
                end    
        end
        if strcmp(acoustic_metric, 'corr')
            sim_vals_acoustic_mean   = apply_inverse_fisher_z_transform(sim_vals_acoustic_mean);
            sim_vals_acoustic_median = apply_inverse_fisher_z_transform(sim_vals_acoustic_median);
        end

        for neu_idx = 1:numel(neural_metrics)
            neural_metric = neural_metrics{neu_idx};

            for con_idx = 1:n_cond
                for sens_idx = 1:n_sens
           
                    switch neural_metric
                        case {'mixed_model', 'pooled_linear_model', 'individual_linear_model'}
                            sim_vals_neuro = squeeze(stats_neuro.(field).(neural_metric).slope(con_idx, sens_idx, :));
                        case 'effect_size'
                            sim_vals_neuro = squeeze(stats_neuro.(field).(neural_metric)(con_idx, sens_idx, :));
                    end
        
                    % Mean
                    %-----
                    [r_pearson, p_pearson] = corr(sim_vals_acoustic_mean(:), sim_vals_neuro(:), 'Type', 'Pearson'); 
                    % Store
                    stats_neuro_audio.(field).(acoustic_metric).(neural_metric).r_mean(con_idx, sens_idx) = r_pearson;
                    stats_neuro_audio.(field).(acoustic_metric).(neural_metric).p_mean(con_idx, sens_idx) = p_pearson;

                    % Median
                    %-------
                    [r_pearson, p_pearson] = corr(sim_vals_acoustic_median(:), sim_vals_neuro(:), 'Type', 'Pearson'); 
                    % Store
                    stats_neuro_audio.(field).(acoustic_metric).(neural_metric).r_median(con_idx, sens_idx) = r_pearson;
                    stats_neuro_audio.(field).(acoustic_metric).(neural_metric).p_median(con_idx, sens_idx) = p_pearson;

                end % sensors
            end % conditions

            % Apply multiple comparisons correction
            %--------------------------------------
            dim_idx                                                                        = 1; % position of conditions argument
            [stats_corrected, n_corrected]                                                 = apply_multiple_comparison_correction(stats_neuro_audio.(field).(acoustic_metric).(neural_metric).p_mean, mcp_method, mcp_scope, dim_idx);
            stats_neuro_audio.(field).(acoustic_metric).(neural_metric).p_mean_adj         = stats_corrected;
            stats_neuro_audio.(field).(acoustic_metric).(neural_metric).p_mean_adj_n_tests = n_corrected;

            [stats_corrected, n_corrected]                                                   = apply_multiple_comparison_correction(stats_neuro_audio.(field).(acoustic_metric).(neural_metric).p_median, mcp_method, mcp_scope, dim_idx);
            stats_neuro_audio.(field).(acoustic_metric).(neural_metric).p_median_adj         = stats_corrected;
            stats_neuro_audio.(field).(acoustic_metric).(neural_metric).p_median_adj_n_tests = n_corrected;

        end % neural metrics
    end % acoustic metric
end % data fields
fprintf('All correlations between neural and acoustic similarity metrics computed.\n')
clear sim_vals_acoustic sim_vals_neuro r_pearson p_pearson stats_corrected n_corrected

%% Scatter plot: Acoustic Similarity vs. Neural Tracking
%--------------------------------------------------------------------------

% Select mean or median
%----------------------
acoustic_sim2use = 'mean';
% acoustic_sim2use = 'median';

% Choose datatype
%----------------
% data2plot = 'shift';
data2plot = 'single_sentences';

% Choose envelope similarity metric
%----------------------------------
acoustic_metric2plot = 'corr';
% acoustic_metric2plot = 'rmse';

% Choose neural tracking metric
%------------------------------
neural_metric2plot = 'mixed_model';
% neural_metric2plot = 'pooled_linear_model';
% neural_metric2plot = 'individual_linear_model';
% neural_metric2plot = 'effect_size'; 

% User Selections - only a single condition possible
%---------------------------------------------------
conditions2plot = 'ses-pooled';

% Add shuffled and mean/median sentence condition
%------------------------------------------------
show_shuffled_mean = true;

% Distinguish between significant and non-significant slopes
%-----------------------------------------------------------
show_significance = true; % does not work with effect size!

% Fade trials (sentence/shift) which dont differ from null distribution
%----------------------------------------------------------------------
show_diff2null = true;
alpha_solid    = 1.0;
alpha_faded    = 0.4;

% Calculate Axis Limits 
%--------------------------------------------------------------------------
con_idx = ismember(conditions, conditions2plot);

% x-axis
%-------
switch data2plot
    case 'single_sentences'
        switch acoustic_sim2use
            case 'mean'
                sim_vals_sentcs = audio_stats.(acoustic_metric2plot).sim_vals_sentcs_marg_mean; % Mean between envelope pairs
                % sim_vals_sentcs = audio_stats.(acoustic_metric2plot).sim_vals_2mean_sentc; % Correlation with mean envelope
            case 'median'
                sim_vals_sentcs = audio_stats.(acoustic_metric2plot).sim_vals_sentcs_marg_median;
        end
    case 'shift'

        switch stats_neuro_audio.('shift').circ_shift_type
            case 'concat'
                sim_vals_sentcs = audio_stats.(acoustic_metric2plot).nt_sim_vals_sentcs_shift; % no mean / median selection possible -> both are the same
            case 'average'
                switch acoustic_sim2use
                    case 'mean'
                        sim_vals_sentcs = audio_stats.(acoustic_metric2plot).mean_sim_vals_sentcs_shift;
                    case 'median'
                        sim_vals_sentcs = audio_stats.(acoustic_metric2plot).median_sim_vals_sentcs_shift;
                end
        end       
end
% label_x = sprintf('Mean Sentence Similarity (%s)', acoustic_metric2plot);
label_x = sprintf('Acoustic similarity $\\left(\\rho_{%s}\\right)$', acoustic_sim2use);

if show_shuffled_mean && strcmp(data2plot, 'single_sentences')
    switch acoustic_sim2use
        case 'mean'
            sim_vals_sentcs_all = vertcat(sim_vals_sentcs(:),...
                                          audio_stats.('corr').sim_vals_sentcs_mean,... % shuffled
                                          audio_stats.('corr').sim_vals_2mean_sentc_mean); % mean sentence
        case 'median'
            sim_vals_sentcs_all = vertcat(sim_vals_sentcs(:),...
                                          audio_stats.('corr').sim_vals_sentcs_median,... % shuffled
                                          audio_stats.('corr').sim_vals_2mean_sentc_median); % mean sentence
    end
else
    sim_vals_sentcs_all = sim_vals_sentcs;
end

if strcmp(acoustic_metric2plot, 'corr')
    sim_vals_sentcs     = apply_inverse_fisher_z_transform(sim_vals_sentcs);
    sim_vals_sentcs_all = apply_inverse_fisher_z_transform(sim_vals_sentcs_all);
end

% y-axis
%-------
switch neural_metric2plot
    case {'pooled_linear_model', 'individual_linear_model'}
        sim_stats_sentcs = squeeze(stats_neuro.(data2plot).(neural_metric2plot).slope(con_idx,:,:));
        label_y          = sprintf('%s (slope)', strrep(neural_metric2plot, '_', '-'));
    case 'mixed_model'
        sim_stats_sentcs = squeeze(stats_neuro.(data2plot).(neural_metric2plot).slope(con_idx,:,:));
        label_y          = 'Fixed-effect slope $\left(\beta_{I}\right)$';
    case 'effect_size'
        sim_stats_sentcs = squeeze(stats_neuro.(data2plot).(neural_metric2plot)(con_idx,:,:));
        label_y          = strrep(neural_metric2plot, '_', '-');
end

if show_shuffled_mean
    sim_stats_sentcs_all = horzcat(sim_stats_sentcs,...
                                   stats_neuro.('shuffled').(neural_metric2plot).slope(con_idx, :)',...
                                   stats_neuro.('mean_sentence').(neural_metric2plot).slope(con_idx, :)');
else
    sim_stats_sentcs_all = sim_stats_sentcs;
end

x_min      = min(sim_vals_sentcs_all,[],"all"); 
x_max      = max(sim_vals_sentcs_all,[],"all"); 
xlim_min   = x_min - 0.05 * (x_max - x_min);
xlim_max   = x_max + 0.05 * (x_max - x_min);

y_min      = min(sim_stats_sentcs_all,[],"all"); 
y_max      = max(sim_stats_sentcs_all,[],"all");
ylim_min   = y_min - 0.05 * (y_max - y_min);
ylim_max   = y_max + 0.05 * (y_max - y_min);

% Visualize data
%--------------------------------------------------------------------------

% Define font sizes for plot elements
sgtitleFontSize   = 24; % Super-title for the whole figure
titleFontSize     = 24; % Title for each individual subplot
axisLabelFontSize = 20; % X and Y axis labels (e.g., 'SNR / dB')
tickLabelFontSize = 20; % The numbers on the axes (e.g., -20, -10, 0)
legendFontSize    = 20; % Legend text
textBoxFontSize   = 20; % The correlation text box
markerSize        = 250;

figure('Name', sprintf('Scatter plot: Similarity metric between sentences (%s/%s/%s/%s) and neural tracking (%s)', data2plot, acoustic_metric2plot, acoustic_sim2use, stats_neuro_audio.('shift').circ_shift_type, neural_metric2plot), 'Color', 'w', 'WindowState', 'maximized');
tiledlayout(1, n_sens, 'TileSpacing', 'compact', 'Padding', 'compact');

for sens_idx = 1:n_sens
    ax          = nexttile;
    ax.FontSize = tickLabelFontSize;
    ax.TickLabelInterpreter = 'latex';
    hold(ax, 'on');

    % Initialize legend collectors
    h_leg = [];
    l_leg = {};

    % Plot
    %----------------------------------------------------------------------

    % Apply fading 
    %-------------
    % % Highlight trials (sentences, shifts) with significant difference from
    alphas = alpha_solid*ones(size(sim_vals_sentcs));
    if show_diff2null
        diff2null_idx = squeeze(stats_null_distr.(data2plot).difference2null(con_idx, sens_idx, :));
        % Fade the points that differ from the null distribution
        alphas(~diff2null_idx) = alpha_faded; 
    end

    if ~show_significance
        % Plot all pointds at once
        h = scatter(sim_vals_sentcs(:), sim_stats_sentcs(sens_idx, :)', markerSize, [0.2 0.4 0.6], 'x', 'LineWidth', 2);

        % Apply custom alpha levels
        h.MarkerEdgeAlpha   = 'flat';
        h.AlphaData         = alphas(:);
        h.AlphaDataMapping = 'none';
    else
        pvals               = squeeze(stats_neuro.(data2plot).(neural_metric2plot).pval_adj(con_idx, sens_idx,:));
        valid_idx           = pvals < 0.05;
        h1                  = scatter(sim_vals_sentcs(valid_idx), sim_stats_sentcs(sens_idx, valid_idx)', markerSize, [0.2 0.4 0.6], 'x', 'LineWidth', 2);
        h1.MarkerEdgeAlpha  = 'flat';
        h1.AlphaData        = alphas(valid_idx);
        h1.AlphaDataMapping = 'none';

        h2                  = scatter(sim_vals_sentcs(~valid_idx), sim_stats_sentcs(sens_idx, ~valid_idx)', markerSize, [0.8 0.1 0.1],  'x', 'LineWidth', 2);
        h2.MarkerEdgeAlpha  = 'flat';
        h2.AlphaData        = alphas(~valid_idx);
        h2.AlphaDataMapping = 'none';

        % Collect for legend (checking if they actually contain data)
        if ~isempty(h1.XData)
            h_leg(end+1) = h1; 
            l_leg{end+1} = 'Significant $\beta_{I}$'; 
        end
        if ~isempty(h2.XData)
            h_leg(end+1) = h2; 
            l_leg{end+1} = 'Non-significant $\beta_{I}$'; 
        end
        
    end

    if show_shuffled_mean && strcmp(data2plot, 'single_sentences')
        % Plot shuffled data
        h3 = scatter(sim_vals_sentcs_all(end-1), sim_stats_sentcs_all(sens_idx, end-1), 1*markerSize, [0.9 0.6 0.1], '>', 'LineWidth', 2, 'MarkerFaceColor', [0.9 0.6 0.1]);
        % Plot mean sentence
        h4 = scatter(sim_vals_sentcs_all(end), sim_stats_sentcs_all(sens_idx, end), 1*markerSize, [0.1 0.6 0.3], '>', 'LineWidth', 2, 'MarkerFaceColor', [0.1 0.6 0.3]);
      
        % Add to collectors
        h_leg(end+1) = h3; 
        l_leg{end+1} = 'Shuffled Sentences';
        h_leg(end+1) = h4; 
        l_leg{end+1} = 'Mean-Sentence';
    end

    % Create Legend 
    %--------------
    if ~isempty(h_leg) && sens_idx==1
        legend(h_leg, l_leg, ...
            'Interpreter', 'latex', ...
            'Location', 'southeast', ... 
            'FontSize', legendFontSize);
    end
    % Select p- and correlation value
    switch acoustic_sim2use
        case 'mean'
            r_pearson = stats_neuro_audio.(data2plot).(acoustic_metric2plot).(neural_metric2plot).r_mean(con_idx, sens_idx);
            p_pearson = stats_neuro_audio.(data2plot).(acoustic_metric2plot).(neural_metric2plot).p_mean_adj(con_idx, sens_idx);
        case 'median'
            r_pearson = stats_neuro_audio.(data2plot).(acoustic_metric2plot).(neural_metric2plot).r_median(con_idx, sens_idx);
            p_pearson = stats_neuro_audio.(data2plot).(acoustic_metric2plot).(neural_metric2plot).p_median_adj(con_idx, sens_idx);
    end
    stars_pearson = get_significance_stars(p_pearson);
    
    ptext = ['Pearson correlation', {sprintf('$\\rho \\approx %.2f^{%-3s}$', r_pearson, stars_pearson)}];
          
    text(0.02, 0.98, ptext, ...
             'Units', 'normalized', ... % Position relative to the axes
             'VerticalAlignment', 'top', ...
             'HorizontalAlignment', 'left', ...
             'BackgroundColor', 'white', ...
             'EdgeColor', 'black', ...
             'Interpreter', 'latex', ...
             'FontSize', textBoxFontSize, ...
             'Color', 'k');

    % Set axes properties
    title(sensor_labels{sens_idx}, 'FontSize', titleFontSize, 'Interpreter', 'latex', 'FontWeight', 'bold');
    box on;
    grid on;
    grid minor;
    axis square;
    xlim([xlim_min, xlim_max]);
    ylim([ylim_min, ylim_max]);
    xlabel(label_x, 'FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');

    % Add Y-label only to the first plot
    if sens_idx == 1
        ylabel(label_y, 'FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
    else
        % For all other plots in the row, hide the y-tick labels
        yticklabels([]);
    end
    
    hold(ax, 'off');

end % sensors

% sgtitle(sprintf('Similarity metric between sentences (%s/%s) vs. neural tracking (%s)', ...
%         strrep(data2plot,'_','-'), acoustic_metric2plot, strrep(neural_metric2plot,'_','-')), ...
%         'Interpreter', 'tex', 'FontSize', titleFontSize)
clear sim_vals_sentcs  sim_stats_sentcs label_x label_y r_pearson p_pearson ptext sim_vals_sentcs_all sim_stats_sentcs_all stars_pearson

% Compare mismatched correlation values with matched case
%--------------------------------------------------------------------------

% Sorted Sentences
matched_slopes = stats_neuro.('sorted').(lin_model2plot).slope(con_idx, :);

% Shuffled Sentences
shuffled_slopes = stats_neuro.('shuffled').('mixed_model').slope(con_idx, :);

% Mean-Sentence
mean_sentence_slopes = stats_neuro.('mean_sentence').('mixed_model').slope(con_idx, :);

% Single Sentences
single_sentence_slopes = squeeze(stats_neuro.('single_sentences').('mixed_model').slope(con_idx,:,:))'; % sentences x sensors

% Compute percentages
%--------------------
shuffled_slopes_rel      = shuffled_slopes./matched_slopes*100;
mean_sentence_slopes_rel = mean_sentence_slopes./matched_slopes*100;
% Compute minimum and maximum
single_sentence_slopes   = single_sentence_slopes./matched_slopes*100;

single_sentence_slopes_min = min(single_sentence_slopes, [], 1);
single_sentence_slopes_max = max(single_sentence_slopes, [], 1);

clear matched_slopes shuffled_slopes mean_sentence_slopes ingle_sentence_slopes shuffled_slopes_rel mean_sentence_slopes_rel single_sentence_slopes single_sentence_slopes_min single_sentence_slopes_max

%% Visualize example neural tracking data
%--------------------------------------------------------------------------
% Select subject, condition and sensor
%-------------------------------------
sub_idx = 11;
con_idx = 3;
% Select sensors for plotting
% sens_idx2plot = [1,2,3]; 
sens_idx2plot = [1];

% Select sentences to plot
sentncs_idx2plot = [3, 30, 85]; % ranking descending
% sentncs_idx2plot = [3]; % ranking descending

% Plot data
%--------------------------------------------------------------------------
data2plot      = 'single_sentences';
lin_model2plot = 'mixed_model';

% Axis limits
%------------
sim_data = sim_vals.(data2plot)(sub_idx, con_idx, sens_idx2plot, :, :); 
sim_data = inv_fisher_z_func(sim_data);
y_min    = min(sim_data(:)); 
y_max    = max(sim_data(:));
ylim_min = y_min - 0.05 * (y_max - y_min);
ylim_max = y_max + 0.1 * (y_max - y_min);

local_lims = [0, max(intelligibilities4fit, [], 'all')];

% Other settings
%---------------
% Covers up to 5 sentences
colors  = {[1, 0, 0], [1, 0.5, 0], [0, 0.5, 0], [0, 0, 1], [1, 0, 1]};
markers = {'s', 'd', '>', 'o', '^'};

titleFontSize     = 30; 
axisLabelFontSize = 30; 
tickLabelFontSize = 30; 
legendFontSize    = 30; 
markerSize        = 250;

figure('Name', sprintf('%s / %s', subjectnames{sub_idx}, conditions{con_idx}), 'Color', 'w', 'WindowState', 'maximized');
tiledlayout(1, numel(sens_idx2plot), 'TileSpacing', 'compact', 'Padding', 'compact');
plot_handles = gobjects(1, numel(sentncs_idx2plot)); 

for sens_idx = sens_idx2plot
    ax          = nexttile;
    ax.FontSize = tickLabelFontSize;
    ax.TickLabelInterpreter = 'latex';
    % clear gca
    % ax = gca;
    hold(ax, 'on');

    legend_combined = cell(1,numel(sentncs_idx2plot)); % Use a cell array for multi-line text

    % Plot data
    %---------------------------
    sim_stats = squeeze(stats_neuro.(data2plot).(lin_model2plot).slope(con_idx, sens_idx, :));
    sim_data  = squeeze(sim_vals.(data2plot)(sub_idx, con_idx, sens_idx, :, :));
    sim_data  = inv_fisher_z_func(sim_data);
    
    [sim_stats, sort_idx] = sort(sim_stats, 'descend');
    trial_sel             = sort_idx(sentncs_idx2plot);   

    % Loop over selected trials
    for sel_idx = 1:numel(sentncs_idx2plot)

        % Select data
        trl_idx      = trial_sel(sel_idx);
        sim_data_sel = squeeze(sim_data(:, trl_idx));
        valid_idx   = ~isnan(sim_data_sel);

        plot_handles(sel_idx) = scatter(unique_intelli_values4fit(valid_idx), sim_data_sel(valid_idx), ...
                                        markerSize, ... 
                                        'MarkerFaceColor', colors{sel_idx}, ...
                                        'MarkerEdgeColor', 'k', ...
                                        'MarkerFaceAlpha', 1, ... 
                                        'MarkerEdgeAlpha', 1, ...
                                        'Marker', markers{sel_idx});
   
  
        % Add linear fit
        %--------------
        if strcmp(lin_model2plot, 'mixed_model')
            % Extract the individual slopes and intercepts
            ind_m     = stats_neuro.(data2plot).(lin_model2plot).ind_slopes(sub_idx, con_idx, sens_idx, trl_idx);
            ind_b     = stats_neuro.(data2plot).(lin_model2plot).ind_intercepts(sub_idx, con_idx, sens_idx, trl_idx); 
            y_fit_ind = ind_m * local_lims + ind_b;
            
            plot(local_lims, y_fit_ind, 'Color', [colors{sel_idx}, 0.3], 'LineStyle', '--', 'LineWidth', 3);
            clear ind_m ind_b y_fit_ind
        end

        legend_combined{sel_idx} = sprintf('sentence %d: %s', sentncs_idx2plot(sel_idx), strrep(envelopes_sentence{trl_idx}, 'ü', '\"u'));
       
        clear trl_type trl_idx sim_data_sel 
    end % trials

    % Set axes properties
    hold(ax, 'off');
    % title(sensor_labels{sens_idx}, 'FontSize', titleFontSize);
    box on;
    grid on;
    grid minor;
    axis square;
    xlim([local_lims(1)-diff(local_lims)*5/100, local_lims(2)+diff(local_lims)*5/100]);
    ylim([ylim_min, ylim_max]);
    xlabel('Intelligibility / \%', 'FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');

    % Set the tick marks to exactly match your scaled fitting points
    xticks(unique_intelli_values4fit);
    % Overwrite the text of those ticks with your original labels
    xticklabels(string(unique_intelli_values));

    % Add the legend only to the first tile
    ldg = legend(plot_handles, legend_combined, 'Location', 'northwest', 'FontSize', legendFontSize, 'Interpreter', 'latex');

    % Add Y-label only to the first plot
    if sens_idx == 1
        if apply_abs
            ylabel(sprintf('$\\left| r_{%s} \\right|$', corr_metric), 'FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
        else
            ylabel(sprintf('$r_{%s}$', corr_metric), 'FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
        end
    else
        % For all other plots in the row, hide the y-tick labels
        yticklabels([]);
    end

end % sensors

% sgtitle(sprintf('%s / %s', subjectnames{sub_idx}, conditions{con_idx}), 'Interpreter', 'tex', 'FontSize', titleFontSize)

%% Visualize example neural tracking data II
%--------------------------------------------------------------------------
sub_idx   = 17;
con_idx   = 3;
data2plot = 'sorted';

% Axis limits
%------------
sim_data = sim_vals.(data2plot)(sub_idx, con_idx, :, :); 
sim_data = inv_fisher_z_func(sim_data);
y_min    = min(sim_data(:)); 
y_max    = max(sim_data(:));
ylim_min = y_min - 0.05 * (y_max - y_min);
ylim_max = y_max + 0.1 * (y_max - y_min);

local_lims = [0, max(intelligibilities4fit, [], 'all')];

% Plot data
%--------------------------------------------------------------------------

% Define font sizes for plot elements
titleFontSize     = 75; 
axisLabelFontSize = 30; 
tickLabelFontSize = 24; 
legendFontSize    = 24; 
markerSize        = 500;

figure('color','white','Name', sprintf('%s / %s / %s', subjectnames{sub_idx}, conditions{con_idx}, data2plot),'WindowState', 'maximized');
tiledlayout(1, n_sens, 'TileSpacing', 'compact', 'Padding', 'compact');

for sens_idx = 1:n_sens
    ax          = nexttile;
    ax.FontSize = tickLabelFontSize;
    ax.TickLabelInterpreter = 'latex';
    hold(ax, 'on');

    % Plot control distributions
    %---------------------------
    sim_data  = squeeze(sim_vals.(data2plot)(sub_idx, con_idx, sens_idx, :));
    sim_data  = inv_fisher_z_func(sim_data);
    valid_idx = ~isnan(sim_data);
   
    plot_handles(dt_idx) = scatter(unique_intelli_values4fit(valid_idx), sim_data(valid_idx), ...
                                   markerSize, ... 
                                   'MarkerFaceColor', 'k', ...
                                   'MarkerEdgeColor', 'k', ...
                                   'MarkerFaceAlpha', 1, ... 
                                   'MarkerEdgeAlpha', 1, ...
                                   'Marker', 'x', ...
                                   'LineWidth', 5);

    % Set axes properties
    % title(sensor_labels{sens_idx}, 'FontSize', titleFontSize);
    set(gca, 'LineWidth', 2);
    box on;
    grid on;
    % grid minor;
    axis square;
    xlim([local_lims(1)-diff(local_lims)*5/100, local_lims(2)+diff(local_lims)*5/100]);
    ylim([ylim_min, ylim_max]);

    if ismember(sens_idx, 3)
        xlabel('Intelligibility / \%', 'FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
    end

    % Set the tick marks to exactly match your scaled fitting points
    xticks(unique_intelli_values4fit);
    % Overwrite the text of those ticks with your original labels
    xticklabels(string(unique_intelli_values));
    
    % Add Y-label only to the first plot
    if ismember(sens_idx, [1,2,3])
        if apply_abs
            ylabel(sprintf('$\\left| r_{%s} \\right|$', corr_metric), 'FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
        else
            ylabel(sprintf('$r_{%s}$', corr_metric), 'FontSize', axisLabelFontSize, 'FontWeight', 'bold', 'Interpreter', 'latex');
        end
    else
        % For all other plots in the row, hide the y-tick labels
        yticklabels([]);
    end

    % Define the label text
    txt = sensor_labels{sens_idx};
    
    % Create the textbox in the upper left 
    text(ax, 0.05, 0.99, txt, ...
        'Units', 'normalized', ...       
        'VerticalAlignment', 'top', ... 
        'HorizontalAlignment', 'left', ...
        'FontSize', titleFontSize, ...  
        'FontWeight', 'bold', ...        
        'BackgroundColor', 'none', ...  
        'Interpreter', 'latex', ...
        'EdgeColor', 'none');           
    
    hold(ax, 'off');

end % sensors