function [epochs] = apply_zscore_epochs(epochs,type)
%--------------------------------------------------------------------------
% Till Habersetzer, 30.07.2025 (Corrected 14.08.2025)
% Communication Acoustics, CvO University Oldenburg
% till.habersetzer@uol.de 
%
% Description:
%   Performs z-score normalization (standardization) on epoched data.
%
% This function standardizes each trial using statistics (mean and standard
% deviation) computed across all trials combined. This is a common
% preprocessing step to ensure data are on a comparable scale. The
% normalization strategy depends on the specified data `type`.
%
%   - 'audio': Each channel is normalized independently using its own mean
%     and standard deviation, calculated across all time and trials.
%
%   - 'eeg', 'ear_eeg': A single grand mean and standard deviation are
%     calculated across ALL channels, time, and trials. This single
%     pair of values is used to normalize all channels.
%
%   - 'meg': Magnetometers and gradiometers are treated as two separate
%     groups. A grand mean/std is computed for all magnetometers, and
%     another for all gradiometers. Each sensor group is then normalized
%     by its respective statistics.
%
%   - 'biochannels': Each channel (e.g., EOG, ECG) is normalized
%     independently using its own mean and standard deviation, similar to
%     the 'audio' strategy.
%
%   - Applies the calculated global statistics to normalize each individual
%     trial using the formula: z = (x - global_mean) / global_std.
%
% Inputs:
%   epochs (cell or struct): The input data structure.
%       - If `type` is 'audio', `epochs` must be a cell array 
%         {1 x n_trials}, where each cell contains a [channels x samples] matrix.
%       - If `type` is neurophysiological, `epochs` must be a struct with fields:
%         .trial - Cell array {1 x n_trials} of trial data.
%         .label - Cell array {n_channels x 1} of channel names.
%
%   type (char/string): Specifies the data type and normalization strategy. 
%                       Supported types: 'audio', 'meg', 'eeg', 'ear_eeg'.
%
% Outputs:
%   epochs (cell or struct): The input data structure with the trial data
%                            replaced by its z-scored version.
%--------------------------------------------------------------------------

if strcmp(type,'audio')
    % For audio, normalize each channel independently based on its own stats.
    
    % Concatenate all trials to compute statistics
    epochs_concat = horzcat(epochs{:});

    % Calculate mean and std for each channel (dimension 2)
    mean_val = mean(epochs_concat, 2);
    std_val  = std(epochs_concat, 0, 2);

    % Apply z-scoring to each trial
    n_trials = length(epochs);
    for trl_idx = 1:n_trials
        epochs{trl_idx} = (epochs{trl_idx} - mean_val) ./ std_val;
    end

elseif ismember(type, {'meg', 'eeg', 'ear_eeg','biochannels'})
        
    % For neuro data, determine which channel groups to normalize together.
    if strcmp(type, 'meg')
        % For MEG, separate magnetometers and gradiometers
        idx_mag = endsWith(epochs.label, '1');
        if sum(idx_mag) ~= 102; error('Unexpected number of magnetometer channels (%i)!', sum(idx_mag)); end
        
        idx_grad = endsWith(epochs.label, {'2', '3'});
        if sum(idx_grad) ~= 204; error('Unexpected number of gradiometer channels (%i)!', sum(idx_grad)); end
        
        % Each column represents a sensor type to be processed
        idx_sensors = [idx_mag, idx_grad];

    elseif ismember(type,{'eeg','ear_eeg'})
        % For EEG, process all channels together
        idx_sensors = true(length(epochs.label), 1);

    elseif strcmp(type, 'biochannels')
        % Process each channel separately
        idx_sensors = logical(eye(length(epochs.label)));
    end

    % Concatenate all trials to compute statistics
    epochs_concat = horzcat(epochs.trial{:});
    n_trials      = length(epochs.trial);

    % Loop through sensor groups (1 for EEG, 2 for MEG, n_channels=3 for biochannels)
    for sens_idx = 1:size(idx_sensors, 2)
        idx_sensor = idx_sensors(:, sens_idx);
        
        % Calculate a SINGLE grand mean and std for all data in this group
        mean_val = mean(epochs_concat(idx_sensor, :), 'all');
        std_val  = std(epochs_concat(idx_sensor, :), 0, 'all');
      
        % Apply this single mean/std to the relevant channels in each trial
        for trl_idx = 1:n_trials
            epochs.trial{trl_idx}(idx_sensor, :) = (epochs.trial{trl_idx}(idx_sensor, :) - mean_val) ./ std_val;
        end
    end

else
    error('Unexpected data type requested (%s)!', type);
end

end % end of function