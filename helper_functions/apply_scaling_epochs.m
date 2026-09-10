function [epochs] = apply_scaling_epochs(epochs,type)
%--------------------------------------------------------------------------
% Till Habersetzer, 14.08.2025
% Communication Acoustics, CvO University Oldenburg
% till.habersetzer@uol.de 
%
% Description:
%   Performs max-abs scaling on epoched data.
%
% This function scales each trial by dividing all values by the maximum
% absolute value computed across all trials combined. This is a common
% preprocessing step to bound data within the range of **[-1, 1]**. The
% exact scaling strategy depends on the specified data `type`.
%
%   - For **'audio'** and **'biochannels'**data, it finds the maximum 
%     absolute value for each channel independently across all trials and 
%     uses that channel-specific value for scaling.
%
%   - For **neurophysiological data** ('meg', 'eeg', 'ear_eeg'), it finds a
%     single grand maximum absolute value across all specified channels
%     and all time points. For 'meg' data, this process is performed
%     separately for magnetometers and gradiometers.
%
%   - Applies the calculated maximum absolute value to scale each individual
%     trial using the formula: x_scaled = x / max_abs_value.
% 
% Inputs:
%   epochs (cell or struct): The input data structure.
%       - If `type` is 'audio', `epochs` must be a cell array 
%         {1 x n_trials}, where each cell contains a [channels x samples] matrix.
%       - If `type` is neurophysiological, `epochs` must be a struct with fields:
%         .trial - Cell array {1 x n_trials} of trial data.
%         .label - Cell array {n_channels x 1} of channel names.
%
%   type (char/string): Specifies the data type and scaling strategy. 
%                       Supported types: 'audio', 'meg', 'eeg', 'ear_eeg'.
%
% Outputs:
%   epochs (cell or struct): The input data structure with the trial data
%                            replaced by its max-abs scaled version.
%--------------------------------------------------------------------------

if strcmp(type,'audio')
     % For audio, scale each channel independently based on its own max value.
    
     % Concatenate all trials to find the max absolute value
    epochs_concat = horzcat(epochs{:});

    % Calculate scaling factor for each channel (dimension 2)
    scaling = max(abs(epochs_concat), [], 2);

    % Apply scaling to each trial
    n_trials = length(epochs);
    for trl_idx = 1:n_trials
        epochs{trl_idx} = epochs{trl_idx} ./ scaling;
    end

elseif ismember(type, {'meg', 'eeg', 'ear_eeg', 'biochannels'})
        
    % For neuro data, determine which channel groups to scale together.
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
        
        % Calculate a SINGLE grand scaling factor for all data in this group
        scaling = max(abs(epochs_concat(idx_sensor, :)), [], 'all');
   
        % Apply this single mean/std to the relevant channels in each trial
        for trl_idx = 1:n_trials
            epochs.trial{trl_idx}(idx_sensor, :) = epochs.trial{trl_idx}(idx_sensor, :) ./ scaling;
        end
    end

else
    error('Unexpected data type requested (%s)!', type);
end

end % end of function