function [stats] = compute_statistic_concatenated_trials_deranged(y, pred, n_perms, dim, corr_metric, circ_shift, fs_audio, shift_target)
%--------------------------------------------------------------------------
% Till Habersetzer, 26.08.2025
% Communication Acoustics, CvO University Oldenburg
% till.habersetzer@uol.de 
%
%   Revision History:
%   14.05.2026 - Added optional 'shift_target' parameter to allow circular 
%                shifting of either the 'audio' or 'prediction' signals. 
%                Default fallback added for backward compatibility.
%
%   Description:
%   Computes a null distribution using up to two layers of randomization:
%
%   1. Trial Mismatching (Derangement): This step is always active. On every
%      permutation, a derangement of trial indices is generated. A derangement
%      is a permutation with no fixpoints, ensuring that the actual signal from
%      one trial (y{i}) is always compared against the predicted signal from a
%      *different* trial (pred{j}, where j~=i). This is the primary method
%      for creating the null hypothesis condition.
%
%   2. Circular Shift (Optional): If the 'circ_shift' flag is set to true,
%      an *additional* randomization step is performed. The specified target
%      signal ('prediction' or 'audio') of the mismatched pairing is 
%      circularly shifted by a random amount. This further decorrelates 
%      the signals.
%
%   The probability of repeating a specific derangement is negligible, so
%   the sampling for the null distribution is considered sufficiently random.
%
%   Inputs:
%   y             (cell array):  A cell array where each cell contains the actual
%                                signal (ground truth) for a single trial.
%   pred          (cell array):  A cell array where each cell contains the
%                                model-predicted signal for a single trial.
%   n_perms       (integer):     The number of permutations to run.
%   dim           (integer):     The dimension along which to compute the
%                                correlation and error (e.g., 1 for columns,
%                                2 for rows).
%   corr_metric   (char):        The type of correlation to compute, e.g.,
%                                'Pearson' or 'Spearman'. This is passed directly
%                                to the mTRFevaluate function.
%   Optional Inputs:
%   circ_shift    (logical):     If true, applies a random circular shift to each
%                                selected trial. Default: false.
%   fs_audio      (integer):     The audio sampling frequency in Hz. If
%                                provided with circ_shift=true, the random shift
%                                will be at least 1 second long.
%   shift_target  (string):      Specifies which signal to shift: 'prediction' or 'audio'.
%                                Default: 'prediction'.
%
%   Outputs:
%   stats         (struct):      A structure containing the null distribution results.
%     .r            (vector):  A 1-by-n_perms vector of correlation scores.
%     .err          (vector):  A 1-by-n_perms vector of error scores.
%     .r_mean       (scalar):  The mean of the correlation distribution.
%     .r_std        (scalar):  The standard deviation of the distribution.
%     .r_prctile    (vector):  Percentiles [2.5, 25, 50, 75, 97.5] of the
%                              correlation distribution.
%     .n_perms_used (scalar):  The number of permutations run.
%
%--------------------------------------------------------------------------

% Handle optional arguments (ensure backward compatibility)
if nargin < 8 || isempty(shift_target)
    shift_target = 'prediction';
else
    % Validate that the provided shift_target is strictly 'audio' or 'prediction'
    if ~any(strcmpi(shift_target, {'audio', 'prediction'}))
        error('Invalid shift_target: ''%s''. Must be either ''audio'' or ''prediction''.', shift_target);
    end
end
if nargin < 7 || isempty(fs_audio)
    fs_audio = []; 
end
if nargin < 6 || isempty(circ_shift)
    circ_shift = false;
end

% Ensure all cell entries are 1xN row vectors for consistent indexing
pred = cellfun(@(x) x(:).', pred, 'UniformOutput', false);
y    = cellfun(@(x) x(:).', y,    'UniformOutput', false);

% This ensures the cell array is a 1xN row, not an Nx1 column.
pred = pred(:).';
y    = y(:).';

% Get the total number of trials available
n_trials = length(y);
if n_trials ~= length(pred)
    error('The number of trials in y and pred must be the same!');
end

% Initialize data structure for correlation values and errors
r   = zeros(1, n_perms); 
err = zeros(1, n_perms);

% Pre-calculate all trial lengths 
lengths_y    = cellfun(@length, y);
lengths_pred = cellfun(@length, pred);

% Define the interval for progress updates (every 10%)
update_step = max(1, floor(n_perms / 10));

%% Compute Permutations
%--------------------------------------------------------------------------
% This loop builds a null distribution by correlating mismatched trials.
fprintf('Running %d permutations...\n', n_perms);

for p_idx = 1:n_perms
    
    % Generate a derangement of trial indices
    %----------------------------------------------------------------------
    % This ensures no trial is ever compared against its correct counterpart
    deranged_indices = generate_derangement(n_trials);
    
    % Prepare data segments for concatenation
    %----------------------------------------------------------------------
    % Determine the minimum length for each trial pairing based on the target
    if strcmpi(shift_target, 'audio')
        segment_lengths = min(lengths_y(deranged_indices), lengths_pred);
    else
        segment_lengths = min(lengths_y, lengths_pred(deranged_indices));
    end
    
    % Prepare all segments in cell arrays before concatenation
    y_segments    = cell(1, n_trials);
    pred_segments = cell(1, n_trials);
    
    for trl_idx = 1:n_trials
        deranged_idx = deranged_indices(trl_idx);
        len          = segment_lengths(trl_idx);
        
        % Extract the base trials (Derange the chosen target)
        if strcmpi(shift_target, 'audio')
            y_trial    = y{deranged_idx};
            pred_trial = pred{trl_idx};
        else
            y_trial    = y{trl_idx};
            pred_trial = pred{deranged_idx};
        end
        
        % Apply optional circular shift to the specified target
        if circ_shift && len > 1
            if ~isempty(fs_audio) && len > fs_audio
                idx_shift = randi([fs_audio, len-1]);
            else
                idx_shift = randi(len-1);
            end
            
            % Shift either the audio or the prediction
            if strcmpi(shift_target, 'audio')
                y_trial = circshift(y_trial, idx_shift, 2);
            else
                pred_trial = circshift(pred_trial, idx_shift, 2);
            end
        end
        
        % Store the prepared (truncated and shifted) segments
        y_segments{trl_idx}    = y_trial(1:len);
        pred_segments{trl_idx} = pred_trial(1:len);
        
    end % Loop over trials
    
    % Concatenate all prepared segments at once
    %----------------------------------------------------------------------
    y_concat    = [y_segments{:}];
    pred_concat = [pred_segments{:}];
  
    % Perform the correlation computation
    %----------------------------------------------------------------------
    [r(p_idx), err(p_idx)] = mTRFevaluate(y_concat, pred_concat, 'error', 'mse', 'dim', dim, 'corr', corr_metric);
    
    % Update progress indicator
    %----------------------------------------------------------------------
    % Check if the current iteration is an update point or the very last one.
    if mod(p_idx, update_step) == 0 || p_idx == n_perms
        fprintf('\nProcessed %d / %d deranged permutations.', p_idx, n_perms);
    end
    
end % Loop over permutations

fprintf('\nPermutation analysis complete.\n');

% Package results into the output struct
%--------------------------------------------------------------------------
% Compute and store summary statistics for this subset size 
stats              = struct();
stats.r            = r;
stats.err          = err;
stats.r_mean       = mean(r);
stats.r_std        = std(r);
stats.r_prctile    = prctile(r, [2.5, 25, 50, 75, 97.5]);
stats.n_perms_used = n_perms;

%% LOCAL FUNCTIONS
%--------------------------------------------------------------------------
% Functions defined below this point are only accessible by this script

function deranged_indices = generate_derangement(n)
%--------------------------------------------------------------------------
% Till Habersetzer, 26.08.2025
% Communication Acoustics, CvO University Oldenburg
% till.habersetzer@uol.de 
%
%   Description:
%   Generates a derangement of integers from 1 to n. A derangement is a
%   permutation of the elements of a set such that no element appears in
%   its original position (i.e., for the output vector 'p', p(i) ~= i for
%   all i).
%
%   This function uses rejection sampling: it generates random permutations
%   until it finds one that satisfies the derangement condition.
%
%   Formula:
%   The number of possible derangements (!n), or the subfactorial, can be
%   closely approximated by the formula: !n = round(factorial(n) / exp(1)).
%
%   Inputs:
%   n (integer):         The number of elements to permute.
%
%   Outputs:
%   deranged_indices (vector): A 1xN row vector containing the deranged
%                              permutation of integers from 1 to n.
%--------------------------------------------------------------------------

deranged_indices = 1:n;

% Loop until no index matches its original position (e.g., p(i) ~= i for all i).
while any(deranged_indices == 1:n)
    deranged_indices = randperm(n);
end

end % End of local function

end % End of main function