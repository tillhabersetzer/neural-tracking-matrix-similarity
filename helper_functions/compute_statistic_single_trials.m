function [stats] = compute_statistic_single_trials(y, pred, dim, corr_metric, circ_shift, fs_audio, shift_target)
%--------------------------------------------------------------------------
% Till Habersetzer, 29.08.2025 (revised: 07.07.2026)
% Communication Acoustics, CvO University Oldenburg
% till.habersetzer@uol.de 
%
%   DESCRIPTION:
%   This function generates an exhaustive null distribution of model
%   performance scores by systematically correlating all mismatched pairs of
%   true (`y`) and predicted (`pred`) trials. This provides a robust,
%   complete permutation-based baseline for statistical significance.
%
%   The function computes all N*(N-1) possible mismatched pairings, where N
%   is the number of trials. An optional circular time-shift of either the 
%   true signal (`y`, 'audio') or the predicted signal (`pred`, 'prediction') 
%   can be applied to create a stricter null baseline that also controls for 
%   temporal contingency.
%
%   SYNTAX:
%   stats = compute_statistic_single_trials(y, pred, dim, corr_metric, circ_shift)
%   stats = compute_statistic_single_trials(y, pred, dim, corr_metric, circ_shift, fs_audio)
%   stats = compute_statistic_single_trials(y, pred, dim, corr_metric, circ_shift, fs_audio, shift_target)
%
%   DEPENDENCIES:
%   - Requires an evaluation function (e.g., `mTRFevaluate`).
%
%   INPUTS:
%   y            (cell array):  1xN cell array of the true signal for each trial.
%   pred         (cell array):  1xN cell array of the model-predicted signal.
%   dim          (integer):     Dimension for correlation (e.g., 1=columns, 2=rows).
%   corr_metric  (char/string): Correlation type ('Pearson' or 'Spearman').
%   circ_shift   (logical):     If true, applies a random circular time-shift.
%   fs_audio     (int, opt):    Sampling rate in Hz. Ensures shift is >= 1s.
%   shift_target (char/string): Target for circular shift: 'audio' or 'prediction'.
%
%   OUTPUTS:
%   stats       (struct):      A structure containing the null distribution results.
%     .r            (vector):  A 1-by-(N*(N-1)) vector of all correlation scores.
%     .err          (vector):  A 1-by-(N*(N-1)) vector of all error scores.
%     .r_mean       (scalar):  The mean of the correlation distribution.
%     .r_std        (scalar):  The standard deviation of the distribution.
%     .r_prctile    (vector):  Key percentiles [2.5, 25, 50, 75, 97.5].
%     .n_perms_used (scalar):  The total number of pairs computed (N*(N-1)).
%--------------------------------------------------------------------------

%% Input Validation and Preparation
%--------------------------------------------------------------------------

% Handle optional arguments (ensure backward compatibility)
if nargin < 7 || isempty(shift_target)
    shift_target = 'prediction';
else
    if ~any(strcmpi(shift_target, {'audio', 'prediction'}))
        error('Invalid shift_target: ''%s''. Must be either ''audio'' or ''prediction''.', shift_target);
    end
end
if nargin < 6 || isempty(fs_audio)
    fs_audio = []; 
end
if nargin < 5 || isempty(circ_shift)
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

% Initialization
n_perms = n_trials * (n_trials - 1);

% Pre-allocate the final n_trials x (n_trials-1) result matrices
r   = zeros(1, n_perms);
err = zeros(1, n_perms);

% Pre-calculate all trial lengths 
lengths_y    = cellfun(@length, y);
lengths_pred = cellfun(@length, pred);

% Define the interval for progress updates (every 10%)
update_step = max(1, floor(n_perms / 10));

%% Compute Null Distribution
%--------------------------------------------------------------------------

fprintf('Computing null distribution for %d mismatched pairs...\n', n_perms);

% Compute all combinations
%-------------------------
[X, Y]    = ndgrid(1:n_trials, 1:n_trials);
all_pairs = [X(:), Y(:)];
all_pairs = all_pairs(all_pairs(:,1) ~= all_pairs(:,2), :);

if size(all_pairs, 1) ~= n_perms
    error('Unexpected number of derangements!')
end

% Compute Correlations
for p_idx = 1:n_perms

    % First, determine the lengths of each segment to pre-allocate memory
    y_idx    = all_pairs(p_idx, 1);
    pred_idx = all_pairs(p_idx, 2);
        
    % Get minimum length
    len = min(lengths_y(y_idx), lengths_pred(pred_idx));

    % Determine the amount of circular shift
    %---------------------------------------
    if circ_shift

        % Select the target variable to shift based on input
        if strcmpi(shift_target, 'audio')
            target_trial = y{y_idx};
            shift_len    = lengths_y(y_idx);
        else
            target_trial = pred{pred_idx};
            shift_len    = lengths_pred(pred_idx);
        end

        % Apply shift
        if shift_len > 1
            if ~isempty(fs_audio) && shift_len > fs_audio
                idx_shift = randi([fs_audio, shift_len - 1]);
            else
                idx_shift = randi(shift_len - 1);
            end
            target_trial = circshift(target_trial, idx_shift, 2);
        end

        % Assign target segments to the correct output
        if strcmpi(shift_target, 'audio')
            y_cropped    = target_trial(1:len);
            pred_cropped = pred{pred_idx}(1:len);
        else
            y_cropped    = y{y_idx}(1:len);
            pred_cropped = target_trial(1:len);
        end
    else
        % No shift applied
        y_cropped    = y{y_idx}(1:len);
        pred_cropped = pred{pred_idx}(1:len);
    end

    % Perform the correlation computation
    %----------------------------------------------------------------------
    [r(p_idx), err(p_idx)] = mTRFevaluate(y_cropped, pred_cropped, 'dim', dim, 'corr', corr_metric);

    % Update progress indicator
    %----------------------------------------------------------------------
    if mod(p_idx, update_step) == 0 || p_idx == n_perms
        fprintf('\nProcessed %d / %d unmatched permutations.', p_idx, n_perms);
    end

end % Loop over permutations
fprintf('\nPermutation analysis complete.\n');

% Package results into the output struct
%--------------------------------------------------------------------------
stats              = struct();
stats.r            = r;
stats.err          = err;

% NOTE: When computing mean and standard deviation of bounded correlation 
% values (like Pearson's r), ideally a Fisher z-transformation should be 
% applied first (z = atanh(r)) to normalize the distribution before taking 
% the mean/std, and then transformed back (r = tanh(z)) if needed.
stats.r_mean       = mean(r);
stats.r_std        = std(r);

stats.r_prctile    = prctile(r, [2.5, 25, 50, 75, 97.5]);
stats.n_perms_used = n_perms;

end