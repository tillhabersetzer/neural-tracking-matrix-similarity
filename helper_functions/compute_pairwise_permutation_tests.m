function results = compute_pairwise_permutation_tests(distributions, contrasts, n_perms, tail, is_paired)
%--------------------------------------------------------------------------
% Till Habersetzer, 10.10.2025
% Communication Acoustics, CvO University Oldenburg
% till.habersetzer@uol.de
%
%   DESCRIPTION:
%   Performs permutation testing on specific pairs of rows within a 
%   data matrix.
%   It supports both **Independent** (shuffling) and **Paired** 
%   (sign-flipping) permutation tests using a fast, vectorized engine.
%
%   WORKFLOW:
%   1.  **Configuration:** Receives the data matrix and a list of pairs 
%       (contrasts) to compare.
%   2.  **Data Preparation:** For each contrast defined in 'contrasts':
%       - Extracts the two rows.
%       - If **Paired**: Checks for equal length and removes columns where 
%         either value is NaN (strict 1-to-1 matching).
%       - If **Independent**: Removes NaNs individually from each vector.
%   3.  **Vectorized Permutation:** 
%       - **Paired**: Generates random sign flips (+1/-1) for difference vectors.
%       - **Independent**: Shuffles group labels across the pooled data.
%   4.  **P-Value Calculation:** Computes the p-value using the 
%       (Count + 1) / (Perms + 1) method for statistical robustness.
%
%   INPUTS:
%   -   distributions: [N_rows x N_samples] numeric matrix containing the data.
%                      Each row represents a distribution/condition.
%   -   contrasts:     [K x 2] matrix of integers. Each row defines a pair
%                      of indices from 'distributions' to compare.
%                      Example: [1 2; 1 3] compares Row 1vs2 and Row 1vs3.
%   -   n_perms:       Scalar integer (e.g., 10000).
%   -   tail:          String: 'two', 'right', or 'left'.
%   -   is_paired:     Boolean (true/false). 
%          - true:  Performs Sign-Flipping test.
%          - false: Performs Shuffling test.
%
%   OUTPUTS:
%   -   results:       Struct containing matrices. 
%         - .p_values:        [K x 1] vector of p-values.
%         - .diffs_observed:  [K x 1] vector of observed mean differences.
%         - .diffs_permuted:  [K x n_perms] matrix of null distributions.
%         - .n_samples_distr: [K x 2] matrix of sample counts.
%         - .contrast_indices: [K x 2] copy of input contrasts for reference.
%
%--------------------------------------------------------------------------

% Input Validation
%-----------------
narginchk(4, 5);
if nargin < 5
    is_paired = false; % Default to independent
end

if ~isnumeric(distributions) || ~ismatrix(distributions)
    error('Input "distributions" must be a numeric matrix.');
end

if ~isnumeric(contrasts) || size(contrasts, 2) ~= 2
    error('Input "contrasts" must be a K x 2 matrix of indices.');
end

% Validate indices
n_rows = size(distributions, 1);
if max(contrasts(:)) > n_rows || min(contrasts(:)) < 1
    error('Indices in "contrasts" exceed dimensions of "distributions".');
end

validatestring(tail, {'two', 'right', 'left'});

n_contrasts = size(contrasts, 1);
type_str = "Independent";
if is_paired
    type_str = "Paired"; 
end

fprintf('Performing %d defined %s pairwise permutation tests...\n', n_contrasts, type_str);

% Initialize Output Matrices
%---------------------------
results                  = struct();
results.tail             = tail;
results.n_perms          = n_perms;
results.is_paired        = is_paired;
results.contrast_indices = contrasts; % Store metadata

% Pre-allocate for speed
results.p_values        = nan(n_contrasts, 1);
results.diffs_observed  = nan(n_contrasts, 1);
results.diffs_permuted  = nan(n_contrasts, n_perms);
results.n_samples_distr = nan(n_contrasts, 2);

% Loop Over Contrasts 
%--------------------
for contr_idx = 1:n_contrasts
    
    % Get row indices for this specific contrast
    idx1 = contrasts(contr_idx, 1);
    idx2 = contrasts(contr_idx, 2);
    
    % Extract data (ensure row vectors)
    data1 = distributions(idx1, :);
    data2 = distributions(idx2, :);
    
    % Data Preparation
    %----------------------------------------------------------------------
    if is_paired
        % PAIRED: Strict 1-to-1 matching (e.g. Subject i in Cond A vs Cond B)
        if length(data1) ~= length(data2)
            error('Paired test failed for Row %d vs %d: Vectors must have equal length.', idx1, idx2);
        end
        
        % Remove pair if EITHER is NaN
        valid_mask = ~isnan(data1) & ~isnan(data2);
        distr1     = data1(valid_mask);
        distr2     = data2(valid_mask);
        
    else
        % INDEPENDENT: Remove NaNs individually (lengths can differ)
        distr1 = data1(~isnan(data1));
        distr2 = data2(~isnan(data2));
    end
    
    % Safety check for empty data
    if isempty(distr1) || isempty(distr2)
        warning('Contrast Row %d vs %d has no valid data. Skipping.', idx1, idx2);
        continue;
    end
    
    % Permutation Test (Vectorized)
    %----------------------------------------------------------------------
    if is_paired
        % Paired: Sign Flipping
        %----------------------
        % Observed Difference
        data_diffs = distr2 - distr1; 
        obs_diff   = mean(data_diffs);
        
        % Generate Random Signs (+1 or -1)
        % Size: [N_samples x N_perms]
        random_signs = (rand(length(data_diffs), n_perms) > 0.5) * 2 - 1;
        
        % Apply Signs and Average
        % (diff * sign) effectively flips the subtraction A-B to B-A
        permuted_diffs = mean(data_diffs(:) .* random_signs, 1);
        
    else
        % Independent: Shuffling
        %-----------------------
        % Observed Difference
        obs_diff = mean(distr2) - mean(distr1);
        
        % Pool Data
        distr_all = [distr1, distr2];
        n1        = length(distr1);
        n_total   = length(distr_all);
        
        % Create a matrix where each column is a shuffled dataset
        [~, rand_idx] = sort(rand(n_total, n_perms), 1); % each column is sorted along rows
        shuffled_mat  = distr_all(rand_idx);
        
        % Split into two matrices for the new groups
        perm_mean1     = mean(shuffled_mat(1:n1, :), 1);
        perm_mean2     = mean(shuffled_mat(n1+1:end, :), 1);
        permuted_diffs = perm_mean2 - perm_mean1;
    end
    
    % P-value calculation
    %----------------------------------------------------------------------
    % Using (Count + 1) / (Perms + 1) for statistical robustness
    switch tail
        case 'two'
            count = sum(abs(permuted_diffs) >= abs(obs_diff));
        case 'right'
            count = sum(permuted_diffs >= obs_diff);
        case 'left'
            count = sum(permuted_diffs <= obs_diff);
    end
    
    p_val = (count + 1) / (n_perms + 1);
    
    % Store results in Matrix Rows
    %----------------------------------------------------------------------
    results.p_values(contr_idx)           = p_val;
    results.diffs_observed(contr_idx)     = obs_diff;
    results.diffs_permuted(contr_idx, :)  = permuted_diffs;
    results.n_samples_distr(contr_idx, :) = [length(distr1), length(distr2)];
    
    fprintf('Contrast Row %d vs %d: p = %.5f\n', idx2, idx1, p_val);
end % Loop over contrasts

fprintf('Permutation tests complete.\n');
end