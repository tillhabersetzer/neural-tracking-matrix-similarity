function [stats_corrected, n_corrected] = apply_multiple_comparison_correction(p_values_full, method, scope, dim_idx, subset_mask)
%--------------------------------------------------------------------------
% Till Habersetzer, 11.11.2025
% Communication Acoustics, CvO University Oldenburg
% till.habersetzer@uol.de 
%
%   Description:
%   Adjusts p-values for multiple comparisons using conservative FWER 
%   controls (Bonferroni, Bonferroni-Holm) or the False Discovery Rate 
%   (Benjamini-Hochberg). Designed for multi-dimensional neural data to 
%   define the "statistical family" across flexible dimensions.
%
%   Correction Methods:
%   'bonferroni'      - Controls FWER. Conservative.
%                       Formula: p_adj = min(p_raw * n, 1.0)
%   'bonferroni-holm' - Controls FWER. Sequentially rejective (step-down).
%                       More power than Bonferroni. Calls: bonf_holm().
%   'fdr'             - Controls FDR. 
%                       Procedure: Benjamini-Hochberg (requires mafdr).
%
%   Operation Scopes:
%   'all'           - Global: Pools all non-NaN values in the matrix 
%                     into a single correction family.
%   'per-condition' - Independent: Iterates through dim_idx, correcting 
%                     each slice separately.
%   'subset'        - Targeted: Applies correction only to indices 
%                     specified by 'subset_mask' along 'dim_idx'.
%
%   Syntax:
%   [p_adj, n] = apply_multiple_comparison_correction(p_mat, 'fdr', 'all')
%   [p_adj, n] = apply_multiple_comparison_correction(p_mat, 'bonferroni-holm', ...
%                                                      'per-condition', 2)
%
%   Inputs:
%   p_values_full   - [matrix] Array of raw p-values (handles NaNs).
%   method          - [string] 'bonferroni', 'bonferroni-holm', or 'fdr'.
%   scope           - [string] 'all', 'per-condition', or 'subset'.
%   dim_idx         - [scalar] The dimension representing conditions.
%   subset_mask     - [logical] Vector used only for 'subset' scope.
%
%   Outputs:
%   stats_corrected - [matrix] Corrected p-values (same shape as input).
%   n_corrected     - [vector/scalar] Number of valid tests per family.
%--------------------------------------------------------------------------

stats_corrected = p_values_full;
n_corrected     = 0;

% Validate Method and Toolbox
is_fdr  = strcmpi(method, 'fdr');
is_bonf = strcmpi(method, 'bonferroni');
is_holm = strcmpi(method, 'bonferroni-holm');

if ~is_fdr && ~is_bonf && ~is_holm
    error('Invalid method: Use ''bonferroni'', ''bonferroni-holm'', or ''fdr''.');
end

% Check for Bioinformatics Toolbox if FDR is requested
if strcmpi(method, 'fdr') && ~exist('mafdr', 'file')
    error('Bioinformatics Toolbox missing. Cannot perform FDR correction.');
end

% 1. Global correction: Everything that isn't NaN
%------------------------------------------------
if strcmpi(scope, 'all')
    valid_mask  = ~isnan(p_values_full);
    p_valid     = p_values_full(valid_mask);
    n_tests     = numel(p_valid);
    n_corrected = n_tests;
    
    if n_tests > 0
        if is_bonf
            stats_corrected(valid_mask) = min(p_valid .* n_tests, 1.0);
        elseif is_holm
            [stats_corrected(valid_mask), ~] = bonf_holm(p_valid);
        else
            stats_corrected(valid_mask) = mafdr(p_valid, 'BHFDR', true);
        end
    end
    fprintf('%s applied globally (n = %d).\n', method, n_corrected);

% 2. Per-condition correction: Loop through the specified dimension
%------------------------------------------------------------------
elseif strcmpi(scope, 'per-condition')
    n_slices    = size(p_values_full, dim_idx);
    indexer     = repmat({':'}, 1, ndims(p_values_full));
    n_corrected = zeros(n_slices, 1);
    
    for i = 1:n_slices
        indexer{dim_idx} = i;
        p_slice          = p_values_full(indexer{:});
        
        valid_mask     = ~isnan(p_slice);
        p_valid        = p_slice(valid_mask);
        n_tests        = numel(p_valid);
        n_corrected(i) = n_tests;
        
        if n_tests > 0
            if is_bonf
                p_slice(valid_mask) = min(p_valid .* n_tests, 1.0);
            elseif is_holm
                [p_slice(valid_mask), ~] = bonf_holm(p_valid);
            else
                p_slice(valid_mask) = mafdr(p_valid, 'BHFDR', true);
            end
            stats_corrected(indexer{:}) = p_slice;
        end

        fprintf('%s applied per index along dim %d (n = %d).\n', method, dim_idx, n_tests);
    end

% 3. Subset correction: Correct only specific indices on dim_idx
%---------------------------------------------------------------
elseif strcmpi(scope, 'subset')
    indexer          = repmat({':'}, 1, ndims(p_values_full));
    indexer{dim_idx} = subset_mask;
    p_subset         = p_values_full(indexer{:});
    
    valid_mask  = ~isnan(p_subset);
    p_valid     = p_subset(valid_mask);
    n_tests     = numel(p_valid);
    n_corrected = n_tests;
    
    if n_tests > 0
        if is_bonf
            p_subset(valid_mask) = min(p_valid .* n_tests, 1.0);
        elseif is_holm
            [p_subset(valid_mask), ~] = bonf_holm(p_valid);
        else
            p_subset(valid_mask) = mafdr(p_valid, 'BHFDR', true);
        end
        stats_corrected(indexer{:}) = p_subset;
    end
    fprintf('%s applied to subset (n = %d).\n', method, n_corrected);

% 4. Error handling for unrecognized scope
%-----------------------------------------
else
    error('Unrecognized scope: ''%s''. Use ''all'', ''per-condition'', or ''subset''.', scope);
end

end