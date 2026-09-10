function out = apply_inverse_fisher_z_transform(z)
%--------------------------------------------------------------------------
% Till Habersetzer, 26.02.2026
% Communication Acoustics, CvO University Oldenburg
% till.habersetzer@uol.de
%
%   DESCRIPTION:
%   Applies the inverse Fisher Z-transformation (tanh) to Z-scores
%   to return them to the correlation coefficient (r) space.
%
%   WORKFLOW:
%   1.  Identifies non-NaN indices to preserve data structure.
%   2.  Computes tanh for valid numeric entries.
%   3.  Ensures z = 0 remains r = 0 and NaNs remain NaNs.
%
%   INPUTS:
%   -   z:     Scalar, vector, or matrix of Fisher Z-scores.
%
%   OUTPUTS:
%   -   out:   Transformed values in Correlation space (r).
%--------------------------------------------------------------------------
    % Initialize output with NaNs to match input shape
    out = nan(size(z));
    
    % Create a logical mask for valid numbers
    idx = ~isnan(z);
    
    if any(idx, 'all')
        % Apply tanh to non-NaN elements to transform back to [-1, 1]
        out(idx) = tanh(z(idx));
    end
end