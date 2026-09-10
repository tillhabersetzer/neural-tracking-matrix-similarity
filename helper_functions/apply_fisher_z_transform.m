function out = apply_fisher_z_transform(x)
%--------------------------------------------------------------------------
% Till Habersetzer, 25.02.2026
% Communication Acoustics, CvO University Oldenburg
% till.habersetzer@uol.de
%
%   DESCRIPTION:
%   Applies the Fisher Z-transformation (atanh) to correlation coefficients
%   to stabilize variance and normalize the sampling distribution.
%
%   WORKFLOW:
%   1.  Identifies non-NaN indices to preserve data structure.
%   2.  Clips values to [-0.999, 0.999] to prevent infinite results at |r|=1.
%   3.  Computes atanh for valid numeric entries.
%   4.  Ensures r = 0 remains z = 0 and NaNs remain NaNs.
%
%   INPUTS:
%   -   x:     Scalar, vector, or matrix of correlation coefficients (r).
%
%   OUTPUTS:
%   -   out:   Transformed values in Z-space.
%--------------------------------------------------------------------------

    % Initialize output with NaNs to match input shape
    out = nan(size(x));
    
    % Create a logical mask for valid numbers
    % This prevents the '0/0' or 'max(NaN)' logic errors
    idx = ~isnan(x);
    
    if any(idx, 'all')
        % 1. Clip values to avoid atanh(1) = Inf
        % 2. Apply atanh only to non-NaN elements
        clipped_vals = max(min(x(idx), 0.999), -0.999);
        out(idx)     = atanh(clipped_vals);
    end

end