function stars = get_significance_stars(p)
%--------------------------------------------------------------------------
% Till Habersetzer, 09.12.2025
% Communication Acoustics, CvO University Oldenburg
% till.habersetzer@uol.de
%
%   DESCRIPTION:
%   Converts a numeric p-value into a character string of asterisks 
%   representing standard statistical significance levels.
%
%   WORKFLOW:
%   1.  Receives a single p-value.
%   2.  Compares it against standard thresholds:
%       - p < 0.001 -> '***'
%       - p < 0.01  -> '**'
%       - p < 0.05  -> '*'
%       - otherwise -> ''
%
%   INPUTS:
%   -   p:     Scalar numeric value (the p-value to check).
%
%   OUTPUTS:
%   -   stars: String/Char array containing the asterisks.
%--------------------------------------------------------------------------

    if p < 0.001
        stars = '***';
    elseif p < 0.01
        stars = '**';
    elseif p < 0.05
        stars = '*';
    else
        stars = '';
    end
end