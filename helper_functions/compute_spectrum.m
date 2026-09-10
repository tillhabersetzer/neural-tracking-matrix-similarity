function [P1, f_axis] = compute_spectrum(envelope, n_fft, fs, varargin)
%--------------------------------------------------------------------------
% Computes the single-sided magnitude spectrum of an envelope.
%
% USAGE:
%   [P1, f_axis] = compute_spectrum(envelope, n_fft, fs)
%   [P1, f_axis] = compute_spectrum(..., 'Detrend', true, 'Window', 'tukey')
%
% INPUTS:
%   envelope - 1D numeric vector containing the input signal/envelope
%   n_fft    - Number of FFT points (defines resolution and padding length)
%   fs       - Sampling frequency of the signal (Hz)
%
% OPTIONAL NAME-VALUE PAIRS:
%   'Detrend' - Boolean (true/false) to remove the mean/DC offset. Default: true.
%   'Window'  - String ('none', 'hanning', 'tukey') to apply before padding. Default: 'none'.
%
% OUTPUTS:
%   P1       - Single-sided magnitude spectrum vector
%   f_axis   - Corresponding frequency bin axis (Hz)
%--------------------------------------------------------------------------

% Parse optional input arguments
%-------------------------------
p = inputParser;
addRequired(p, 'envelope', @isnumeric);
addRequired(p, 'n_fft', @isnumeric);
addRequired(p, 'fs', @isnumeric);
addParameter(p, 'Detrend', true, @islogical);
addParameter(p, 'Window', 'none', @(x) any(validatestring(x, {'none', 'hanning', 'tukey'})));

parse(p, envelope, n_fft, fs, varargin{:});

% Ensure envelope is a row vector to match window transpositions if needed
if iscolumn(envelope)
    envelope = envelope';
end

% Get signal length for windowing allocation
L = length(envelope);

% Optional Detrending (Offset Correction)
%----------------------------------------
if p.Results.Detrend
    % envelope = envelope - mean(envelope);
    envelope = detrend(envelope, 'linear'); 
end

% Optional Windowing
%-------------------
switch lower(p.Results.Window)
    case 'hanning'
        % hanning() returns a column vector; transpose to row
        envelope = envelope .* hanning(L)';
    case 'tukey'
        % Using a standard 10% taper taper for speech
        envelope = envelope .* tukeywin(L, 0.1)';
    case 'none'
        % Do nothing, leave signal as is
end

% Compute Single-Sided Magnitude Spectrum
%----------------------------------------
fft_env = fft(envelope, n_fft);

num_freq_bins = floor(n_fft / 2) + 1;
P2            = abs(fft_env / L);
P1            = P2(1:num_freq_bins);
P1(2:end-1)   = 2 * P1(2:end-1);

% Generate the corresponding frequency axis
f_axis = (0:num_freq_bins-1) * (fs/n_fft);

end