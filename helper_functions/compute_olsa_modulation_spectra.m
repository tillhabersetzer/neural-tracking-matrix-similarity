function [spectra_matrix, f_axis] = compute_olsa_modulation_spectra(olsa_audio, fs_orig, num_sentences_to_concat, apply_lowpass, varargin)
% Randomly groups OLSA sentences, extracts their Hilbert envelopes, applies 
% optional processing (lowpass, downsampling, detrending, windowing) and 
% computes their single-sided magnitude spectra.
%
% INPUTS:
%   olsa_audio              - Cell array containing individual OLSA sentence audio vectors
%   fs_orig                 - Sampling frequency (Hz)
%   num_sentences_to_concat - Number of sentences to randomly group per block
%   apply_lowpass           - Boolean (true/false) to apply 50Hz lowpass filter
%
% OPTIONAL NAME-VALUE PAIRS:
%   'fs_new'                - Target downsampling frequency (Hz). Default: [] (No downsampling)
%   'Detrend'               - Boolean (true/false) to remove linear trends. Default: true.
%   'Window'                - String ('none', 'hanning', 'tukey'). Default: 'none'.
%--------------------------------------------------------------------------

% Parse optional input arguments
p = inputParser;
addRequired(p, 'olsa_audio', @iscell);
addRequired(p, 'fs_orig', @isnumeric);
addRequired(p, 'num_sentences_to_concat', @isnumeric);
addRequired(p, 'apply_lowpass', @islogical);
addParameter(p, 'fs_new', [], @(x) isempty(x) || (isnumeric(x) && x > 0));
addParameter(p, 'Detrend', true, @islogical);
addParameter(p, 'Window', 'none', @(x) any(validatestring(x, {'none', 'hanning', 'tukey'})));
parse(p, olsa_audio, fs_orig, num_sentences_to_concat, apply_lowpass, varargin{:});

% Set active sampling frequency for processing
fs_active = fs_orig;
if ~isempty(p.Results.fs_new)
    fs_active = p.Results.fs_new;
end

n_sentences      = length(olsa_audio);
shuffled_indices = randperm(n_sentences);
num_blocks       = floor(n_sentences / num_sentences_to_concat);

fprintf('\n==================================================\n');
fprintf('OLSA Processing: Found %d total sentences.\n', n_sentences);
fprintf('Grouping into blocks of %d sentences per block.\n', num_sentences_to_concat);
fprintf('--> Total number of blocks generated: %d\n', num_blocks);
if mod(n_sentences, num_sentences_to_concat) > 0
    fprintf('Note: %d leftover sentence(s) omitted to keep blocks uniform.\n', ...
        mod(n_sentences, num_sentences_to_concat));
end
fprintf('==================================================\n\n');

% Trim indices to fit perfect blocks
usable_indices = shuffled_indices(1 : num_blocks * num_sentences_to_concat);
block_matrix   = reshape(usable_indices, num_sentences_to_concat, num_blocks);

% Dynamic n_fft Calculation (Accounting for potential downsampling)
max_block_length = 0;
for b_idx = 1:num_blocks
    current_block_sentences = block_matrix(:, b_idx);
    current_block_length    = 0;
    for s_idx = 1:num_sentences_to_concat
        sentence_idx         = current_block_sentences(s_idx);
        current_block_length = current_block_length + length(olsa_audio{sentence_idx});
    end
    
    % Scale length if downsampling is requested
    if ~isempty(p.Results.fs_new)
        current_block_length = ceil(current_block_length * (p.Results.fs_new / fs_orig));
    end
    if current_block_length > max_block_length
        max_block_length = current_block_length;
    end
end

% Calculate uniform n_fft based on the active sample lengths
n_fft = 2^nextpow2(max_block_length);

% Preallocate output matrix based on uniform resolution
num_freq_bins  = floor(n_fft / 2) + 1;
spectra_matrix = zeros(num_freq_bins, num_blocks);

% Processing Loop
for b_idx = 1:num_blocks
    current_block_sentences = block_matrix(:, b_idx);
    % Concatenate the randomly chosen audio segments
    combined_audio = [];
    for s_idx = 1:num_sentences_to_concat
        sentence_pool_idx = current_block_sentences(s_idx);
        current_sentence  = olsa_audio{sentence_pool_idx};
        if iscolumn(current_sentence)
            current_sentence = current_sentence';
        end
        combined_audio = [combined_audio, current_sentence]; 
    end
    
    % Envelope Extraction & Processing
    envelope = abs(hilbert(combined_audio));
    
    % 1. Optional Lowpass Filtering
    if apply_lowpass 
        [b, a]   = butter(4, 50 / (fs_orig / 2), 'low');
        envelope = filtfilt(b, a, envelope); 
    end
    
    % 2. Optional Downsampling (Applied after lowpass filtering)
    if ~isempty(p.Results.fs_new)
        envelope = resample(envelope, p.Results.fs_new, fs_orig);
    end

    % figure
    % plot(envelope)
    
    L = length(envelope); 
    
    % 3. Optional Detrending
    if p.Results.Detrend
        envelope = detrend(envelope, 'linear'); 
    end
    
    % 4. Optional Windowing
    switch lower(p.Results.Window)
        case 'hanning'
            envelope = envelope .* hanning(L)';
        case 'tukey'
            envelope = envelope .* tukeywin(L, 0.1)';
        case 'none'
    end
    
    % Compute Single-Sided Magnitude Spectrum
    fft_env     = fft(envelope, n_fft);
    P2          = abs(fft_env / L);
    P1          = P2(1:num_freq_bins);
    P1(2:end-1) = 2 * P1(2:end-1);
    spectra_matrix(:, b_idx) = P1;
end

% Generate Uniform Frequency Axis using the active sampling rate
f_axis = (0:num_freq_bins-1) * (fs_active / n_fft);

end