function [envelope] = cal_envelope(cfg, audiodata)
%--------------------------------------------------------------------------
% Till Habersetzer, 26.06.2025
% Communication Acoustics, CvO University Oldenburg
% till.habersetzer@uol.de 
% Last Update; 04.07.2025
%
% Description:
%   Extracts an envelope from an audio signal based on a configuration.
%   The filtering for Hilbert-based envelopes is handled internally using
%   the FieldTrip function 'ft_preproc_lowpassfilter'.
%
%   'envelope':         Log-compressed amplitude envelope via Hilbert
%                       transform and low-pass filtering.
%   'onset_envelope':   Emphasizes onsets by taking the rectified derivative
%                       of the smoothed envelope. (Output is 1 sample shorter).
%   'auditory_envelope': Perceptual envelope from a gammatone filterbank.
%                        (Ignores 'b' and 'a'; requires Auditory Modeling Toolbox).
%
% Input Arguments:
%   cfg (struct): Configuration structure with the following fields:
%       .type (char): 'envelope', 'onset_envelope', or 'auditory_envelope'.
%       .fs   (scalar): Sampling frequency in Hz.
%
%       Required for 'envelope' & 'onset_envelope':
%       .lpfreq (scalar): Low-pass filter cutoff frequency (e.g., 15 Hz).
%
%       Optional for 'envelope' & 'onset_envelope':
%       .lpfiltord (scalar): Filter order
%       .filtertype (char): Filter type, e.g., 'but' or 'firws' (default: 'but').
%       .plotfiltresp (char): Plot filter response, e.g., 'yes' or 'no' (default: 'no').
%    
%   audiodata (vector): Input audio time-series.
%
% Output Arguments:
%   envelope (vector): The calculated envelope time-series.
%
% Examples:
%   % For a standard log-envelope with a 15 Hz low-pass filter
%   cfg = [];
%   cfg.type   = 'envelope';
%   cfg.fs     = 44100;
%   cfg.lpfreq = 15;
%   log_env = cal_envelope(cfg, audio);
%
%   % For an auditory envelope (requires the AMT toolbox)
%   cfg      =  [];
%   cfg.type = 'auditory_envelope';
%   cfg.fs   = 44100;
%   aud_env  = cal_envelope(cfg_aud, audio);
%--------------------------------------------------------------------------

% Input Validation 
%--------------------------------------------------------------------------
type        = cfg.type;
fs          = cfg.fs;
valid_types = {'envelope', 'onset_envelope', 'auditory_envelope'};
if ~ismember(lower(type), valid_types)
    error('CAL_ENVELOPE: Unsupported Type! Type must be one of: %s.', strjoin(valid_types, ', '));
end

% Envelope types
%--------------------------------------------------------------------------

if strcmpi(type, 'auditory_envelope') % case insensitive
    % Auditory Envelope (Gammatone Filterbank-based) 
    %----------------------------------------------------------------------
    % Ref: 
    % Biesmans W, Das N, Francart T, Bertrand A. Auditory-Inspired Speech 
    % Envelope Extraction Methods for Improved EEG-Based Auditory Attention 
    % Detection in a Cocktail Party Scenario. IEEE Trans Neural Syst Rehabil 
    % Eng. 2017 May;25(5):402-412. doi: 10.1109/TNSRE.2016.2571900. Epub 2016 May 24. PMID: 27244743.
    if ~exist('auditoryfilterbank', 'file')
        error('CAL_ENVELOPE: Missing Dependency! "auditory_envelope" type requires the "auditoryfilterbank" function from the Auditory Modeling Toolbox (AMT).');
    end

    % must be column
    if isrow(audiodata)
        audiodata = audiodata';
    end
    
    % Process through a gammatone-like filterbank.
    [audio_filtered, ~] = auditoryfilterbank(audiodata, fs, 'flow', 50, 'fhigh', 5000);
    
    % Apply power-law compression (exponent 0.6) and average across frequency channels.
    envelope = mean(abs(audio_filtered).^(0.6), 2);
    
else % Handles 'onset_envelope' and 'envelope'
    % Hilbert-based Envelopes 
    %----------------------------------------------------------------------
    
    if ~isfield(cfg, 'lpfreq')
        error('cfg.lpfreq (low-pass frequency) is required for this envelope type.');
    end
    if ~isfield(cfg, 'lpfiltord'), cfg.lpfiltord = []; end
    if ~isfield(cfg, 'filtertype'), cfg.filtertype = 'but'; end
    if ~isfield(cfg, 'plotfiltresp'), cfg.plotfiltresp = 'no'; end

    % must be row
    if iscolumn(audiodata)
        audiodata = audiodata';
    end

    % 1. Calculate instantaneous amplitude via Hilbert transform.
    inst_amplitude = abs(hilbert(audiodata));
    
    % 2. Apply zero-phase low-pass filtering to smooth the amplitude.
    filtered_amplitude = ft_preproc_lowpassfilter(inst_amplitude, fs, cfg.lpfreq, cfg.lpfiltord, cfg.filtertype, [], [], [], [], [], cfg.plotfiltresp, []);
    
    % 3. Apply final, type-specific transformation.
    switch lower(type)
        case 'onset_envelope'
            % Emphasizes signal onsets by calculating the rectified derivative.
            %------------------------------------------------------------------
            % Ref: 
            % Petersen, Eline Borch, et al. "Neural tracking of attended versus 
            % ignored speech is differentially affected by hearing loss." 
            % Journal of neurophysiology 117.1 (2017): 18-27.
            
            % Differentiate to find the rate of change. (Note: output is 1 sample shorter).
            derivative = diff(filtered_amplitude) * fs;
            
            % Half-wave rectify to keep only positive changes (energy increases).
            envelope = max(0, derivative);
            
        case 'envelope'
            % Calculates the log-compressed amplitude, similar to a dB scale.
            %----------------------------------------------------------------
            
            % Floor at zero before log to handle potential filter ringing artifacts.
            filtered_amplitude = max(0, filtered_amplitude);
            
            % Add a small epsilon to prevent log(0) for silent segments.
            epsilon = 1e-9;
            
            % Apply log transformation.
            envelope = 20 * log10(filtered_amplitude + epsilon);
    end
end

% Return as row
if iscolumn(envelope)
    envelope = envelope';
end

end