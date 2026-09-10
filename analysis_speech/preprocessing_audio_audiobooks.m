function preprocessing_audio_audiobooks(settings)
%--------------------------------------------------------------------------
% Till Habersetzer, 27.07.2025
% Communication Acoustics, CvO University Oldenburg
% till.habersetzer@uol.de 
%
% Description:
%   Preprocesses audiobook stimuli for neural decoding analysis. This function
%   reads raw .wav files from different sessions, computes their auditory
%   envelopes, and then segments the continuous envelopes into fixed-duration
%   epochs suitable for modeling.
%
%   It uses a local helper function, `epoch_audiobook_decoding`, to perform
%   the core filtering and epoching, ensuring the audio features are aligned
%   with the corresponding neurophysiological data parameters.
%
%   Key Steps:
%   1.  Iterates through audiobook .wav files organized by session and run.
%   2.  For each file, computes the continuous auditory envelope.
%   3.  Calls `epoch_audiobook_decoding` to filter, resample, and segment the
%       envelope into epochs of a specified duration (e.g., 60 seconds).
%   4.  Collects all epoched data and corresponding labels from all runs.
%   5.  Saves a single .mat file containing a structure with the collected
%       epochs and labels.
%
% Input:
%   settings (struct): A structure containing all necessary paths and
%                      processing parameters (e.g., sampling rates, filter
%                      frequencies, and epoch duration) for the analysis.
%
% Outputs:
%   (None directly, saves a file to disk)
%   - A .mat file is saved to the 'derivatives/stimuli' directory. The
%     filename is dynamically generated based on the frequency band, e.g.,
%     'freqband-05to4_preprocessed_audio_audiobooks_decoding.mat'. The file
%     contains a single struct `audio_envelopes` with the fields:
%       .epochs_audio (cell): Contains the epoched data for each run.
%       .audiobook_labels (cell): Contains the corresponding run labels.
%--------------------------------------------------------------------------

%% Script settings 
%--------------------------------------------------------------------------

% Filter settings
filtertype = settings.decoding.filtertype; % windowed sinc type I linear phase FIR filter

% Load frequency band for filtering
bpfreq = settings.decoding.bpfreq; % same for meeg and audio

% Epoch length
trialdur = settings.decoding.trialdur;

% Downsampling frequency
fs_down = settings.decoding.fs_down;

% Audio frequency
fs_audio = settings.fs_audio;

% MEG sampling frequency
fs_neuro = settings.decoding.fs_neuro;

bids_dir = settings.path2bids;
stim_dir = fullfile(bids_dir,'stimuli','audiobook'); % for audio data

%% Preprocess audio - same for most of the subjects
%--------------------------------------------------------------------------
f_idx            = 1; % file counter
audiobook_labels = cell(1,4);
epochs_audio     = cell(1,4);

% Audio is different in first and second session

for task_idx = 1:2     
    task = sprintf('audiobook%i',task_idx);
    ses  = sprintf('ses-0%i',task_idx);

    for run_idx = 1:2
        run = sprintf('0%i',run_idx);

        % Import raw audio and compute envelope
        fname_audio             = sprintf('task-%s_run-%s',task,run);
        audiobook_labels{f_idx} = fname_audio;

        [raw_audiodata,fs] = audioread(fullfile(stim_dir,ses,[fname_audio,'_stim.wav'])); 

        if ~isequal(fs,fs_audio)
            error('Unexpected sampling frequency in audiofile (%i)!',fs)
        end

        % Compute auditory envelope
        cfg      = [];
        cfg.type = 'auditory_envelope';
        cfg.fs   = fs_audio;
        envelope = cal_envelope(cfg, raw_audiodata);

        % Cut into epochs and downsampling
        % Includes intermediate downsampling step to fs_neuro and
        % applies same filter
        cfg              = [];
        cfg.fs_audio     = fs_audio;
        cfg.fs_neuro     = fs_neuro;
        cfg.fs_down      = fs_down;
        cfg.trialdur     = trialdur;
        cfg.bpfreq       = bpfreq;
        cfg.filtertype   = filtertype;
        cfg.plotfiltresp =  'no'; 
        envelope_epochs  = epoch_audiobook_decoding(cfg, envelope);
        fprintf('%s preprocessed.\n',fname_audio)

        epochs_audio{f_idx} = envelope_epochs;
        clear envelope_epochs raw_audiodata envelope

        f_idx = f_idx + 1;

    end % loop over runs

end % loop over tasks

% Save results
%--------------
dir2save = fullfile(settings.path2derivatives,'stimuli');
if ~exist(dir2save,'dir')
    mkdir(dir2save)
end
fname = sprintf(settings.fnames.audio_audiobook, settings.helper.formatFreqBand(bpfreq)); 

audio_envelopes                  = struct();
audio_envelopes.epochs_audio     = epochs_audio;
audio_envelopes.audiobook_labels = audiobook_labels;

save(fullfile(dir2save,fname),'audio_envelopes','-v7.3'); 
fprintf("%s saved.\n",fname)

%% Functions
%--------------------------------------------------------------------------

function [audiodata_epoched] = epoch_audiobook_decoding(cfg, audiodata)
%--------------------------------------------------------------------------
% Till Habersetzer, 27.06.2025
% Communication Acoustics, CvO University Oldenburg
% till.habersetzer@uol.de 
%
% Description:
%   Segments and resamples audio data to align with neurophysiological data.
%
%   This function processes a continuous audio signal by first resampling it
%   to an intermediate sampling rate (typically matching EEG/MEG data).
%   It then filters the signal, segments it into fixed-duration epochs,
%   and finally resamples each epoch to a final target frequency.
%
%   The output format is a structure similar to the FieldTrip data format.
%
% Input:
%   cfg (struct): Configuration with the following fields:
%       .fs_audio  (numeric) - Original sampling rate of the audio (Hz).
%       .fs_neuro  (numeric) - Intermediate sampling rate, to match neuro-data (Hz).
%       .fs_down   (numeric) - Final target sampling rate for each epoch (Hz).
%       .trialdur  (numeric) - Duration of each epoch (seconds).
%       .bpfreq    (1x2 numeric) - Bandpass filter frequencies [low, high] (Hz).
%       .filtertype  (char) - Type of filter to use, e.g., 'firws', 'but'.
%       .plotfiltresp (char) - 'yes' or 'no'
%
%   audiodata (vector): The raw audio time-series (1xT or Tx1).
%
% Output:
%   audiodata_epoched (struct): Data structure containing the epoched audio:
%       .trial  (cell)   - 1xN cell array, each containing an epoched and
%                          resampled audio trial.
%       .time   (cell)   - 1xN cell array, containing the time vector for each trial.
%       .fsample (numeric) - The final sampling rate of the data in .trial.
%       .trl    (matrix) - Nx3 matrix of [start, end, offset] sample indices for
%                          each epoch, relative to the intermediate resampled signal.
%       .cfg    (struct) - The original configuration structure.
%--------------------------------------------------------------------------

% 1. Input Validation and Preparation
%--------------------------------------------------------------------------
% Ensure audiodata is a row vector for consistent processing
if iscolumn(audiodata)
    audiodata = audiodata';
end

% 2. Resample and Filter Audio
%--------------------------------------------------------------------------
% This two-step process is chosen to mirror the processing pipeline of the
% corresponding neurophysiological (MEG/EEG) data.

% Processing Notes 
%-----------------
% To align with MEG/EEG data processing, the entire audio signal is first
% resampled and then filtered before being epoched. This avoids edge artifacts.
% For best resampling quality, `fs_audio` should be an integer multiple of
% `fs_neuro` (e.g., 44kHz -> 1kHz is better than 44.1kHz -> 1kHz).

% Step A: Resample entire audio to the intermediate rate (fs_neuro)
% The 'resample' function includes an anti-aliasing filter.
audiodata_resampled = resample(audiodata, cfg.fs_neuro, cfg.fs_audio);

% Step B: Apply the same bandpass filter used on the neuro-data
% This requires the FieldTrip toolbox.
audiodata_filtered = ft_preproc_bandpassfilter(audiodata_resampled, cfg.fs_neuro, cfg.bpfreq, [], cfg.filtertype, [], [], [], [], [], cfg.plotfiltresp, []);

% 3. Define Epochs
%--------------------------------------------------------------------------
% Create trial boundaries based on the filtered, intermediate-rate signal.
num_samples_neuro = length(audiodata_filtered);
epoch_len_samples = round(cfg.trialdur * cfg.fs_neuro);

% Determine start and end samples for each trial
trl_starts = (1:epoch_len_samples:num_samples_neuro)';
trl_ends   = trl_starts + epoch_len_samples - 1;

% Ensure the final epoch does not exceed the total number of samples
trl_ends(end) = min(trl_ends(end), num_samples_neuro);

% Create the trial definition matrix [start_sample, end_sample, offset]
trl = [trl_starts, trl_ends];

% Trim any potential final trial that is empty
if trl(end, 1) > num_samples_neuro
    trl(end, :) = [];
end
num_trials = size(trl, 1);

% 4. Segment and Downsample each epoch
%--------------------------------------------------------------------------
trials = cell(1, num_trials);
time   = cell(1, num_trials); 

for trl_idx = 1:num_trials
    % Extract the audio segment for the current trial
    start_idx      = trl(trl_idx, 1);
    end_idx        = trl(trl_idx, 2);
    original_trial = audiodata_filtered(start_idx:end_idx);

    % Resample the extracted epoch to the final target frequency (fs_down)
    resampled_trial = resample(original_trial, cfg.fs_down, cfg.fs_neuro);
    trials{trl_idx} = resampled_trial;

    % time vector
    num_samples_final = length(resampled_trial);
    start_time_sec    = (trl(trl_idx, 1) - 1) / cfg.fs_neuro;
    end_time_sec      = start_time_sec + (num_samples_final - 1) / cfg.fs_down;
    time{trl_idx}     = linspace(start_time_sec, end_time_sec, num_samples_final);
end

% 5. Package Data into Output Structure
%--------------------------------------------------------------------------
audiodata_epoched         = struct();
audiodata_epoched.trial   = trials;
audiodata_epoched.time    = time;
audiodata_epoched.fsample = cfg.fs_down;
audiodata_epoched.trl     = trl;
audiodata_epoched.cfg     = cfg;

end

end % end of function