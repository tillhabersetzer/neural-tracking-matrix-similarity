function preprocessing_audio_olsa(settings)
%--------------------------------------------------------------------------
% Till Habersetzer, 27.07.2025
% Communication Acoustics, CvO University Oldenburg
% till.habersetzer@uol.de 
%
% Description:
%   Preprocesses generic OLSA sentence stimuli for neural decoding analysis.
%   This function converts raw .wav sentence files into processed auditory
%   envelopes that are filtered and resampled to match the parameters of
%   the corresponding neurophysiological data.
%
%   Key Steps:
%   1.  Reads each raw .wav sentence file from the BIDS stimuli directory.
%   2.  Computes the auditory envelope for each sentence and adds pre- and
%       post-stimulus zero-padding.
%   3.  Creates a corresponding binary speech detection vector for each sentence.
%   4.  Applies the bandpass filter (defined in settings.decoding.bpfreq)
%       and downsamples the envelope to the final target sampling rate.
%   5.  Calculates and appends the average envelope and a combined detection
%       vector across all sentences.
%   6.  Saves a single .mat file containing a structure with all processed
%       sentence envelopes and detection vectors.
%
% Input:
%   settings (struct):  Struct containing all necessary paths and processing
%                       parameters, including the bandpass frequencies
%                       (settings.decoding.bpfreq).
%
% Outputs:
%   (None directly, saves a file to disk)
%   - A .mat file is saved to the 'derivatives/stimuli' directory. The filename
%     is dynamically generated based on the frequency band found in the
%     settings struct, e.g.,
%     'freqband-05to4_preprocessed_audio_olsa_decoding.mat'.
%--------------------------------------------------------------------------

%% Script settings 
%--------------------------------------------------------------------------

% Filter settings
filtertype = settings.decoding.filtertype; % windowed sinc type I linear phase FIR filter

% Load frequency band for filtering
bpfreq = settings.decoding.bpfreq; % same for meeg and audio

% Downsampling frequency
fs_down = settings.decoding.fs_down;

% Audio frequency
fs_audio = settings.fs_audio;

% MEG sampling frequency
fs_neuro = settings.decoding.fs_neuro;

% Epoch length for olsa trials
prestim  = settings.decoding.olsa.prestim;
poststim = settings.decoding.olsa.poststim;

% Zeropadding before filtering
padding = settings.decoding.olsa.padding;

bids_dir = settings.path2bids;
stim_dir = fullfile(bids_dir,'stimuli','olsa','sentences'); 

%% Process audio
%--------------------------------------------------------------------------
% Loop over all Olsa sentences
contents        = dir(stim_dir);
fnames_audio    = {contents.name};
fnames_audio    = fnames_audio(endsWith(fnames_audio, ".wav", "IgnoreCase", true));
n_sentences     = length(fnames_audio);
padding_samples = round(padding*1/min(bpfreq)*fs_neuro);

audio_envelopes    = struct();
audio_envelopes.fs = fs_down;

% Save results
%--------------
dir2save = fullfile(settings.path2derivatives,'stimuli');
% dir2save = fullfile(stim_dir,'envelopes');
if ~exist(dir2save,'dir')
    mkdir(dir2save)
end

for stn_idx = 1:n_sentences % loop over sentences

    fname_audio = fnames_audio{stn_idx};
    [audio, fs] = audioread(fullfile(stim_dir,fname_audio));
    if ~isequal(fs,fs_audio)
        error('Unexpected sampling frequency (%i)!',fs)
    end
    % Audio is stereo signal; focus on left channel which was also used in
    % the original experiment for both left and right ear.
    audio     = audio(:,1);
    len_audio = length(audio);

    % Add prestim/poststim interval
    audio = [zeros(round(prestim*fs_audio),1);audio;zeros(round(poststim*fs_audio),1)];

    % Speech detection vector
    audio_detection = [zeros(round(prestim*fs_audio),1);ones(len_audio,1);zeros(round(poststim*fs_audio),1)];

    % Compute auditory envelope
    cfg      = [];
    cfg.type = 'auditory_envelope';
    cfg.fs   = fs_audio;
    envelope = cal_envelope(cfg, audio);

    % figure
    % hold on
    % plot(audio)
    % plot(envelope)

    % Resample entire audio to the intermediate rate (fs_neuro)
    % The 'resample' function includes an anti-aliasing filter.
    envelope_resampled = resample(envelope, fs_neuro, fs_audio);

    % Apply the same bandpass filter used on the neuro-data
    % This requires the FieldTrip toolbox.
    % Tha audio data will be padded cause it is rather short the filtere
    % frequences very low and the filter sharp
    envelope_padded   = [zeros(1,padding_samples),envelope_resampled,zeros(1,padding_samples)];
    plotfiltresp      = 'no';
    envelope_filtered = ft_preproc_bandpassfilter(envelope_padded, fs_neuro, bpfreq, [], filtertype, [], [], [], [], [], plotfiltresp, []);
    envelope_filtered = envelope_filtered(padding_samples+1:end-padding_samples);

    % figure
    % hold on
    % plot(envelope_resampled)
    % plot(envelope_filtered)
    % legend({'padded','filtered'})
    % -> with and without padding leads to same result

    % Resample the extracted epoch to the final target frequency (fs_down)
    envelope_filtered_resampled = resample(envelope_filtered, fs_down, fs_neuro);
    audio_detection             = audio_detection(1:round(fs_audio/fs_down):end)';

    % Adjust detection vector to same length
    audio_detection = [audio_detection, zeros(1, max(0, length(envelope_filtered_resampled) - length(audio_detection)))]; % Append if necessary
    audio_detection = audio_detection(1:min(length(audio_detection), length(envelope_filtered_resampled))); % Cut to same length
     
    % figure
    % hold on;
    % plot(envelope_filtered_resampled./max(abs(envelope_filtered_resampled)))
    % plot(audio_detection)

    % Save data in struct
    [~, base_name, ~]                                         = fileparts(fname_audio);  
    audio_envelopes.(sprintf('envelope_%s',base_name))        = envelope_filtered_resampled;
    audio_envelopes.(sprintf('audio_detection_%s',base_name)) = audio_detection;
    fprintf('Sentence %i/%i (%s) processed.\n',stn_idx,n_sentences,fname_audio)

    % Additionally, save each envelope as a wav file. If olsa sentences are
    % removed, the dataset is not bids compliant anymore cause stim files
    % are missing.
    % audiowrite(fullfile(dir2save,sprintf('%s.wav',base_name)),envelope_filtered_resampled,fs_down);

    clear envelope envelope_resampled envelope_padded envelope_filtered envelope_filtered_resampled audio_detection

end % sentences

% Append mean envelope - for testing
%-----------------------------------
env_idx       = contains(fieldnames(audio_envelopes), "envelope", "IgnoreCase", true);
envelopes     = struct2cell(audio_envelopes);
envelopes     = envelopes(env_idx);
%max_samples      = max(cellfun(@numel, envelopes));
%envelopes = cellfun(@(x) [x, zeros(1, max_samples - numel(x))], envelopes, 'UniformOutput', false);
min_samples   = min(cellfun(@numel, envelopes));
envelopes     = cellfun(@(x) x(:,1:min_samples), envelopes, 'UniformOutput', false);
envelopes     = vertcat(envelopes{:}); %  % Vertically concatenate all padded row vectors into a matrix
envelope_avg  = mean(envelopes,1);

detection_idx = contains(fieldnames(audio_envelopes), "detection", "IgnoreCase", true);
detections    = struct2cell(audio_envelopes);
detections    = detections(detection_idx);
detections    = cellfun(@(x) x(:,1:min_samples), detections, 'UniformOutput', false);
detections    = vertcat(detections{:}); %  % Vertically concatenate all padded row vectors into a matrix
detection_avg = any(detections); % tests along the first array dimension of A whose size does not equal 1, and determines if any element is a nonzero

audio_envelopes.('envelope_avg')        = envelope_avg;
audio_envelopes.('audio_detection_avg') = detection_avg;
clear envelopes envelopes_padded
% 
% figure
% plot(envelope_avg);

% Save data
%----------
fname = sprintf(settings.fnames.audio_olsa, settings.helper.formatFreqBand(bpfreq));     
save(fullfile(dir2save,fname),'audio_envelopes','-v7.3'); 
fprintf("%s saved.\n",fname)

end % end of function