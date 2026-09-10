%--------------------------------------------------------------------------
% Till Habersetzer, 25.07.2025
% Communication Acoustics, CvO University Oldenburg
% till.habersetzer@uol.de 
%--------------------------------------------------------------------------

% ensure that we don't mix up settings
clear settings

% Define essential paths 
%--------------------------------------------------------------------------
% settings.rootpath       = fullfile('/mnt','localSSDPOOL'); % server
settings.rootpath       = fullfile('M:'); % local computer
settings.path2project   = fullfile(settings.rootpath,'oksima');
settings.path2toolboxes = fullfile(settings.rootpath,'toolboxes');

% Defined based on upper paths
settings.path2bids        = fullfile(settings.path2project,'bidsdata');
settings.path2derivatives = fullfile(settings.path2bids,'derivatives');
settings.path2fieldtrip   = fullfile(settings.path2toolboxes,'fieldtrip-20250523');
settings.path2amtoolbox   = fullfile(settings.path2toolboxes,'amtoolbox-1.6.0');
settings.path2mtrftoolbox = fullfile(settings.path2toolboxes,'mTRF-Toolbox-2.7','mtrf');

settings.path2decoding    = @(subject) fullfile(settings.path2bids, 'derivatives', subject, 'speech', 'decoding');
settings.path2ica         = @(subject) fullfile(settings.path2bids, 'derivatives', subject, 'ica');

settings.path2decoding_preprocessing = @(subject) fullfile(settings.path2decoding(subject), 'preprocessing');
settings.path2decoding_training      = @(subject) fullfile(settings.path2decoding(subject), 'training');
settings.path2decoding_testing       = @(subject) fullfile(settings.path2decoding(subject), 'testing');
settings.path2decoding_monotonicity  = @(subject) fullfile(settings.path2decoding(subject), 'monotonicity');

settings.path2layout = @(subject) fullfile(settings.path2bids, 'derivatives', subject, 'layout');

% Define filenames
%--------------------------------------------------------------------------
settings.fnames.audio_olsa                       = 'freqband-%s_preprocessed_audio_olsa_decoding.mat'; % freqband
settings.fnames.audio_audiobook                  = 'freqband-%s_preprocessed_audio_audiobooks_decoding.mat'; % freqband
settings.fnames.preprocessed_audiobooks          = '%s_%s_%s_freqband-%s_preprocessed_audiobooks_decoding.mat'; % subject / condition / sensor / freqband
settings.fnames.preprocessed_olsa                = '%s_%s_%s_freqband-%s_preprocessed_olsa_decoding.mat';  % subject / condition / sensor / freqband
settings.fnames.training_decoder                 = '%s_%s_%s_freqband-%s_intgrwin-%s_training_decoding.mat';  % subject / condition / sensor / freqband / integration window
settings.fnames.prediction_decoder               = '%s_%s_%s_freqband-%s_intgrwin-%s_test-olsa_%s_decoding.mat'; % subject / condition / sensor / freqband / integration window / single or concat sentences
settings.fnames.prediction_decoder_across_trials = '%s_%s_%s_freqband-%s_intgrwin-%s_test-olsa_across_sentences_decoding.mat'; % subject / condition / sensor / freqband / integration window
settings.fnames.monotonicity                     = 'results_monotonicity.mat';
settings.fnames.layout                           = '%s_%s_layout-%s.mat'; % subject / session / sensor
% forward model
settings.fnames.training_forward                 = '%s_%s_%s_freqband-%s_intgrwin-%s_training_forward.mat';  % subject / condition / sensor / freqband / integration window'
settings.fnames.prediction_forward               = '%s_%s_%s_freqband-%s_intgrwin-%s_test-olsa_%s_forward.mat';
settings.fnames.audio_phoneme_sequence           = 'audio_%s_phoneme_sequence.mat'; 
settings.fnames.audio_phonetic_features          = 'audio_%s_phonetic_features.mat'; 
% further analysis
settings.fnames.prediction_decoder_olsa_sim      = '%s_%s_%s_freqband-%s_intgrwin-%s_test-olsa_sim_%s_decoding.mat'; % subject / condition / sensor / freqband / integration window / single or concat sentences

% Analysis settings
%--------------------------------------------------------------------------
settings.analysis.sensors = {'meg','eeg','ear_eeg'}; % for preprocessing
% settings.analysis.sensors = {'ear_eeg'}; % for preprocessing

% Other settings
%--------------------------------------------------------------------------
settings.use_maxfilter            = true;
settings.apply_latency_correction = true;
settings.audio_latency            = 3/1000; % 3 ms (see measurements 28.06.24)
settings.use_ica                  = false;
settings.remove_onset             = false; % only for "prediction_concatenated_sentences_olsa_similarity_decoding.m" or "prediction_single_sentences_olsa_similarity_decoding.m"
settings.onset_tmax               = 0.5;
% Only effects EEG
settings.interpolate_bad_channel  = true; % can not be applied in combination with ICA (has already interpolated bad channels)

% Apply ICA suffix to filenames if enabled
%--------------------------------------------------------------------------
if settings.use_ica
    fnames_fields = fieldnames(settings.fnames);
    
    % List the exact field names that should NOT get the ICA suffix
    skip_fields = {'audio_olsa', ...
                   'audio_audiobook', ...
                   'audio_phoneme_sequence', ...
                   'audio_phonetic_features', ...
                   'layout'};
                   
    for i = 1:length(fnames_fields)
        fn = fnames_fields{i};
        
        % If the current filename field is NOT in the skip list, modify it
        if ~ismember(fn, skip_fields)
            
            % Insert '_ica-on' right before '_decoding' or '_forward'
            if contains(settings.fnames.(fn), '_decoding')
                settings.fnames.(fn) = strrep(settings.fnames.(fn), '_decoding', '_ica-on_decoding');
                
            elseif contains(settings.fnames.(fn), '_forward')
                settings.fnames.(fn) = strrep(settings.fnames.(fn), '_forward', '_ica-on_forward');
                
            else
                % Fallback for files like results_monotonicity.mat
                settings.fnames.(fn) = strrep(settings.fnames.(fn), '.mat', '_ica-on.mat');
            end
            
        end
    end
end

% Sensors
%--------------------------------------------------------------------------
settings.cfg_sensors.meg       = 'meg';
settings.cfg_sensors.megmag    = 'megmag';
settings.cfg_sensors.megplanar = 'megplanar';
settings.cfg_sensors.eeg       = 'eeg';
settings.cfg_sensors.cap_eeg   = cell(64,1);
for chanidx = 1:64
    settings.cfg_sensors.cap_eeg{chanidx} = sprintf('EEG%03i', chanidx);
end
% ear-EEG
settings.cfg_sensors.ear_eeg.left  = {'EEG008','EEG060','EEG065','EEG067','EEG069','EEG071','EEG073','EEG075'}';
settings.cfg_sensors.ear_eeg.right = {'EEG012','EEG061','EEG066','EEG068','EEG070','EEG072','EEG074','EEG076'}';
settings.cfg_sensors.ear_eeg.both  = [settings.cfg_sensors.ear_eeg.left; settings.cfg_sensors.ear_eeg.right];

settings.cfg_sensors.biochannels = {'EOG001','EOG002','ECG003'};

% Speech
%--------------------------------------------------------------------------
settings.fs_audio = 44100;

% Decoder
settings.decoding.bpfreq              = [0.5,4]; % default settings / can be overwritten
settings.decoding.trialdur            = 60;
settings.decoding.filtertype          = 'firws';
settings.decoding.fs_neuro            = 1000;
settings.decoding.fs_down             = 64;
settings.decoding.apply_normalization = true;
settings.decoding.olsa.prestim        = 0;
settings.decoding.olsa.poststim       = 0.5;
settings.decoding.olsa.padding        = 10; % s zeropad olsa wav files before filtering (10 cycles lowest frequency)

% mTRF-model hyperparameters
settings.decoding.run_nested_cv              = true; % Set to false to skip audiobook nested CV and only train for OLSA
settings.decoding.decoder.n_portion_training = 0.8; % percentage of trials used for nested cross validation for Audiobooks
settings.decoding.decoder.drct               = -1; % direction: backward model
settings.decoding.decoder.intgrwin           = [0,400]; % minimum/maximum time lag (ms) - default settings / can be overwritten
settings.decoding.decoder.lambda_exp         = -6:2:6; % exponent regularization values
settings.decoding.decoder.zeropad            = 1; % zero-pad the outer rows of the design matrix or delete them
settings.decoding.decoder.corr_metric        = 'Spearman'; % 'Pearson' specifiy accuracy metric
settings.decoding.decoder.fast               = 1; % use the fast cross-validation method (requires more memory)
settings.decoding.decoder.dim                = 2; % work along the rows, observations in columns
settings.decoding.decoder.type               = 'multi'; % use all lags simultaneously to fit a multi-lag model

% Helper functions for naming
%--------------------------------------------------------------------------
settings.helper.formatFreqBand = @formatFrequencyBand;
settings.helper.formatIntgrWin = @formatFrequencyBand; % use same function here

%% Helper Functions 
%--------------------------------------------------------------------------

function freqbandName = formatFrequencyBand(freqband)
%--------------------------------------------------------------------------
% Transforms a frequency band array into a string name.
% Final Version: Converts numbers to strings and removes the decimal point.
% Automatically sorts the input to ensure min-to-max order.
% - Example: [0.2222, 0.44] -> '02222to044'
% - Example: [4, 0.5]       -> '05to4'
%--------------------------------------------------------------------------
% Define a simple handle to format one number.
% It converts to a high-precision string, then removes the decimal point.
formatNum = @(n) strrep(num2str(n, '%.15g'), '.', '');

% Ensure order and apply the formatting to the min and max values.
startStr = formatNum(min(freqband));
endStr   = formatNum(max(freqband));

% Combine the two parts into the final string.
freqbandName = [startStr, 'to', endStr];
    
end