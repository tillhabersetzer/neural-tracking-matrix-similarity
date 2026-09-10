function [trl, event] = my_trialfun_olsa(cfg)
%--------------------------------------------------------------------------
% Till Habersetzer, 05.07.2025
% Communication Acoustics, CvO University Oldenburg
% till.habersetzer@uol.de 
%
% Description:
%   Custom FieldTrip trial function ('trialfun') for defining epochs from
%   an OLSA (Oldenburg Sentence Test) experiment. This function is designed
%   to parse detailed trial information directly from event markers.
% 
% Key Steps:
%   -   Reads all events from the BIDS-compatible dataset specified in cfg.dataset.
%   -   Parses the event type string (e.g., 'olsa: list: 9 / sentence: 3 / snr: -5.0 / intelligibility: 1.0')
%       to extract the list number, sentence number, SNR, and intelligibility for each trial.
%   -   Constructs a detailed 'trl' matrix that includes not only the standard
%       [begin, end, offset] timing but also additional columns for the trigger value,
%       list number, sentence number, SNR, and intelligibility.
% 
% Inputs:
%   cfg (struct):   The FieldTrip configuration structure. It must contain:
%                   - cfg.dataset: Path to the data file.
%                   - cfg.trialdef.prestim: Time in seconds before the event.
%                   - cfg.trialdef.poststim: Time in seconds after the event.
%
% Outputs:
%   trl (matrix):   An Nx8 trial definition matrix with the following columns:
%                   1. Begin sample
%                   2. End sample
%                   3. Offset (in samples)
%                   4. List number
%                   5. Sentence number
%                   6. SNR value
%                   7. Intelligibility 
%                   8. Trigger value
%   event (struct): The original event structure read from the data file.
%--------------------------------------------------------------------------

% Read events
%--------------------------------------------------------------------------
event = ft_read_event(cfg.dataset, 'readbids', 'yes');  
   
val  = [event.value]';
smp  = [event.sample]';
dur  = [event.duration]'; % durations in ms
type = {event.type}';

% Check number of events
%--------------------------------------------------------------------------
n_events = length(val);
if ~ismember(n_events,[100,140,200]) % expected number of trials
    error('Unexpected number of events (%i)!',n_events)
end

% Extract list and sentence number
%--------------------------------------------------------------------------
list_num          = zeros(n_events,1);
sentence_num      = zeros(n_events,1);
snrs              = zeros(n_events,1);
intelligibilities = zeros(n_events,1);

% format specifier
formatSpec = 'olsa: list: %d / sentence: %d / snr: %f / intelligibility: %f';

% Loop through each cell and extract the numbers
for idx = 1:n_events
    values                 = sscanf(type{idx}, formatSpec);
    list_num(idx)          = values(1);
    sentence_num(idx)      = values(2);
    snrs(idx)              = values(3);
    intelligibilities(idx) = values(4);
end

% Compute trl-matrix
%--------------------------------------------------------------------------
hdr         = ft_read_header(cfg.dataset);
prestim     = round(hdr.Fs*cfg.trialdef.prestim); % in samples
poststim    = round(hdr.Fs*cfg.trialdef.poststim); % in samples
dur_samples = round(hdr.Fs*dur/1000); % in samples
trl         = [smp - prestim, smp + dur_samples + poststim, - prestim*zeros(n_events,1),list_num,sentence_num,snrs,intelligibilities,val]; 

end