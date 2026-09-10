function [trl, event] = my_trialfun_audiobook(cfg)
%--------------------------------------------------------------------------
% Till Habersetzer, 27.06.2025
% Communication Acoustics, CvO University Oldenburg
% till.habersetzer@uol.de 
%
% MY_TRIALFUN_AUDIOBOOK Defines trials for continuous audiobook EEG/MEG data.
%
%   [trl, event] = MY_TRIALFUN_AUDIOBOOK(cfg) segments continuous
%   neurophysiological data into fixed-length trials for audiobook paradigms.
%   It first identifies the full listening block using specific event
%   triggers (`cfg.trialdef.eventtype`, `cfg.trialdef.eventvalue`), then
%   divides this block into trials of `cfg.trialdef.trialdur` seconds.
%
%   Event reading supports:
%   - 'fieldtrip': Custom `read_events_modified` for stimulus channels.
%   - 'bids': `ft_read_event` for BIDS-compliant event files.
%
% Input `cfg` fields:
%   - `cfg.dataset`: Path to data file.
%   - `cfg.trialdef.eventtype`: Event trigger type (e.g., 'STI001').
%   - `cfg.trialdef.eventvalue`: Event trigger value (e.g., 5).
%   - `cfg.trialdef.trialdur`: Trial length in seconds.
%   - `cfg.trialdef.method`: 'fieldtrip' or 'bids'.
%
% Outputs:
%   - `trl` (N x 3 matrix): FieldTrip trial definition [begin_sample end_sample offset_sample].
%   - `event` (struct array): Filtered event information matching trial triggers.
%--------------------------------------------------------------------------

% Extract parameters
path2dataset = cfg.dataset;
event_type   = cfg.trialdef.eventtype; % e.g. STI101
trial_dur    = cfg.trialdef.trialdur; % 10 (s)
event_val    = cfg.trialdef.eventvalue; % e.g. 1
method       = cfg.trialdef.method; % 'fieldtrip' 'bids'

switch method
    case 'fieldtrip'
        % Use own modified function to read in events
        min_duration_samples = 10;
        event                = read_events_modified(path2dataset, event_type, min_duration_samples);
    case 'bids'
        event = ft_read_event(path2dataset, 'readbids', 'yes');  
    otherwise
        error('Unexpected method (%s) requested! Use "bids" or "fieldtrip".',method)
end

typ = {event.type}';
val = [event.value]';
smp = [event.sample]';
idx = find(strcmp(typ,event_type) & ismember(val, event_val));

% Interval length in samples
trlbegin = smp(idx(1:trial_dur:end));
if trlbegin(end) == smp(idx(end))
    trlbegin(end) = [];  
end
trlend = [trlbegin(2:end)-1;smp(idx(end))];
trl    = [trlbegin trlend zeros(length(trlbegin),1)];

% Check whether difference matches triallength
%---------------------------------------------
hdr             = ft_read_header(path2dataset);
trial_dur_check = (trl(:,2)-trl(:,1))./hdr.Fs; % in sec
% Last trial can be shorter but checker whether the first trials are close
% to trial duration
if any(abs(trial_dur_check(1:end-1)-trial_dur)>trial_dur*0.5/1000) % > tolerance: 1/2ms delay per sec trigger
    error('Unexpected trial duration!')
end

event = event(idx);

end