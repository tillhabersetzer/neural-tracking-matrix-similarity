function [filtered_events] = read_events_modified(path_dataset, stim_channel, min_duration_samples)
%--------------------------------------------------------------------------
% Till Habersetzer, 06.06.2025
% Communication Acoustics, CvO University Oldenburg
% till.habersetzer@uol.de 
%
% READ_EVENTS_MODIFIED Extracts triggers from a specified channel in a Neuromag FIF file,
% filtering by a minimum sustained non-zero duration.
%
%   This function identifies periods where the specified stim_channel is
%   continuously non-zero. Each such period is considered a single event.
%   The event's 'value' is the first non-zero value at its onset.
%
%   Inputs:
%     path_dataset         - Full path to the Neuromag FIF data file.
%     stim_channel         - Name of the stimulus channel (e.g., 'STI101').
%     min_duration_samples - Minimum duration (in samples) for a valid event.
%
%   Output:
%     filtered_events      - Struct array of detected events ('type', 'sample',
%                            'value', 'offset', 'duration').
%                            'value' is the initial non-zero trigger.
%--------------------------------------------------------------------------

% Read header for channel labels
hdr = ft_read_header(path_dataset, 'headerformat', 'neuromag_fif', 'readbids','no');

% Find the specified stimulus channel index
stim_channel_idx = find(strcmp(hdr.label, stim_channel));
if isempty(stim_channel_idx)
    error('read_events_modified:ChannelNotFound', 'Channel ''%s'' not found in the dataset.', stim_channel);
end

% Read raw data from the stimulus channel
stim_raw = ft_read_data(path_dataset,'header',hdr,'chanindx',stim_channel_idx); 

% figure
% plot(stim_raw)

% Initialize event storage
filtered_events = struct('type', {}, 'sample', {}, 'value', {}, 'offset', {}, 'duration', {});
event_count = 0;

% Iterate through raw stimulus data to find trigger pulses
i = 1;
while i <= length(stim_raw)
    current_sample_value = stim_raw(i);

    % Look for the start of a non-zero pulse
    if current_sample_value ~= 0
        onset_sample = i;

        % Find the end of this continuous non-zero segment
        end_of_pulse_segment = onset_sample;
        while end_of_pulse_segment <= length(stim_raw) && stim_raw(end_of_pulse_segment) ~= 0
            end_of_pulse_segment = end_of_pulse_segment + 1;
        end
        
        pulse_duration = end_of_pulse_segment - onset_sample;

        % If duration meets minimum, determine event value and add event
        if pulse_duration >= min_duration_samples

            % Extract the segment of the pulse where value is non-zero
            pulse_segment_values = stim_raw(onset_sample : end_of_pulse_segment - 1);
            
            % Find unique non-zero values and their counts within this segment
            % Filter out any potential zeros if they somehow sneaked in (shouldn't happen with current loop logic)
            non_zero_values = pulse_segment_values(pulse_segment_values ~= 0);
            
            if isempty(non_zero_values)
                % This case should theoretically not happen if the loop started with non-zero
                % but adding a safeguard.
                i = end_of_pulse_segment; % Skip this segment
                continue;
            end

            unique_values = unique(non_zero_values);
            counts        = zeros(size(unique_values));
            for k = 1:length(unique_values)
                counts(k) = sum(non_zero_values == unique_values(k));
            end

            % Find the value with the maximum count (majority value)
            [~, max_idx] = max(counts);
            
            % If there are ties, max() returns the first index.
            % So, unique_values(max_idx) gives the first occurring majority value.
            determined_trigger_value = unique_values(max_idx);

            event_count                           = event_count + 1;
            filtered_events(event_count).type     = stim_channel;
            filtered_events(event_count).sample   = onset_sample;
            filtered_events(event_count).value    = determined_trigger_value; % Use the determined majority value
            filtered_events(event_count).offset   = 0;
            filtered_events(event_count).duration = pulse_duration;
        end

        % Move index past the processed pulse
        i = end_of_pulse_segment;
    else
        % Move to next sample if current value is zero
        i = i + 1;
    end
end

disp(['Found ', num2str(event_count), ' events on channel ''', stim_channel, ''' meeting min duration of ', num2str(min_duration_samples), ' samples.']);

end % End of function