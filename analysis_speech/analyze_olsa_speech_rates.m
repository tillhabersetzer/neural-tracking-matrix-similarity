%% OLSA Acoustic and Linguistic Feature Extraction
%--------------------------------------------------------------------------
% Till Habersetzer, 29.06.2026
% Communication Acoustics, CvO University Oldenburg
% till.habersetzer@uol.de 
%
% Description: 
%   This script analyzes stimuli from the Oldenburg Sentence Test (OLSA). 
%   It reads paired audio (.wav) and text (.txt) files to extract sentence 
%   durations and automatically counts the number of words and syllables 
%   using a predefined dictionary matching the standardized OLSA matrix.
%
%   Outputs:
%   - Computes Sentence, Word, and Syllable rates (in Hz).
%   - Generates a 6-panel tiled figure with histograms and summary 
%     statistics (Mean, Median, Std, absolute Min/Max, and the 95% 
%     data range via 2.5th/97.5th percentiles) for all metrics.
%
% Dependencies / Assumptions:
%   - A pre-defined structure `settings` containing `settings.path2bids`.
%   - BIDS-compliant directory structure containing matched .wav and .txt files.
%-------------------------------------------------------------------------

close all
clearvars
clc 

% Import main settings
%---------------------
settings_decoding

% Define Number of Syllables
%---------------------------
% 10x5 String Matrix of OLSA Words
olsa_words = [
    "peter",    "bekommt",  "drei",     "große",    "blumen";
    "kerstin",  "sieht",    "neun",     "kleine",   "tassen";
    "tanja",    "kauft",    "sieben",   "alte",     "autos";
    "ulrich",   "gibt",     "acht",     "nasse",    "bilder";
    "britta",   "schenkt",  "vier",     "schwere",  "dosen";
    "wolfgang", "verleiht", "fünf",     "grüne",    "sessel";
    "stefan",   "hat",      "zwei",     "teure",    "messer";
    "thomas",   "gewann",   "achtzehn", "schöne",   "schuhe";
    "doris",    "nahm",     "zwölf",    "rote",     "steine";
    "nina",     "malt",     "elf",      "weiße",    "ringe"
];

% 10x5 Numeric Matrix of Syllable Counts (matching positions above)
olsa_syllables = [
    2, 2, 1, 2, 2;
    2, 1, 1, 2, 2;
    2, 1, 2, 2, 2;
    2, 1, 1, 2, 2;
    2, 1, 1, 2, 2;
    2, 2, 1, 2, 2;
    2, 1, 1, 2, 2;
    2, 2, 2, 2, 2;
    2, 1, 1, 2, 2;
    2, 1, 1, 2, 2
];

% Flatten the 10x5 matrices into 1D arrays for easy lookup
dict_words     = olsa_words(:);
dict_syllables = olsa_syllables(:);

% Extract filenames
%------------------
stim_dir        = fullfile(settings.path2bids,'stimuli','olsa','sentences'); 
contents        = dir(stim_dir);
fnames_audio    = {contents.name};
fnames_audio    = fnames_audio(endsWith(fnames_audio, ".wav", "IgnoreCase", true));
n_sentences     = length(fnames_audio);
basenames       = regexprep(fnames_audio, '\.wav$', '', 'ignorecase'); % Extract basenames

% Import wav files and sentences
%-------------------------------
sentence_durations = zeros(n_sentences, 1);
sentence_contents  = cell(n_sentences, 1);
sentence_words     = zeros(n_sentences, 1);
sentence_syllables = zeros(n_sentences, 1);

for s_idx = 1:n_sentences
    [audio, fs] = audioread(fullfile(stim_dir, sprintf('%s.wav', basenames{s_idx})));
    % Duration
    sentence_durations(s_idx) = size(audio,1)/fs; % sec
    % Sentence
    sentence_contents{s_idx} = fileread(fullfile(stim_dir,[basenames{s_idx},'.txt']));

    % Compute Number of Words and Syllables
    %--------------------------------------
    raw_sentence = string(sentence_contents{s_idx});
    
    % Clean the string: lowercase it and remove punctuation (like periods)
    raw_sentence = lower(erasePunctuation(raw_sentence));
    
    % Split into individual words
    words              = split(raw_sentence);
    words(words == "") = []; % Remove any accidental empty spaces
    
    % Store the number of words (for OLSA, this should almost always be 5)
    sentence_words(s_idx) = length(words);

    if sentence_words(s_idx) ~= 5
        warning("File '%s' (Sentence %d) has %d words instead of the expected 5.", ...
                basenames{s_idx}, s_idx, sentence_words(s_idx));
    end
    
    % Look up and sum the syllables for this sentence
    syllable_count = 0;
    for w_idx = 1:length(words)
        % Find where this word exists in our flattened dictionary
        match_idx = find(dict_words == words(w_idx), 1);
        
        if ~isempty(match_idx)
            syllable_count = syllable_count + dict_syllables(match_idx);
        else
            warning("Word '%s' in sentence %d not found in OLSA matrix.", words(w_idx), s_idx);
        end
    end
    
    % Store the total syllables for this sentence
    sentence_syllables(s_idx) = syllable_count;

end
fprintf('Word and syllable counts computed for all sentences.\n')
clear audio fs

% Compute Word and Syllable Rates
%--------------------------------
sentence_rate = 1 ./ sentence_durations;                    % Sentences per second (Hz)
word_rate     = sentence_words ./ sentence_durations;       % Words per second (Hz)
syllable_rate = sentence_syllables ./ sentence_durations;   % Syllables per second (Hz)

% Convert rates to per-minute
sentence_rate_min = sentence_rate * 60;
word_rate_min     = word_rate * 60;
syllable_rate_min = syllable_rate * 60;

% Display Mean Rates in Command Window
fprintf('\n--- Average Speech Rates (Across %d Sentences) ---\n', n_sentences);
fprintf('Sentences per minute : %6.2f\n', mean(sentence_rate_min));
fprintf('Words per minute     : %6.2f\n', mean(word_rate_min));
fprintf('Syllables per minute : %6.2f\n', mean(syllable_rate_min));
fprintf('--------------------------------------------------\n\n');

%% Plot Summary Histograms
%-------------------------
% Bundle data and titles into a cell array for easy looping
plot_data = {sentence_durations, 'Sentence Duration / s'; ...
             sentence_words,     'Number of Words'; ...
             sentence_syllables, 'Number of Syllables'; ...
             syllable_rate,      'Syllable Rate / Hz'; ...
             word_rate,          'Word Rate / Hz'; ...
             sentence_rate,      'Sentence Rate / Hz' ...
};

% Create a large figure window
figure('color','white','Name', 'OLSA Summary Statistics','WindowState', 'maximized');
tiledlayout(2, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

for i = 1:6
    nexttile;
    
    % Extract current variable and title
    curr_data  = plot_data{i, 1};
    curr_title = plot_data{i, 2};
    
    % Compute statistics
    val_mean   = mean(curr_data);
    val_median = median(curr_data);
    val_std    = std(curr_data);
    val_min    = min(curr_data);
    val_max    = max(curr_data);

    % Compute 95% Data Range (2.5th and 97.5th percentiles)
    val_95_low = prctile(curr_data, 2.5);
    val_95_hi  = prctile(curr_data, 97.5);
    
    % Draw histogram
    % MATLAB auto-adjusts bins if all values are identical (like 5 words)
    histogram(curr_data, 'FaceColor', [0.2, 0.5, 0.7], 'EdgeColor', 'w');
    hold on;
    
    % Add vertical lines for Mean and Median
    h_mean = xline(val_mean, '-b', 'LineWidth', 2);
    h_med  = xline(val_median, '--r', 'LineWidth', 2);
    
    % Formatting and Labels
    title(curr_title, 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Count');
    xlabel('Value');
    
    % Add the specific numeric values as a formatted subtitle
    stats_string = sprintf('Mean: %.2f | Med: %.2f | Std: %.2f\nAbsolute Range: %.2f to %.2f\n95%% of Data falls between: %.2f and %.2f', ...
                            val_mean, val_median, val_std, val_min, val_max, val_95_low, val_95_hi);
    subtitle(stats_string, 'FontSize', 10, 'Color', [0.3 0.3 0.3]);
    
    % Only add the legend to the first plot to avoid clutter
    if i == 1
        legend([h_mean, h_med], {'Mean', 'Median'}, 'Location', 'best');
    end
    
    hold off;
    grid on;
    grid minor;
    box on;
end