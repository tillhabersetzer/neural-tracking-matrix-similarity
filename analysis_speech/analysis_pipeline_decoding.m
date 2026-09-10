%--------------------------------------------------------------------------
% Till Habersetzer, 05.07.2025
% Communication Acoustics, CvO University Oldenburg
% till.habersetzer@uol.de 
%
%   DESCRIPTION:
%   This script is the master entry point for the MEG-audio neural decoding
%   analysis pipeline. It orchestrates the entire workflow, from data
%   preprocessing to model training and testing, for a specified set of
%   subjects. The pipeline is designed for parallel execution on a server
%   to efficiently process large datasets.
%
%   WORKFLOW:
%   1.  **Configuration:** Loads base settings from 'settings_decoding.m' and
%       applies run-specific parameters defined in the "Analysis Parameters"
%       section of this script.
%   2.  **Task Definition:** Generates a comprehensive list of all required
%       processing steps (e.g., preprocessing, training) and their expected
%       output file paths.
%   3.  **Task Filtering:** Determines which tasks need to be run by checking
%       for the existence of output files (unless 'recompute_all' is true).
%   4.  **Execution:**
%       -   Processes shared audio stimuli serially.
%       -   Executes subject-level tasks in parallel using dynamically
%           managed parfor loops with task-specific worker counts.
%   5.  **Logging:** Records all command window output and total execution
%       time to a timestamped log file in the root project directory.
%
%   INPUTS:
%   -   **settings_decoding.m:** A script on the MATLAB path that defines the
%       'settings' struct. This struct must contain all necessary paths
%       (e.g., path2project, path2derivatives) and default parameters.
%   -   **Raw Data:** BIDS-formatted MEG and audio data, accessible via the
%       paths defined in the 'settings' struct.
%
%   OUTPUTS:
%   -   **Derivative Data:** Preprocessed data, trained models, and prediction
%       results saved as .mat files in the derivatives directory.
%   -   **Log File:** A .txt file named 'analysis_pipeline_decoding_log_...'
%       containing the full command window output for the run.
%
%   USAGE:
%   -   **From MATLAB IDE:** Ensure paths in 'settings_decoding.m' are correct
%       and run this script directly (F5).
%   -   **From Linux Command Line (recommended for servers):**
%       matlab2024b -nodisplay -nosplash -r "run('analysis_pipeline_decoding.m'); exit;"
%       matlab2023a -nodisplay -nosplash -r "run('analysis_pipeline_decoding.m'); exit;"
%--------------------------------------------------------------------------

%% Initial setup
%--------------------------------------------------------------------------
% Clears workspace, closes figures, and loads the main settings file

close all;
clearvars;
clc;

% Import main settings from an external file
settings_decoding;

%% Analysis parameters
%--------------------------------------------------------------------------

% High-level switches for the entire pipeline.
%-------------------------------------------------------------------------
recompute_all = true;     % true: re-run all tasks; false: skip completed tasks.
delete_data   = false;     % DANGER: true to delete previous derivative data before starting.

% Data Selection 
%--------------------------------------------------------------------------
% Defines the scope of the data to be processed
subjects            = 0:20;     % Subject IDs to process (0:20)
sessions            = 1:2;      % Session numbers
conditions          = 1:3;      % Condition numbers (1:ses-01, 2:ses-02, 3:pooled)
sensors             = {'meg', 'eeg', 'ear_eeg'};
% sensors             = {'biochannels'}; % EOG / ECG
% sentence_conditions = {'concat', 'single'}; % Prediction conditions to compute
% sentence_conditions = {'concat'};
sentence_conditions = {'concat_sim'}; % compute olsa similarity
% sentence_conditions = {'single_sim'}; % compute olsa similarity

% Model & Decoding Hyperparameters 
%--------------------------------------------------------------------------
% These values are assigned to the main 'settings' struct for use in functions
settings.analysis.sensors          = sensors;    % Active sensor group(s)
settings.decoding.bpfreq           = [0.5, 8];   % Bandpass filter frequencies in Hz [0.5, 4]
settings.decoding.decoder.intgrwin = [0, 400];    % Decoder integration window in ms [0, 75], [0,250]

% Execution & Pipeline Control
%--------------------------------------------------------------------------
% Manages which parts of the pipeline run and with what resources

% Parallel Computing Parameters 
%------------------------------
% Define the number of workers for each step ('Inf' uses all available)
max_worker2use                       = struct();
max_worker2use.preprocessing_meeg    = Inf;  % High memory load, may require fewer workers (15)
max_worker2use.training_decoder      = 12; % if nested cross validation is selected, it takes up ram
max_worker2use.testing_decoder       = Inf;
max_worker2use.testing_across_trials = Inf;

% Pipeline Steps to Run 
%----------------------
% Toggles to enable or disable major stages of the analysis
computation_status                       = struct();
computation_status.preprocessing_audio   = false;
computation_status.preprocessing_meeg    = false;
computation_status.training_decoder      = false;
computation_status.testing_decoder       = true;
computation_status.testing_across_trials = false;

%% Logging setup
%--------------------------------------------------------------------------
% Configures a text file to record the command window output

fname_log = "analysis_pipeline_decoding_log_" + string(datetime('now'), 'yyyy-MM-dd_HH-mm') + ".txt";

% Delete the old log file if it exists
% if exist(fname_log, 'file')
%     delete(fname_log);
% end

% Start recording all command window output to the log file
diary(fname_log);

%% Environent Initialization
%--------------------------------------------------------------------------
% Starts the timer, prints startup messages, and adds toolboxes to the path

tic; % Start the main timer for the analysis

% Print Startup Header 
fprintf('--- Starting Decoding Pipeline ---\n');
fprintf('Date: %s\n', datetime('now', 'Format', 'dd-MMM-yyyy HH:mm:ss'));
fprintf('Processing %d subjects across %d sessions.\n\n', length(subjects), length(sessions));

% Add Required Toolboxes to MATLAB Path 
fprintf('Adding toolboxes to path...\n');
addpath(fullfile(settings.path2project, 'analysis', 'helper_functions'));

% Add FieldTrip path only when on the server
if contains(settings.rootpath, '/mnt/localSSDPOOL')
    addpath(settings.path2fieldtrip);
    ft_defaults;
    fprintf('  - FieldTrip added (Server environment).\n');
end

% Initialize Auditory and mTRF Toolboxes
run(fullfile(settings.path2amtoolbox, 'amtstart'));
addpath(genpath(settings.path2mtrftoolbox));
fprintf('  - AMToolbox and mTRF Toolbox initialized.\n\n');

fprintf('--- Setup complete. Starting main analysis... ---\n');

%% Create tasklists
%--------------------------------------------------------------------------
n_subj = length(subjects);
n_ses  = length(sessions);
n_cond = length(conditions);
n_sens = length(sensors);

compute_concat_sentences          = ismember('concat',sentence_conditions);
compute_single_sentences          = ismember('single',sentence_conditions);
compute_concat_sentences_olsa_sim = ismember('concat_sim',sentence_conditions);
compute_single_sentences_olsa_sim = ismember('single_sim',sentence_conditions);

% (1) Preprocessesing Audio
%--------------------------------------------------------------------------
% Single task

filelist    = cell(2,1);
dir2save    = fullfile(settings.path2derivatives,'stimuli');
% Audiobook
fname       = sprintf(settings.fnames.audio_audiobook, settings.helper.formatFreqBand(settings.decoding.bpfreq)); 
filelist{1} = fullfile(dir2save, fname);
% Olsa
fname       = sprintf(settings.fnames.audio_olsa, settings.helper.formatFreqBand(settings.decoding.bpfreq));  
filelist{2} = fullfile(dir2save, fname);

% Define task parameters
%----------------------- 
tasks_audio          = struct();
tasks_audio(1).filelist = filelist;
clear fname dir2save filelist
fprintf('Tasks for audio preprocessing computed.\n');

% (2) Preprocessing MEEG
%--------------------------------------------------------------------------
n_tasks = n_subj * n_ses;

% Pre-allocate a lean struct array
tasks_preprocessing = repmat(struct('subject','','session',''), n_tasks, 1);
    
task_idx = 1; % Initialize a single counter
for sub_idx = 1:n_subj
    for ses_idx = sessions(:)' % Use a different loop variable

        subject = sprintf('sub-%02d', subjects(sub_idx));
        session = sprintf('ses-0%d', ses_idx);
       
        % Append expected files
        filelist = cell(2*n_sens,1);
        dir2save = settings.path2decoding_preprocessing(subject);
        for sens_idx = 1:n_sens    
            sensor = sensors{sens_idx};

            % Audiobook
            fname = sprintf(settings.fnames.preprocessed_audiobooks,subject,session,sensor,settings.helper.formatFreqBand(settings.decoding.bpfreq)); 
            filelist{2*sens_idx-1} = fullfile(dir2save, fname);
            % Olsa
            fname                  = sprintf(settings.fnames.preprocessed_olsa,subject,session,sensor,settings.helper.formatFreqBand(settings.decoding.bpfreq));
            filelist{2*sens_idx}   = fullfile(dir2save, fname);
            
        end

        % Define task parameters
        tasks_preprocessing(task_idx).subject  = subject;
        tasks_preprocessing(task_idx).session  = session;
        tasks_preprocessing(task_idx).filelist = filelist;

        clear fname dir2save subject session sensor filelist

        task_idx = task_idx + 1;
    end
end
fprintf('Tasks for MEEG preprocessing computed.\n');

% (3) Training and Testing
%--------------------------------------------------------------------------
n_tasks = n_subj * n_cond;

% Pre-allocate a lean struct array
tasks_training = repmat(struct('subject','', 'condition',''), n_tasks, 1);
tasks_testing  = repmat(struct('subject','', 'condition',''), n_tasks, 1);
    
task_idx = 1; % Initialize a single counter
for sub_idx = 1:n_subj
    for con_idx = conditions(:)' 

        subject   = sprintf('sub-%02d', subjects(sub_idx));
        
        if ismember(con_idx, [1,2])
            condition = sprintf('ses-0%d', conditions(con_idx));
        else
            condition = 'ses-pooled';
        end

        % Training
        %---------
        % Append expected files
        filelist = cell(n_sens,1);
        dir2save = settings.path2decoding_training(subject);

        for sens_idx = 1:n_sens    
            sensor = sensors{sens_idx};
            fname  = sprintf(settings.fnames.training_decoder,subject,condition,sensor, ...
                             settings.helper.formatFreqBand(settings.decoding.bpfreq), ...
                             settings.helper.formatIntgrWin(settings.decoding.decoder.intgrwin));
            filelist{sens_idx} = fullfile(dir2save, fname);

        end

        tasks_training(task_idx).subject   = subject;
        tasks_training(task_idx).condition = condition;
        tasks_training(task_idx).filelist  = filelist;

        % Testing
        %--------
        % Append expected files
        filelist1 = cell(n_sens,1); % concatenated sentences
        filelist2 = cell(n_sens,1); % single sentences
        filelist3 = cell(n_sens,1); % concatenated sentences olsa similarity
        dir2save  = settings.path2decoding_testing(subject);

        for sens_idx = 1:n_sens    
            sensor = sensors{sens_idx};

            if compute_concat_sentences 
                fname = sprintf(settings.fnames.prediction_decoder,subject,condition,sensor, ...
                                settings.helper.formatFreqBand(settings.decoding.bpfreq), ...
                                settings.helper.formatIntgrWin(settings.decoding.decoder.intgrwin), ...
                                'concat_sentences');
                filelist1{sens_idx} = fullfile(dir2save, fname);
            end

            if compute_single_sentences
                fname = sprintf(settings.fnames.prediction_decoder,subject,condition,sensor, ...
                                settings.helper.formatFreqBand(settings.decoding.bpfreq), ...
                                settings.helper.formatIntgrWin(settings.decoding.decoder.intgrwin), ...
                                'single_sentences');
                filelist2{sens_idx} = fullfile(dir2save, fname);
            end

            if compute_concat_sentences_olsa_sim
                fname = sprintf(settings.fnames.prediction_decoder_olsa_sim,subject,condition,sensor, ...
                                settings.helper.formatFreqBand(settings.decoding.bpfreq), ...
                                settings.helper.formatIntgrWin(settings.decoding.decoder.intgrwin), ...
                                'concat_sentences');
                filelist3{sens_idx} = fullfile(dir2save, fname);
            end

            if compute_single_sentences_olsa_sim 
                fname = sprintf(settings.fnames.prediction_decoder_olsa_sim,subject,condition,sensor, ...
                                settings.helper.formatFreqBand(settings.decoding.bpfreq), ...
                                settings.helper.formatIntgrWin(settings.decoding.decoder.intgrwin), ...
                                'single_sentences');
                filelist3{sens_idx} = fullfile(dir2save, fname);
            end

        end

        % Only keep non-empty entries
        filelist                              = [filelist1;filelist2;filelist3];
        filelist(cellfun(@isempty, filelist)) = [];

        tasks_testing(task_idx).subject   = subject;
        tasks_testing(task_idx).condition = sprintf('ses-0%d', con_idx);
        tasks_testing(task_idx).filelist  = filelist;

        clear fname dir2save subject condition sensor filelist1 filelist2 filelist 3 filelist

        task_idx = task_idx + 1;
    end
end
fprintf('Tasks for Training and Testing computed.\n');

% (4) Testing across number of concatenated trials
%--------------------------------------------------------------------------
n_tasks = n_subj * n_cond;

% Pre-allocate a lean struct array
tasks_across_trials = repmat(struct('subject','','condition',''), n_tasks, 1);
    
task_idx = 1; % Initialize a single counter
for sub_idx = 1:n_subj
    for con_idx = conditions(:)' 

        subject = sprintf('sub-%02d', subjects(sub_idx));

        if ismember(con_idx, [1,2])
            condition = sprintf('ses-0%d', conditions(con_idx));
        else
            condition = 'ses-pooled';
        end
       
        % Append expected files
        filelist = cell(n_sens,1);
        dir2save = settings.path2decoding_testing(subject);
        for sens_idx = 1:n_sens    
            sensor = sensors{sens_idx};

            fname = sprintf(settings.fnames.prediction_decoder_across_trials,subject,condition,sensor, ...
                            settings.helper.formatFreqBand(settings.decoding.bpfreq), ...
                            settings.helper.formatIntgrWin(settings.decoding.decoder.intgrwin));
            filelist{sens_idx} = fullfile(dir2save, fname);
        end

        % Define task parameters
        tasks_across_trials(task_idx).subject   = subject;
        tasks_across_trials(task_idx).condition = sprintf('ses-0%d', con_idx);
        tasks_across_trials(task_idx).filelist  = filelist;

        clear fname dir2save subject condition sensor filelist

        task_idx = task_idx + 1;
    end
end
fprintf('Tasks for Testing across trials computed.\n');

%% Clean Start
%--------------------------------------------------------------------------
if delete_data

    % Loop through tasklist and delete files
    %---------------------------------------
    all_files = vertcat(tasks_audio.filelist, ...
                        tasks_preprocessing.filelist, ...
                        tasks_training.filelist, ...
                        tasks_testing.filelist, ...
                        tasks_across_trials.filelist);

    n_files = length(all_files);
    fprintf('Found %d files to delete...\n', n_files);

    for file_idx = 1:n_files
        % Get the current file path from the cell array
        current_file = all_files{file_idx};
        
        % Check if the file exists before attempting to delete it
        if isfile(current_file)
            delete(current_file);
            fprintf('Deleted: %s\n', current_file);
        else
            fprintf('Skipped (not found): %s\n', current_file);
        end
    end

    fprintf('\nDeletion process complete.\n');

end

%% Update all tasks
%--------------------------------------------------------------------------
tasks_audio         = get_tasks_to_process(tasks_audio, recompute_all);
tasks_preprocessing = get_tasks_to_process(tasks_preprocessing, recompute_all);
tasks_training      = get_tasks_to_process(tasks_training, recompute_all);
tasks_testing       = get_tasks_to_process(tasks_testing, recompute_all);
tasks_across_trials = get_tasks_to_process(tasks_across_trials, recompute_all);

%% Analysis
%--------------------------------------------------------------------------

% (1) Preprocessesing Audio
%--------------------------------------------------------------------------
% only a single task

if computation_status.preprocessing_audio

    % Preprocessing of olsa envelopes
    preprocessing_audio_olsa(settings)

    % Preprocessing of audiobook envelopes
    preprocessing_audio_audiobooks(settings)

end

% (2) Preprocessing MEEG
%--------------------------------------------------------------------------

if computation_status.preprocessing_meeg

    % Execute Jobs in Parallel
    n_tasks        = length(tasks_preprocessing);
    max_workers    = parcluster('local').NumWorkers;
    workers_to_use = min([n_tasks, max_workers, max_worker2use.preprocessing_meeg]);

    if workers_to_use > 0

        parpool(workers_to_use);
        parfor task_idx = 1:n_tasks 

            % Reconstruct the full settings for the current job
            current_subject = tasks_preprocessing(task_idx).subject;
            current_session = tasks_preprocessing(task_idx).session;

            % Propecessing of audiobook MEG recordings 
            preprocessing_meeg_audiobooks_decoding(settings,current_subject,current_session)

            % Propecessing of olsa MEG recordings 
            preprocessing_meeg_olsa_decoding(settings,current_subject,current_session); % single sentences

        end % tasks

        % Cleanup
        %--------
        delete(gcp('nocreate')); % Close the parallel pool if it's running
        fprintf('Preprocessing MEEG completed for all jobs.\n');

    end % check workers
end % check computation

% (3) Training and Testing
%--------------------------------------------------------------------------

% Training
%---------
if computation_status.training_decoder

    % Execute Jobs in Parallel
    n_tasks        = length(tasks_training);
    max_workers    = parcluster('local').NumWorkers;
    workers_to_use = min([n_tasks, max_workers, max_worker2use.training_decoder]);

    if workers_to_use > 0

        parpool(workers_to_use);
        parfor task_idx = 1:n_tasks 

            % Reconstruct the full settings for the current job
            current_subject   = tasks_training(task_idx).subject;
            current_condition = tasks_training(task_idx).condition;

            % Training decoder
            %-----------------
            training_decoding(settings, current_subject, current_condition, settings.decoding.run_nested_cv) 

        end % tasks

        % Cleanup
        %--------
        delete(gcp('nocreate')); % Close the parallel pool if it's running
        fprintf('Training completed for all jobs.\n');

    end % check workers
end % check computation

% Testing
%--------
if computation_status.testing_decoder

    % Execute Jobs in Parallel
    n_tasks        = length(tasks_testing);
    max_workers    = parcluster('local').NumWorkers;
    workers_to_use = min([n_tasks, max_workers, max_worker2use.testing_decoder]);

    if workers_to_use > 0

        parpool(workers_to_use);
        parfor task_idx = 1:n_tasks 

            % Reconstruct the full settings for the current job
            current_subject   = tasks_testing(task_idx).subject;
            current_condition = tasks_testing(task_idx).condition;
    
            % Prediction OLSA - concatenated sentences
            if compute_concat_sentences
                prediction_concatenated_sentences_decoding(settings, current_subject, current_condition)
            end
            % Prediction OLSA - single sentences
            if compute_single_sentences
                prediction_single_sentences_decoding(settings, current_subject, current_condition)
            end
            % Prediction OLSA - concatenated sentences - olsa similarity measure
            if compute_concat_sentences_olsa_sim
                prediction_concatenated_sentences_olsa_similarity_decoding(settings, current_subject, current_condition)
            end

            % Prediction OLSA - single sentences - olsa similarity measure
            if compute_single_sentences_olsa_sim
                prediction_single_sentences_olsa_similarity_decoding(settings, current_subject, current_condition)
            end

        end % tasks

        % Cleanup
        %--------
        delete(gcp('nocreate')); % Close the parallel pool if it's running
        fprintf('Testing completed for all jobs.\n');

    end % check workers
end % check computation

% (4) Testing across number of concatenated trials
%--------------------------------------------------------------------------

if computation_status.testing_across_trials

    % Execute Jobs in Parallel
    n_tasks        = length(tasks_across_trials);
    max_workers    = parcluster('local').NumWorkers;
    workers_to_use = min([n_tasks, max_workers, max_worker2use.testing_across_trials]);

    if workers_to_use > 0

        parpool(workers_to_use);
        parfor task_idx = 1:n_tasks 

            % Reconstruct the full settings for the current job
            current_subject   = tasks_across_trials(task_idx).subject;
            current_condition = tasks_across_trials(task_idx).condition;

            % Prediction OLSA - concatenated sentences across trials
            prediction_concatenated_sentences_across_trials_decoding(settings, current_subject, current_condition)

        end % tasks

        % Cleanup
        %--------
        delete(gcp('nocreate')); % Close the parallel pool if it's running
        fprintf('Testing across trials completed for all jobs.\n');

    end % check workers
end % check computation

%% Stop timer and log the elapsed time
%--------------------------------------------------------------------------
elapsed_time = toc;
fprintf('\n--- Pipeline Finished ---\n');
fprintf('Total processing time: %.2f minutes.\n', elapsed_time / 60);

diary off; % Stop logging output

%% Visualization of results
%--------------------------------------------------------------------------
% Use 'plot_training_decoding.m' to visualize audiobook training results
% Use 'plot_prediction_single_sentences.m' to visualize prediction results on single sentences
% Use 'plot_prediction_concatenated_sentences.m' to visualize prediction results on concatenated sentences
