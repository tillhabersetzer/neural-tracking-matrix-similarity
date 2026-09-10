function tasks_to_process = get_tasks_to_process(tasks, recomputeAll)
%--------------------------------------------------------------------------
% Till Habersetzer, 05.07.2025
% Communication Acoustics, CvO University Oldenburg
% till.habersetzer@uol.de 
%
%   DESCRIPTION:
%   This utility function filters a list of analysis tasks to determine which
%   ones need to be processed. It enables an efficient 'skip-if-complete'
%   workflow by identifying tasks with missing output files.
%
%   KEY STEPS:
%   1.  **Check Recompute Flag:** If `recomputeAll` is true, the function
%       bypasses all checks and returns the full list of input tasks.
%   2.  **Identify Incomplete Tasks:** If `recomputeAll` is false, it iterates
%       through each task provided in the input array.
%   3.  **File Existence Check:** For each task, it checks if every file
%       specified in the `.filelist` field exists on disk. A task is
%       marked as incomplete if one or more files are missing.
%   4.  **Return Subset:** The function returns a struct array containing
%       only the tasks that were identified as incomplete.
%
%   SYNTAX:
%   tasks_to_process = get_tasks_to_process(tasks, recomputeAll)
%
%   INPUTS:
%   tasks (struct array):   A structure array where each element represents a
%                           single task. Must contain the field `.filelist`,
%                           which is a cell array of file paths.
%   recomputeAll (logical): A flag to control behavior. If true, all tasks
%                           are returned. If false (default), only tasks with
%                           missing files are returned.
%
%   OUTPUTS:
%   tasks_to_process (struct array): A subset of the input `tasks` struct
%                                    array containing only the tasks that
%                                    require processing.
%--------------------------------------------------------------------------

    if recomputeAll
        tasks_to_process = tasks;
        fprintf('RecomputeAll is true. Preparing to process all %d tasks.\n', numel(tasks));
        return;
    end

    n_tasks = numel(tasks);
    if n_tasks == 0
        tasks_to_process = []; 
        return;
    end

    idx_missing = false(n_tasks, 1);
    for task_idx = 1:n_tasks
        
        filelist = tasks(task_idx).filelist;
        
        for path2file = filelist(:)'
            if ~isfile(path2file)
                idx_missing(task_idx) = true;
                break; % Efficiently skip to the next task
            end
        end
    end
    
    % Use the logical index to select only the tasks marked as missing
    tasks_to_process = tasks(idx_missing); % CORRECTED: Was 'missing_tasks'

    num_missing = numel(tasks_to_process);
    if num_missing > 0
        fprintf('Found %d tasks that still need to be computed (missing files).\n', num_missing);
    else
        fprintf('All %d tasks are complete. No files are missing.\n', n_tasks);
    end

end